#!/usr/bin/env python3
"""
lecture2slides — turn the video(s) of a lecture into a study pack:
transcript, summary, prerequisites and a slide deck (HTML / PDF / PPTX).

Works for any subject and any spoken language.

Usage:
    python lecture2slides.py "Lecture 3 - Supply and demand" lecture3.mp4
    python lecture2slides.py "Lecture 5 - Spin" part1.mp4 part2.mp4 --mode mixed \
        --subject "quantum mechanics" --audience "engineering student, strong in linear algebra"

Pipeline (each step is skipped if its output already exists, so re-runs are cheap):
  1. Extract audio (ffmpeg)
  2. Transcribe locally (mlx-whisper on Apple Silicon, faster-whisper elsewhere)
  3. Sample frames, remove the lecturer with a temporal median, keep the fullest board/slide
  4. Analyze transcript + frames (vision model)                    -> analysis.md
  5-7. Summary, prerequisites and Marp slides (text model)         -> *.md
  8. Render slides to HTML, PDF and PPTX (Marp CLI)

Setup: see README.md
"""

import argparse
import base64
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------- Configuration ----------------
WHISPER_MODEL_MLX = "mlx-community/whisper-large-v3-turbo"   # Apple Silicon
WHISPER_MODEL_CT2 = "large-v3-turbo"                         # faster-whisper (Linux / Windows / Intel Mac)
DEFAULT_AUDIENCE = "a motivated learner who is new to this subject"
MAX_FRAMES = 80
ANALYSIS_WIDTH = 480   # frames are analyzed downscaled; only the selected ones are saved full-size
API_IMAGE_SIDE = 1280  # longest side of frames sent to the model
API_IMAGE_BUDGET = 12_000_000  # total bytes of images per request (providers cap request size ~20 MB)

PROVIDERS = {
    # name: (base_url, env var, analysis model, text model)
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                   "~google/gemini-pro-latest", "~anthropic/claude-opus-latest"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               "gemini-pro-latest", "gemini-pro-latest"),
}

# interval:  seconds between samples
# window:    samples in the temporal median (odd). Removes the lecturer walking in front of the
#            board; 1 = off. Larger removes people better but lags behind fast changes.
# threshold: perceptual-hash distance that counts as a scene change (new slide, camera switch).
#            None = off: a lecturer moving in front of a sparse board looks like a scene change,
#            so pure-whiteboard lectures rely on erase detection only.
# drop:      relative drop in "ink" that counts as an erase (0.25 = 25% of the writing gone)
# max_gap:   force an intermediate capture after this many seconds without one
MODES = {
    "whiteboard": dict(interval=5, window=9, threshold=None, drop=0.25, max_gap=180, width=1600,
        hint="The video shows only the lecturer standing in front of a whiteboard or blackboard, "
             "writing, erasing and talking; there are no slides. Frames were cleaned with a temporal "
             "median to remove the lecturer, so faint ghosting or partly hidden areas are possible. "
             "The transcript is the main source of reasoning; use the frames to reconstruct what was "
             "written (formulas, diagrams, lists, code). If something appears in several frames, use "
             "the most complete version and cross-check it with what is said. Mark anything illegible "
             "instead of guessing."),
    "slides": dict(interval=4, window=1, threshold=8, drop=0.25, max_gap=None, width=1280,
        hint="The frames are slides; their text, figures and formulas are reliable. The most valuable "
             "part is what the lecturer SAYS that is not on the slide (examples, clarifications, "
             "warnings); capture it explicitly."),
    "mixed": dict(interval=5, window=5, threshold=16, drop=0.25, max_gap=120, width=1600,
        hint="The video switches between layouts: (a) the lecturer writing and erasing on a board, "
             "(b) slides, and (c) a split screen with a slide on one side and the board on the other. "
             "Say which layout each frame shows. Slides give the structure; the board usually holds "
             "worked examples and derivations, which are the most valuable. Frames were cleaned with "
             "a temporal median to remove the lecturer, so faint ghosting is possible. For board "
             "content, cross-check with the transcript and mark anything illegible instead of guessing."),
}
# ------------------------------------------------


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def fmt_ts(seconds):
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def slugify(text):
    return re.sub(r"[^\w]+", "-", text.strip().lower()).strip("-") or "lecture"


# ---------- 1. Audio ----------
def extract_audio(video, out_wav):
    if not out_wav.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", out_wav])


# ---------- 2. Transcription ----------
def whisper_prompt(ctx):
    parts = []
    if ctx["subject"]:
        parts.append(f"Lecture on {ctx['subject']}.")
    if ctx["vocab"]:
        parts.append(ctx["vocab"].strip().rstrip(".") + ".")
    return " ".join(parts) or None


def transcribe(wav, out_json, ctx):
    if out_json.exists():
        return json.loads(out_json.read_text())
    prompt = whisper_prompt(ctx)
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if apple_silicon:
        import mlx_whisper
        print(f"  transcribing {wav.name} with mlx-whisper (first run downloads the model, ~1.5 GB)…")
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODEL_MLX,
                                   language=ctx["lecture_language"], initial_prompt=prompt)
        language, segs = r.get("language"), r["segments"]
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segs]
    else:
        from faster_whisper import WhisperModel
        print(f"  transcribing {wav.name} with faster-whisper (first run downloads the model)…")
        model = WhisperModel(WHISPER_MODEL_CT2, device="auto", compute_type="default")
        segs, info = model.transcribe(str(wav), language=ctx["lecture_language"],
                                      initial_prompt=prompt, vad_filter=True)
        segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
        language = info.language
    data = {"language": language, "segments": segments}
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


# ---------- 3. Frames ----------
def extract_raw_frames(video, part, frames_dir, cfg):
    raw_dir = frames_dir / f"_raw_p{part}_{cfg['interval']}s_{cfg['width']}w"
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-vf", f"fps=1/{cfg['interval']},scale={cfg['width']}:-2", "-q:v", "3", raw_dir / "%05d.jpg"])
    return sorted(raw_dir.glob("*.jpg"))


def _window(i, n, size):
    k = size // 2
    lo, hi = max(0, i - k), min(n, i + k + 1)
    return lo, hi


def _ink(gray):
    """Fraction of pixels on an edge: grows as the board fills up, drops when it is erased.
    Works for dark-on-white and chalk-on-black boards."""
    import numpy as np
    g = gray.astype(np.int16)
    gx = np.abs(np.diff(g, axis=1))[:-1, :] > 25
    gy = np.abs(np.diff(g, axis=0))[:, :-1] > 25
    return float((gx | gy).mean())


def select_frames(raws, cfg):
    """Return the indices of the frames to keep.

    1. Downscale every sample and take a per-pixel temporal median over `window` samples:
       the board is static, the lecturer moves, so the median mostly removes the lecturer.
       With an odd window, the median also doesn't blend across a camera switch.
    2. Split the timeline into states at scene changes (perceptual hash) or erases (ink drop).
    3. From each state keep the frame with the most ink: the fullest board / last slide build.
    """
    import imagehash
    import numpy as np
    from PIL import Image

    n = len(raws)
    if n == 0:
        return []

    def load(p):
        im = Image.open(p)
        im.draft("L", (ANALYSIS_WIDTH, ANALYSIS_WIDTH))
        im = im.convert("L")
        im.thumbnail((ANALYSIS_WIDTH, ANALYSIS_WIDTH * 2))
        return np.asarray(im)

    small = np.stack([load(p) for p in raws])
    if cfg["window"] > 1:
        clean = np.stack([np.median(small[slice(*_window(i, n, cfg["window"]))], axis=0).astype(np.uint8)
                          for i in range(n)])
    else:
        clean = small
    hashes = [imagehash.phash(Image.fromarray(c)) for c in clean] if cfg["threshold"] else None
    ink = [_ink(c) for c in clean]

    picks, last_pick, peak = [], 0, 0

    def pick(i):
        nonlocal last_pick
        if picks and cfg["threshold"] and i - picks[-1] <= 2 and hashes[i] - hashes[picks[-1]] < 4:
            picks[-1] = i                   # practically the same image as the previous pick
        else:
            picks.append(i)
        last_pick = i

    for i in range(1, n):
        scene_change = bool(cfg["threshold"]) and hashes[i] - hashes[i - 1] >= cfg["threshold"]
        erased = ink[peak] > 0.005 and ink[i] < ink[peak] * (1 - cfg["drop"])
        if scene_change or erased:
            pick(peak)                      # fullest frame of the state that just ended
            peak = i
            continue
        if ink[i] >= ink[peak]:
            peak = i
        if cfg["max_gap"] and (i - last_pick) * cfg["interval"] >= cfg["max_gap"]:
            pick(i)
    pick(peak)
    return sorted(set(picks))


def save_frame(raws, i, cfg, dest):
    """Save a selected frame at full resolution, applying the same temporal median."""
    if cfg["window"] <= 1:
        shutil.copy(raws[i], dest)
        return
    import numpy as np
    from PIL import Image
    # backward-looking window: the frames *before* i show the same board (nothing erased yet)
    lo, hi = max(0, i - cfg["window"] + 1), i + 1
    stack = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in raws[lo:hi]])
    Image.fromarray(np.median(stack, axis=0).astype(np.uint8)).save(dest, quality=90)


def build_frames(videos, frames_dir, cfg, keep_raw=False):
    index_file = frames_dir / "frames.json"
    if index_file.exists():
        return json.loads(index_file.read_text())
    raw_by_part = [(part, extract_raw_frames(v, part, frames_dir, cfg))
                   for part, v in enumerate(videos, start=1)]
    selected = []
    for part, raws in raw_by_part:
        print(f"   selecting frames for part {part} ({len(raws)} samples)…")
        selected += [(part, raws, i) for i in select_frames(raws, cfg)]
    if len(selected) > MAX_FRAMES:  # even sampling if there are still too many
        step = len(selected) / MAX_FRAMES
        selected = [selected[int(j * step)] for j in range(MAX_FRAMES)]
    kept = []
    for part, raws, i in selected:
        t = i * cfg["interval"]
        name = f"p{part}_{fmt_ts(t).replace(':', '-')}.jpg"
        save_frame(raws, i, cfg, frames_dir / name)
        kept.append({"file": name, "part": part, "t": t, "ts": fmt_ts(t)})
    index_file.write_text(json.dumps(kept, indent=1))
    if not keep_raw:  # raw samples take ~250 MB per hour of video
        for d in frames_dir.glob("_raw_*"):
            shutil.rmtree(d)
    return kept


def encode_images(paths):
    """JPEG-encode frames for the API, shrinking them until the whole set fits the request budget."""
    import io
    from PIL import Image
    for side, quality in ((API_IMAGE_SIDE, 85), (API_IMAGE_SIDE, 70), (1024, 70), (896, 60), (768, 55)):
        out = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            out.append(buf.getvalue())
        if sum(map(len, out)) * 4 / 3 <= API_IMAGE_BUDGET:
            break
    return [base64.b64encode(b).decode() for b in out]


# ---------- LLM (OpenAI-compatible API: OpenRouter or Google) ----------
def ask_llm(client, model, content, max_tokens, attempts=3):
    import time
    import openai
    for attempt in range(1, attempts + 1):
        print(f"  → {model} ", end="", flush=True)
        try:
            stream = client.chat.completions.create(
                model=model, max_tokens=max_tokens, stream=True,
                messages=[{"role": "user", "content": content}])
            out, finish = [], None
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    out.append(choice.delta.content)
                    if len(out) % 50 == 0:
                        print(".", end="", flush=True)
                finish = choice.finish_reason or finish
            print()
            break
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            print(f"\n  ! {type(e).__name__}: {e}")
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)
    text = "".join(out)
    if not text.strip():
        sys.exit(f"Model {model} returned an empty response.")
    if finish == "length":
        print(f"  ! the response hit the {max_tokens}-token limit and may be cut off")
    return text


def build_transcript_md(transcripts):
    lines = []
    for part, data in transcripts:
        lines.append(f"\n## Part {part} (detected language: {data.get('language')})\n")
        lines += [f"[P{part} {fmt_ts(s['start'])}] {s['text']}" for s in data["segments"]]
    return "\n".join(lines)


def context_block(title, cfg, ctx):
    lines = [f'Lecture title: "{title}"']
    if ctx["subject"]:
        lines.append(f"Subject: {ctx['subject']}")
    lines.append(f"Lecture format: {cfg['hint']}")
    return "\n".join(lines)


# ---------- 4. Analysis (with images) ----------
def analyze(client, model, title, cfg, ctx, transcript_md, frames, frames_dir, out_md):
    if out_md.exists():
        return out_md.read_text()
    print(f"  analyzing ({len(frames)} frames)")
    content = [{"type": "text", "text":
        f"You will analyze a recorded lecture.\n{context_block(title, cfg, ctx)}\n\n"
        "Below are frames from the video, each preceded by its file name and timestamp, followed by "
        "the full transcript (automatic speech recognition; technical terms and names may be "
        "misheard) with [Part time] markers."}]
    for f, b64 in zip(frames, encode_images([frames_dir / f["file"] for f in frames])):
        content.append({"type": "text", "text": f"Frame `{f['file']}` (Part {f['part']}, {f['ts']}):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": f"""
<transcript>
{transcript_md}
</transcript>

Write an analysis in {ctx['output_language']}, in Markdown, with these sections:

1. **Topic map**: ordered list of topics with their start timestamp [P# mm:ss].
2. **Key content**: every important definition, formula (in LaTeX, $...$ / $$...$$), diagram, list,
   code snippet or worked example, with its frame and timestamp. If something is illegible or
   ambiguous, say so; do not invent.
3. **Line of argument**: how each idea leads to the next, including steps the lecturer skipped or
   treated as obvious.
4. **Useful frames**: table `file | what it shows | what it is useful for`. Only frames that add
   something (diagrams, a full board, a key slide); ignore duplicates, blurry or covered ones.
5. **Transcript corrections**: terms or names the speech recognition probably got wrong.
6. **Likely points of confusion**: what will probably be hard to understand, and why.
"""})
    text = ask_llm(client, model, content, max_tokens=16000)
    out_md.write_text(text)
    return text


# ---------- 5-7. Summary, prerequisites, slides ----------
def generate(client, model, title, cfg, ctx, transcript_md, analysis, out_dir):
    targets = {k: out_dir / f"{k}.md" for k in ("summary", "prerequisites", "slides")}
    if all(p.exists() for p in targets.values()):
        return
    prompt = f"""{context_block(title, cfg, ctx)}
Audience: {ctx['audience']}

<analysis>
{analysis}
</analysis>

<transcript>
{transcript_md}
</transcript>

Produce three documents in {ctx['output_language']}, each inside its XML tag. Write for the audience
above: explain jargon the first time it appears and build intuition before formal detail.

<summary>
Lecture summary in Markdown: the central idea, each topic with an intuitive explanation followed by
the precise one, key results or takeaways, and [P# mm:ss] timestamps to jump back to the video.
</summary>

<prerequisites>
Prior knowledge needed to follow this lecture, in Markdown, from most basic to most advanced. For
each item: what it is in 1-2 sentences, where THIS lecture relies on it, and a quick self-check
(one short question with its answer). Only include what the audience may not already know.
</prerequisites>

<slides>
A Marp Markdown presentation. Start with exactly this front matter:
---
marp: true
math: mathjax
paginate: true
size: 16:9
---
Separate slides with `---`. Structure: title slide, prerequisites (brief), one block per topic
(intuition → precise explanation → example), step-by-step walkthroughs where the lecturer skipped
steps, a final summary and 3-5 review questions. At most ~8 lines per slide. Rewrite formulas in
LaTeX and code in code blocks instead of pasting photos of the board; use images only for diagrams
or figures, with `![h:380](frames/NAME.jpg)`, and ONLY names listed in the useful-frames table.
</slides>
"""
    text = ask_llm(client, model, [{"type": "text", "text": prompt}], max_tokens=32000)
    for key, path in targets.items():
        m = re.search(rf"<{key}>\s*(.*?)\s*</{key}>", text, re.S)
        if not m:
            (out_dir / "_raw_response.md").write_text(text)
            sys.exit(f"Could not find <{key}> in the response; see _raw_response.md")
        path.write_text(m.group(1).strip() + "\n")


# ---------- 8. Render ----------
def render_slides(out_dir):
    md = out_dir / "slides.md"
    for fmt in ("html", "pdf", "pptx"):   # html needs no browser; pdf/pptx need Chrome/Edge/Firefox
        out = out_dir / f"slides.{fmt}"
        if out.exists() and out.stat().st_mtime > md.stat().st_mtime:
            continue
        flag = [] if fmt == "html" else [f"--{fmt}"]   # html is Marp's default output
        try:
            run(["npx", "--yes", "@marp-team/marp-cli@latest", md, *flag,
                 "--allow-local-files", "--no-stdin", "-o", out])
        except subprocess.CalledProcessError:
            print(f"  ! could not render slides.{fmt} (is Chrome/Edge/Firefox installed?)")


DOWNSTREAM = {2: {2, 4, 5, 8}, 3: {3, 4, 5, 8}, 4: {4, 5, 8}, 5: {5, 8}, 8: {8}}


def invalidate(out_dir, frames_dir, audio_dir, steps):
    """Delete cached outputs of the given steps so they are regenerated."""
    if 2 in steps:
        for f in audio_dir.glob("part*.json"):
            f.unlink()
    if 3 in steps:
        for f in frames_dir.glob("p*_*.jpg"):
            f.unlink()
        (frames_dir / "frames.json").unlink(missing_ok=True)
    if 4 in steps:
        (out_dir / "analysis.md").unlink(missing_ok=True)
    if 5 in steps:
        for k in ("summary", "prerequisites", "slides"):
            (out_dir / f"{k}.md").unlink(missing_ok=True)
    if 8 in steps:
        for ext in ("html", "pdf", "pptx"):
            (out_dir / f"slides.{ext}").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Lecture video(s) → transcript, summary, prerequisites and slides.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # whiteboard-only lecture, check the selected frames first (free)
  python lecture2slides.py "Lecture 5" lecture5.mp4 --mode whiteboard --frames-only

  # full run, tailored to the reader
  python lecture2slides.py "Lecture 5 - Spin" part1.mp4 part2.mp4 --mode mixed \\
      --subject "quantum mechanics" --audience "engineering student, little physics"

  # only regenerate the documents with another model
  python lecture2slides.py "Lecture 5 - Spin" part1.mp4 part2.mp4 --mode mixed \\
      --text-model "~google/gemini-pro-latest"

Changing an option re-runs only the steps that depend on it; everything else is cached.
See README.md for setup.""")
    ap.add_argument("title", help='lecture name; also names the output folder, e.g. "Lecture 3 - Supply and demand"')
    ap.add_argument("videos", nargs="+", help="one or more video files of the SAME lecture, in order")

    g = ap.add_argument_group("content")
    g.add_argument("--mode", choices=MODES, default="mixed", help=
        "how the video looks. whiteboard: only a board with the lecturer writing/erasing in front; "
        "slides: only slides; mixed: board, slides and/or split screen (default: mixed)")
    g.add_argument("--subject", help=
        'topic of the course, e.g. "organic chemistry". Helps the transcription spell terms '
        "correctly and focuses the explanations")
    g.add_argument("--audience", default=DEFAULT_AUDIENCE, help=
        "who the material is for, in plain words (background, what they already know, what they "
        "lack). Controls the depth of the explanations and which prerequisites are listed "
        f'(default: "{DEFAULT_AUDIENCE}")')
    g.add_argument("--vocab", help=
        "comma-separated names/terms the transcription should spell right, "
        'e.g. "Schrödinger, Hamiltonian, Hilbert space"')

    g = ap.add_argument_group("language")
    g.add_argument("--lecture-language", metavar="CODE", help=
        "spoken language as a code: en, es, it, fr… (default: auto-detect)")
    g.add_argument("--output-language", metavar="LANG", help=
        'language of the generated material, e.g. "English" (default: same as the lecture)')

    g = ap.add_argument_group("models")
    g.add_argument("--provider", choices=PROVIDERS, default="openrouter", help=
        "openrouter (needs OPENROUTER_API_KEY) or google (needs GEMINI_API_KEY) (default: openrouter)")
    g.add_argument("--analysis-model", metavar="MODEL", help=
        "vision model that reads the frames and maps the lecture "
        f"(default: {PROVIDERS['openrouter'][2]} on openrouter)")
    g.add_argument("--text-model", metavar="MODEL", help=
        "model that writes summary, prerequisites and slides "
        f"(default: {PROVIDERS['openrouter'][3]} on openrouter)")

    g = ap.add_argument_group("run control")
    g.add_argument("-o", "--out", default="output", metavar="DIR", help="base output folder (default: output)")
    g.add_argument("--frames-only", action="store_true", help=
        "stop after selecting frames: free, no API calls. Use it to check that the mode fits the video")
    g.add_argument("--keep-raw", action="store_true", help=
        "keep all sampled frames and the extracted audio (~450 MB per hour of video) instead of "
        "deleting them once processed; useful when tuning MODES")
    g.add_argument("--from-step", type=int, choices=[2, 3, 4, 5, 8], metavar="N", help=
        "force a redo from step N: 2=transcript, 3=frames, 4=analysis, 5=documents, 8=render")
    args = ap.parse_args()

    import importlib.util
    needed = ["imagehash", "PIL", "numpy"] + ([] if args.frames_only else ["openai"])
    missing = [m for m in needed if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"Missing Python packages: {', '.join(missing)}\n"
                 f"This Python is {sys.executable}\n"
                 "Activate your virtualenv (source .venv/bin/activate) and run: pip install -r requirements.txt")
    for v in args.videos:
        if not Path(v).is_file():
            sys.exit(f"Video not found: {v}")
    for tool in ("ffmpeg",) + (() if args.frames_only else ("npx",)):
        if not shutil.which(tool):
            sys.exit(f"'{tool}' not found. See README.md for setup.")

    base_url, env_key, def_an, def_tx = PROVIDERS[args.provider]
    an_model, tx_model = args.analysis_model or def_an, args.text_model or def_tx
    cfg = MODES[args.mode]
    ctx = {"subject": args.subject, "audience": args.audience, "vocab": args.vocab,
           "lecture_language": args.lecture_language,
           "output_language": args.output_language or "the same language as the lecture"}

    out_dir = Path(args.out) / slugify(args.title)
    (audio_dir := out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (frames_dir := out_dir / "frames").mkdir(exist_ok=True)

    # If settings changed since the last run, redo only what depends on them
    meta_file = out_dir / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    current = {
        2: {"lecture_language": args.lecture_language, "subject": args.subject, "vocab": args.vocab},
        3: {"mode": args.mode},
        4: {"analysis_model": an_model, "subject": args.subject, "output_language": ctx["output_language"]},
        5: {"text_model": tx_model, "audience": args.audience},
    }
    redo = set()
    if args.from_step:
        redo |= {s for s in DOWNSTREAM if s >= args.from_step}
    for step, values in current.items():
        if args.frames_only and step != 3:
            continue
        if any(k in meta and meta[k] != v for k, v in values.items()):
            redo |= DOWNSTREAM[step]
    invalidate(out_dir, frames_dir, audio_dir, redo)
    for step, values in current.items():
        if not (args.frames_only and step != 3):
            meta.update(values)
    meta_file.write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    print(" 3. frames")
    frames = build_frames(args.videos, frames_dir, cfg, keep_raw=args.keep_raw)
    print(f"   {len(frames)} frames selected (mode: {args.mode}) → {frames_dir}/")
    if args.frames_only:
        return

    if not os.environ.get(env_key):
        sys.exit(f"Missing environment variable {env_key} (see README.md)")
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=os.environ[env_key])

    transcripts = []
    for part, video in enumerate(args.videos, start=1):
        wav, tjson = audio_dir / f"part{part}.wav", audio_dir / f"part{part}.json"
        if not tjson.exists():
            print(f" 1-2. audio + transcription, part {part}: {video}")
            extract_audio(video, wav)
        transcripts.append((part, transcribe(wav, tjson, ctx)))
        if not args.keep_raw:
            wav.unlink(missing_ok=True)
    transcript_md = build_transcript_md(transcripts)
    (out_dir / "transcript.md").write_text(f"# {args.title}\n{transcript_md}\n")

    print(" 4. analysis")
    analysis = analyze(client, an_model, args.title, cfg, ctx, transcript_md, frames, frames_dir,
                       out_dir / "analysis.md")
    print(" 5-7. summary / prerequisites / slides")
    generate(client, tx_model, args.title, cfg, ctx, transcript_md, analysis, out_dir)
    print(" 8. render")
    render_slides(out_dir)

    print(f"\nDone → {out_dir}/  (summary.md, prerequisites.md, slides.html/.pdf/.pptx)")


if __name__ == "__main__":
    main()
