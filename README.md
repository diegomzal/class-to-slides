# lecture2slides

Turn a recorded lecture into a study pack: **transcript, summary, prerequisites and slides** (HTML / PDF / PPTX).
Any subject, any spoken language; works with whiteboard lectures, slides, or both.

```
python lecture2slides.py "Lecture 3 - Supply and demand" lecture3.mp4 --mode slides
```

## How it works

1. Transcribes the audio **locally** with Whisper.
2. Picks the useful frames: the fullest board before each erase (with the lecturer removed) and one frame per slide.
3. A vision model reads the frames + transcript and maps the lecture.
4. A text model writes the summary, prerequisites and slides for your audience; [Marp](https://marp.app) renders them.

Every step is cached: changing an option only re-runs what depends on it.

## Install

Needs Python 3.10+, [ffmpeg](https://ffmpeg.org), [Node.js](https://nodejs.org), and Chrome/Edge/Firefox for PDF/PPTX.

```
# macOS: brew install ffmpeg node   ·   Ubuntu: sudo apt install ffmpeg nodejs npm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...      # or GEMINI_API_KEY with --provider google
```

Transcription uses `mlx-whisper` on Apple Silicon and `faster-whisper` elsewhere (slow without a GPU).

## Use

Pick the mode that matches the video:

| `--mode` | The video shows… |
|---|---|
| `whiteboard` | only a board, the lecturer writing and erasing in front of it |
| `slides` | only slides |
| `mixed` (default) | board, slides and/or a split screen |

Check the frames first (free, no API calls), then run it for real:

```
python lecture2slides.py "Test" lecture.mp4 --mode whiteboard --frames-only   # look at output/test/frames/
python lecture2slides.py "Test" lecture.mp4 --mode whiteboard \
    --subject "quantum mechanics" --audience "engineering student, little physics"
```

Results land in `output/<title>/` (see [Output](#output)).

## Options

```
python lecture2slides.py TITLE VIDEO [VIDEO ...] [options]
```

`TITLE` names the lecture and its output folder. Pass several `VIDEO` files if one lecture is split into parts, in order.
`python lecture2slides.py -h` shows the same list in the terminal.

**Content**

| Option | Default | What it does |
|---|---|---|
| `--mode` | `mixed` | How the video looks: `whiteboard`, `slides` or `mixed` (table above) |
| `--subject TEXT` | – | Course topic, e.g. `"organic chemistry"`. Helps the transcription spell terms right and focuses the explanations |
| `--audience TEXT` | a motivated learner new to the subject | Who the material is for: background, what they know, what they lack. Sets the depth of the explanations and which prerequisites are listed |
| `--vocab TEXT` | – | Comma-separated names/terms the transcription should spell right, e.g. `"Schrödinger, Hamiltonian"` |
| `--slide-images N` | `6` | About how many lecture frames to show in the slides. `0` = text only |

**Language**

| Option | Default | What it does |
|---|---|---|
| `--lecture-language CODE` | auto-detect | Spoken language: `en`, `es`, `it`, `fr`… |
| `--output-language LANG` | same as the lecture | Language of the generated material, e.g. `"English"` |

**Models**

| Option | Default | What it does |
|---|---|---|
| `--provider` | `openrouter` | `openrouter` (needs `OPENROUTER_API_KEY`) or `google` (needs `GEMINI_API_KEY`) |
| `--analysis-model MODEL` | `~google/gemini-pro-latest` | Vision model that reads the frames and maps the lecture |
| `--text-model MODEL` | `~anthropic/claude-opus-latest` | Model that writes the summary, prerequisites and slides |

Defaults are for OpenRouter; with `--provider google` both default to `gemini-pro-latest`. Any model name your provider accepts works.

**Run control**

| Option | Default | What it does |
|---|---|---|
| `-o DIR`, `--out DIR` | `output` | Base output folder |
| `--frames-only` | off | Stop after selecting frames. Free, no API calls: use it to check the mode fits the video |
| `--keep-raw` | off | Keep the extracted audio and all sampled frames (~450 MB per hour) instead of deleting them; useful when tuning `MODES` |
| `--from-step N` | – | Force a redo from step N: `2` transcript, `3` frames, `4` analysis, `5` documents, `8` render |

### What re-runs when you change an option

Every step is cached, and the script remembers the options of the last run. Changing one redoes only what depends on it:

| You change… | Redone | Approx. cost |
|---|---|---|
| `--audience`, `--slide-images`, `--text-model` | documents + render | ~$0.65 |
| `--analysis-model`, `--output-language` | analysis + documents + render | ~$0.95 |
| `--mode` | frames + analysis + documents + render | ~$0.95 |
| `--subject` | transcript + analysis + documents + render | ~$0.95 + transcription time |
| `--vocab`, `--lecture-language` | transcript + analysis + documents + render | ~$0.95 + transcription time |
| nothing | nothing (re-renders only if `slides.md` was edited) | free |

Edited `slides.md` by hand? Re-run the same command and only the render is redone.

## Output

```
output/<title>/
  summary.md          the lecture explained: intuition first, then the precise version, with timestamps
  prerequisites.md    what to know beforehand, each with a self-check question
  slides.md           Marp source of the deck (editable)
  slides.html/.pdf/.pptx
  transcript.md       timestamped transcript
  analysis.md         topic map, key content, skipped steps, likely points of confusion
  frames/             the selected frames
  audio/              transcript cache
  meta.json           options of the last run
```

The PPTX has each slide as an image (not editable); edit `slides.md` instead.

**Cost:** about $1 per 1–2 h lecture with the default models on OpenRouter.

## Notes

- The transcript and selected frames are sent to the model provider you choose. Only process recordings you are allowed to use.
- Generated material can contain mistakes; timestamps let you check against the video.
- Frame selection can be tuned in `MODES` at the top of the script.

## License

[MIT](LICENSE)
