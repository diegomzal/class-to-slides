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

Results land in `output/<title>/`: `summary.md`, `prerequisites.md`, `slides.html/.pdf/.pptx`, `transcript.md`, `analysis.md`.
Run `python lecture2slides.py -h` for all options (audience, languages, vocabulary, models…).

**Cost:** about $1 per 1–2 h lecture with the default models on OpenRouter.

## Notes

- The transcript and selected frames are sent to the model provider you choose. Only process recordings you are allowed to use.
- Generated material can contain mistakes; timestamps let you check against the video.
- Frame selection can be tuned in `MODES` at the top of the script.

## License

[MIT](LICENSE)
