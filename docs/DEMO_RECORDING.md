# Recording a 1–2 minute demo of the chat UI (Windows)

A short video or GIF of the chat page for the README, LinkedIn or a portfolio. Everything runs locally.

## 1. Prepare (5 minutes)

1. Plug the laptop in, set **Settings → System → Power → Power mode → Best performance**, and close heavy apps.
   Replies take longer when the GPU throttles. Check before recording; the temperature should be well below
   80 °C and `SW Thermal Slowdown` should read `Not Active`:
   ```powershell
   nvidia-smi --query-gpu=temperature.gpu --format=csv
   nvidia-smi -q -d PERFORMANCE | Select-String "Thermal Slowdown"
   ```
2. Start the app on the real model (Ollama must be running):
   ```powershell
   $env:MOCK_LLM = "false"; $env:LLM_PROVIDER = "ollama"
   .venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
   ```
   Or use the full stack: `docker compose up` with `MOCK_LLM=false` and `LLM_PROVIDER=ollama` in `.env`. The image
   limits each visitor to 6 questions a minute, so space the questions out or set `RATE_LIMIT_PER_MINUTE=0` in
   `.env` for the recording.
3. Open http://127.0.0.1:8000 in Edge or Chrome:
   - make the window 1280×720 or 1440×900, at 100 % zoom;
   - hide the bookmarks bar (Ctrl+Shift+B);
   - turn on **Do not disturb** (Win+N) so notifications don't appear.
4. **Warm up:** ask one sample question first so the model and the embedding index are loaded, then press
   **New conversation**. The first answer after start-up is the slowest.

## 2. Record

Pick one tool:

- **Snipping Tool** (built in): open it, switch to **Record** (the video-camera icon) or press **Win+Shift+R**, drag
  over the browser window, then **Start**. Stop it from the toolbar; it saves an MP4.
- **Xbox Game Bar** (built in): focus the browser, then **Win+Alt+R** starts and stops recording. Files go to
  `Videos\Captures`.
- **OBS Studio** (free) if you want a webcam overlay or a fixed 1080p canvas.

## 3. Suggested script (about 1:45 before trimming)

The sample questions are clickable buttons, in the left sidebar on wide screens and above the message box on narrow
ones.

| Time | Action | What to point out |
|---|---|---|
| 0:00 | Page loaded, dark theme | The **Ollama · qwen3:4b** badge, and the note that the policies are simulated and the orders fictional |
| 0:08 | Click **Policy: How long do I have to return shoes?** | Live steps: Checking safety → Searching policies → Composing → Reviewing; the answer cites **KB-01** |
| 0:25 | Open **How this answer was made** | Sources, tools (`rag_lookup`), groundedness passed, AutoGen review approved, latency, tokens |
| 0:35 | Click **Order lookup: ORD-1016** | "Looking up order" appears; status, value and the escalation decision come from the order tool |
| 0:55 | Click **Multi-part: ORD-1029 …** | Two separately cited policies (KB-01, KB-12) plus the order facts, all in one reply |
| 1:15 | Click **No order ID** | The agent asks for the order ID instead of guessing |
| 1:25 | Click **Prompt injection** | Blocked by the guardrail before any model call: the panel shows the category and 0 tokens |
| 1:35 | Click **Out of scope** | It declines rather than inventing an answer |
| 1:45 | Toggle the **light theme**, then **New conversation** | Same layout in both themes |

**Waits:** real answers take a while on a laptop GPU (see `evaluation/V2_PROVIDER_COMPARISON.md` for measured
latency). Shorten the waits when editing, but keep a second of the progress steps on screen. If you speed up or cut
the waits, say so in a caption ("waits shortened"), so the video doesn't suggest the model is faster than it is.

## 4. Edit and export

- **Clipchamp** (built in on Windows 11): import the MP4, split and delete the long waits, and add short captions
  (for example "Grounded in KB-01" or "Injection blocked"). Export 1080p MP4.
- **GIF for the README:** use **ScreenToGif** (`winget install NickeManarin.ScreenToGif`), which records a region
  directly or imports the MP4. Set 10–12 fps and about 960 px wide. Or use ffmpeg:
  ```powershell
  ffmpeg -i demo.mp4 -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 demo.gif
  ```
  Keep the GIF under about 10 MB.
- **Where to put it:** a large video doesn't belong in git. For the README, drag the MP4 into a GitHub issue or pull
  request comment; GitHub hosts it and gives a link you can paste into the README. A small GIF can go in
  `docs/screenshots/`.

## 5. Before you publish

- The badge must say what really answered (Ollama or Mock).
- Use only the sample questions or other fictional data: no real names, phone numbers or order details.
- The policies are simulated, not official Nykaa policy. Keep the on-screen note visible at least once.
