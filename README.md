# Ollama Chat

A small desktop app for chatting with AI models running **locally** through [Ollama](https://ollama.com). No account, no cloud, and **no Python dependencies to install** (just Tkinter and the standard library).

## Features

- Replies stream in as they are generated, with a **Stop** button
- Pick any model installed in Ollama, adjustable temperature
- Optional system prompt (e.g. "Always answer in French")
- Shows the "thinking" of reasoning models (Qwen3, DeepSeek R1, ...)
- Generation speed (tokens/s) shown after each reply
- Export the conversation to Markdown
- Works on Windows, macOS and Linux

## Requirements

1. **[Ollama](https://ollama.com/download)** installed and running.
2. **Python 3.8 or newer** with Tkinter:
   - Windows / macOS: the installer from [python.org](https://www.python.org/downloads/) includes it.
   - Linux (Debian/Ubuntu): `sudo apt install python3-tk`
3. At least one downloaded model, for example:

   ```bash
   ollama pull qwen3:8b
   ```

   Pick a model that fits your hardware: the model file should fit in your RAM / VRAM with some headroom.

## Run it

```bash
git clone https://github.com/ReynoxRNX/ollama-chat.git
cd ollama-chat
python ollama_chat.py        # or python3 on macOS / Linux
```

**In VS Code**: open the folder (`File > Open Folder...`), open `ollama_chat.py`, then click the ▶ button at the top right (or press `F5`). The Microsoft Python extension is recommended.

Usage: choose a model, type your message, press `Enter` to send, `Shift+Enter` for a new line, and `Esc` to stop a reply.

## Configuration

By default the app talks to `http://localhost:11434`. To use another Ollama server (another machine on your network, a different port), set the `OLLAMA_HOST` environment variable before launching:

```bash
OLLAMA_HOST=192.168.1.20:11434 python ollama_chat.py          # macOS / Linux
$env:OLLAMA_HOST="192.168.1.20:11434"; python ollama_chat.py   # PowerShell
```

## Troubleshooting

| Message | Likely cause |
| --- | --- |
| "Could not reach Ollama..." | Ollama is not running: open the Ollama app or run `ollama serve`, then click *Refresh*. |
| "No model installed" | Download one with `ollama pull <name>`. |
| `ModuleNotFoundError: tkinter` | Install Tkinter (see Requirements). |
| Very slow replies | The model is too big for your machine: try a smaller one. |

## Known limitations

- Text is shown as plain text: no Markdown rendering (headings, bold, code highlighting).
- Text only: no image or file attachments.
- Ollama truncates the history beyond the model's context window; for very long conversations, start a new one.

## Project layout

```
ollama_chat.py      Tkinter interface
ollama_client.py    HTTP client for the Ollama API (testable without the UI)
tests/              unit tests (fake Ollama server, no model required)
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m doctest ollama_client.py
```

## Contributing

Issues and pull requests are welcome. Please run the tests before submitting a change.

## License

[MIT](LICENSE)
