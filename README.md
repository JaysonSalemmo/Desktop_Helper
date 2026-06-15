# Desktop Helper

A conversational desktop assistant powered by an original language model trained from scratch. The model is a GPT-style transformer (~350M parameters) built and trained in PyTorch — not a fine-tune, not an API wrapper.

## How It Works

The model handles language (understanding your question, forming a response). Live data — your calendar, files, screen — is fetched at query time via tools the model learns to call during training.

```
You:       "What's on my calendar today?"
Model:     [CALL: calendar]
App:       → reads calendar → returns events
Model:     "You have standup at 9am and lunch at 12pm."
```

## Features

- **Conversational chat** — natural Q&A via a Textual TUI
- **Calendar** — reads today's events via macOS EventKit
- **Screen capture** — takes a screenshot and describes what's on screen
- **Reminders & notes** — read and create reminders/notes
- **App launcher** — open allowed apps by name
- **Spotify control** — play, pause, skip
- **Startup briefing** — weather, news, and calendar summary on launch
- **Voice input** — local transcription via faster-whisper
- **Persistent memory** — conversation history via ChromaDB
- **Hotkeys & menubar** — quick access from anywhere on the desktop

## Model

| Property | Value |
|---|---|
| Architecture | GPT-style decoder-only transformer |
| Parameters | ~350M |
| Framework | PyTorch |
| Tool calling | Special tokens (`[CALL: <tool>]` / `[RESULT: ...]`) |
| Training hardware | NVIDIA RTX GPU (CUDA) |
| Inference | Mac Apple Silicon (MPS) or NVIDIA (CUDA) |

## Stack

| Purpose | Library |
|---|---|
| Model | PyTorch + torch.compile |
| Training utilities | HuggingFace Accelerate |
| UI | Textual |
| Voice input | faster-whisper |
| Memory | ChromaDB |
| Calendar / Reminders | pyobjc EventKit |
| Stocks | yfinance |
| Hotkeys | pynput |
| Menubar | rumps |

## Setup

```bash
cp config.example.json config.json
# Edit config.json with your name and preferences
uv sync
uv run python -m src.main
```

## Project Structure

```
src/
  assistant/          # Model inference and tool dispatch
  calendar_integration/
  screen_capture/
  reminders/
  notes/
  spotify/
  launcher/
  memory/
  ui/                 # Textual TUI
  config/
model/                # Transformer architecture and training (coming)
```
