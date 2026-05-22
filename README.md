# Desktop-Helper
A Python desktop assistant powered by OpenAI. Helps with daily productivity, scheduling, and information at a glance.

> Test in a VM until finalized.

## Planned Features
- **Chat interface** — conversational assistant via OpenAI GPT-4o (Textual TUI)
- **Calendar & notes** — scheduling integration via macOS EventKit (pyobjc)
- **App launcher** — open allowed apps by name ("Open League Client")
- **Permission system** — user-managed allowlist for apps and API access
- **Voice input/output** — faster-whisper (local transcription) + OpenAI TTS
- **Startup briefing** — weather report + daily news on launch
- **Daily summary** — upcoming events, tasks, and highlights
- **Persistent memory** — conversation logs via ChromaDB for personalization
- **Stock/web data** — market data via yfinance, optional web scraping
- **Hotkeys & menubar widget** — quick access via pynput + rumps

## Stack
| Purpose | Library |
|---|---|
| AI brain | OpenAI GPT-4o |
| UI | Textual |
| Voice in | faster-whisper |
| Voice out | OpenAI TTS API |
| Memory | ChromaDB |
| Calendar | pyobjc EventKit |
| Stocks | yfinance |
| Hotkeys | pynput |
| Menubar | rumps |
