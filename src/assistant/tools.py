"""
Tool handlers for the dispatcher.

Each handler takes the user's message and returns a result string that gets
injected as `[RESULT]...[/RESULT]`. The model has already chosen the tool; the
handler parses the message to pick the specific action within that tool.

Only Spotify is wired to a real backend so far (it needs no API key). The other
eight tools are registered as stubs so the dispatch loop works end-to-end; they
get replaced with real implementations as Phase 4 continues.
"""
from src.spotify import spotify


def spotify_handler(message: str) -> str:
    m = message.lower()
    if "pause" in m or "stop" in m:
        spotify.pause()
        return "Paused"
    if "skip" in m or "next" in m:
        spotify.next_track()
        return f"Now playing: {spotify.current_track()}"
    if "previous" in m or "go back" in m or "last song" in m:
        spotify.previous_track()
        return f"Now playing: {spotify.current_track()}"
    if "volume up" in m or "turn up" in m or "louder" in m:
        spotify.set_volume(85)
        return "Volume set to 85%"
    if "volume down" in m or "turn down" in m or "quieter" in m or "lower" in m:
        spotify.set_volume(35)
        return "Volume set to 35%"
    if "resume" in m or "play" in m:
        spotify.play()
        return f"Now playing: {spotify.current_track()}"
    # default intent: report what's currently playing
    return spotify.current_track()


def _stub(tool: str):
    def handler(message: str) -> str:
        return f"{tool} is not wired up yet"
    return handler


# registry the dispatcher looks up by tool name (the value from is_tool_call)
HANDLERS = {
    "spotify": spotify_handler,
    "calendar": _stub("calendar"),
    "screen": _stub("screen"),
    "reminders": _stub("reminders"),
    "notes": _stub("notes"),
    "launcher": _stub("launcher"),
    "weather": _stub("weather"),
    "news": _stub("news"),
    "stocks": _stub("stocks"),
}
