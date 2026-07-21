"""Entry point for the menu bar app (source and frozen bundle).

``multiprocessing.freeze_support()`` MUST run before anything else. On macOS the
default 'spawn' start method re-executes this frozen binary for every worker
process (chromadb, torch, and faster-whisper all spawn them). Without this call
each spawned child falls through to ``main()`` and launches a whole new copy of
the app — an unbounded cascade of menu bar icons and windows. freeze_support()
makes a spawned child run its worker and exit instead of reaching ``main()``;
in the real parent process it's a no-op and startup continues normally.
"""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from src.menubar.app import main
    main()
