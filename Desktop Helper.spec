# PyInstaller spec — freezes the menu bar app into a self-contained
# "Desktop Helper.app" that needs neither uv nor the project source tree.
#
#   uv run pyinstaller "Desktop Helper.spec" --noconfirm
#
# The 3.4GB model checkpoint is deliberately NOT bundled — it stays under
# ~/Library/Application Support/Desktop Helper/ (see src/paths.py), so the .app
# stays lean and the weights can be swapped without a rebuild.
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# packages with dynamic imports / bundled data files PyInstaller can't infer
# from static analysis — collect their submodules, data, and libs wholesale
for pkg in ("transformers", "tokenizers", "chromadb", "faster_whisper",
            "openwakeword", "rumps", "yfinance"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# pyobjc frameworks are imported by name (some lazily, inside functions), so
# static analysis misses them — list every framework the runtime touches
hiddenimports += [
    "AppKit", "Foundation", "EventKit", "Quartz", "Vision",
    "objc", "PyObjCTools", "PyObjCTools.AppHelper",
]

# our own read-only assets (must match src.paths.resource_path layout)
datas += [
    ("model/hf_tokenizer", "model/hf_tokenizer"),
    ("config.example.json", "."),
    ("assets/AppIcon.icns", "assets"),
]

# training-only, dev-only, TUI, and never-installed ML backends — excluded to
# keep the bundle from ballooning (torch is bundled via PyInstaller's own hook)
excludes = [
    "datasets", "tensorboard", "pytest", "textual", "matplotlib",
    "tensorflow", "jax", "flax",
]

INFO_PLIST = {
    "CFBundleName": "Desktop Helper",
    "CFBundleDisplayName": "Desktop Helper",
    "CFBundleIdentifier": "com.kai.desktop-helper",
    "CFBundleShortVersionString": "0.2.0",
    "CFBundleVersion": "0.2.0",
    "LSMinimumSystemVersion": "13.0",
    "LSUIElement": True,  # menu bar app: no Dock icon, no app-switcher entry
    "NSHighResolutionCapable": True,
    # TCC reads these from the responsible app's Info.plist; without them macOS
    # silently refuses to even prompt (the calendar-permission saga)
    "NSCalendarsFullAccessUsageDescription":
        "Desktop Helper reads today's events to answer questions about your calendar.",
    "NSCalendarsUsageDescription":
        "Desktop Helper reads today's events to answer questions about your calendar.",
    "NSRemindersFullAccessUsageDescription":
        "Desktop Helper reads your reminders to answer questions about your to-dos.",
    "NSRemindersUsageDescription":
        "Desktop Helper reads your reminders to answer questions about your to-dos.",
    "NSMicrophoneUsageDescription":
        "Desktop Helper records while you hold the hotkey so it can transcribe "
        "your question locally. Audio never leaves this Mac.",
    "NSAppleEventsUsageDescription":
        "Desktop Helper controls Spotify and other apps via AppleScript.",
}

a = Analysis(
    ["src/menubar/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesktopHelper",
    console=False,          # GUI/menu-bar app — no terminal window
    target_arch="arm64",    # Apple Silicon only (matches the thin launcher)
    codesign_identity=None,  # ad-hoc signed in COLLECT/BUNDLE post-step
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="DesktopHelper",
)

app = BUNDLE(
    coll,
    name="Desktop Helper.app",
    icon="assets/AppIcon.icns",
    bundle_identifier="com.kai.desktop-helper",
    info_plist=INFO_PLIST,
)
