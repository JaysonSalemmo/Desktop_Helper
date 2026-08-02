"""
Build the FROZEN "Buddy.app" (self-contained — no uv, no source tree)
and seed its per-user data so it can actually run.

This supersedes scripts/build_app.py (the old thin launcher that just ran
`uv run` against the project). The freeze bundles the Python runtime + all deps
via PyInstaller; only the 3.4GB checkpoint and the editable config stay outside
the bundle, under ~/Library/Application Support/Desktop Helper/ (see
src/paths.py).

usage:
    uv run python scripts/freeze_app.py                 # build + seed data
    uv run python scripts/freeze_app.py --install        # also copy to /Applications
    uv run python scripts/freeze_app.py --symlink-model  # dev: link (not copy) the 3.4GB checkpoint
    uv run python scripts/freeze_app.py --no-build        # only (re)seed data

After building, GRANT PERMISSIONS to the new bundle (its identity changed):
System Settings → Privacy & Security → Calendars / Reminders / Microphone /
Accessibility / Screen & System Audio Recording. macOS prompts on first use.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "Buddy"
APP_SUPPORT = Path.home() / "Library" / "Application Support" / APP_NAME


def build() -> None:
    print("Building the frozen bundle (PyInstaller — a few minutes)…")
    subprocess.run(
        ["uv", "run", "pyinstaller", f"{APP_NAME}.spec", "--noconfirm"],
        cwd=PROJECT_ROOT, check=True,
    )


def seed_data(symlink_model: bool) -> None:
    """Put config.json + the checkpoint where the frozen app looks for them."""
    APP_SUPPORT.mkdir(parents=True, exist_ok=True)

    # config: copy the project's real config so the frozen app inherits it
    src_config = PROJECT_ROOT / "config.json"
    dst_config = APP_SUPPORT / "config.json"
    if src_config.exists():
        shutil.copy(src_config, dst_config)
        print(f"Seeded {dst_config}")
        checkpoint_rel = json.loads(src_config.read_text())["model"]["checkpoint"]
    else:
        print("No project config.json — the app will seed a template on first run.")
        return

    # checkpoint: the one big artifact that stays outside the bundle
    src_ckpt = PROJECT_ROOT / checkpoint_rel
    dst_ckpt = APP_SUPPORT / checkpoint_rel
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if dst_ckpt.exists() or dst_ckpt.is_symlink():
        print(f"Checkpoint already at {dst_ckpt} — leaving it.")
    elif not src_ckpt.exists():
        print(f"WARNING: checkpoint not found at {src_ckpt}; place it at {dst_ckpt}.")
    elif symlink_model:
        dst_ckpt.symlink_to(src_ckpt)
        print(f"Linked checkpoint → {dst_ckpt} (dev mode).")
    else:
        print(f"Copying 3.4GB checkpoint → {dst_ckpt} (once)…")
        shutil.copy(src_ckpt, dst_ckpt)
        print("Checkpoint copied.")


SIGN_IDENTITY = "Desktop Helper Signing"  # stable self-signed cert (scripts/make_signing_cert.sh)


def _signing_identity(stable: bool) -> str:
    """Ad-hoc ("-") by default; the stable self-signed identity only when
    explicitly requested (--stable-sign) AND present in the keychain.

    Ad-hoc is the safe default: it needs no keychain private key, so it never
    triggers a login-password prompt. Stable signing (opt-in) is what makes TCC
    grants persist across rebuilds — but codesign then needs the cert's private
    key, which macOS guards with a password prompt. Kept opt-in so a routine
    rebuild never surprises you with an auth dialog."""
    if not stable:
        return "-"
    found = subprocess.run(["security", "find-certificate", "-c", SIGN_IDENTITY],
                           capture_output=True)
    if found.returncode != 0:
        sys.exit(f"--stable-sign given but '{SIGN_IDENTITY}' isn't in the "
                 "keychain. Run scripts/make_signing_cert.sh first.")
    return SIGN_IDENTITY


def _sign_and_verify(app: Path, stable: bool) -> None:
    """Re-sign the whole bundle and HARD-FAIL if it doesn't validate.

    A bundle whose signature doesn't verify gets ALL its TCC permissions
    silently ignored by macOS — the grant checkboxes do nothing. So a valid
    signature is not optional (this is the copytree-broke-the-seal bug that
    cost us hours)."""
    identity = _signing_identity(stable)
    subprocess.run(["codesign", "--force", "--deep", "--sign", identity, str(app)],
                   check=True)
    verify = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)])
    if verify.returncode != 0:
        sys.exit("SIGNATURE INVALID after signing — TCC permissions will not "
                 "work. Aborting.")
    if identity == "-":
        print("Signed ad-hoc and verified ✓")
    else:
        print(f"Signed with '{identity}' and verified ✓  (TCC grants persist)")


def install(stable: bool) -> None:
    src = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    if not src.exists():
        sys.exit(f"{src} not found — build first (drop --no-build).")
    dst = Path("/Applications") / src.name
    if dst.exists():
        shutil.rmtree(dst)
    # ditto (NOT shutil.copytree) — copytree drops the code-signature seal, and
    # macOS then refuses every TCC permission for the installed copy
    subprocess.run(["ditto", str(src), str(dst)], check=True)
    _sign_and_verify(dst, stable)
    print(f"Installed {dst}")
    print("Add it to login items: System Settings → General → Login Items.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build + seed the frozen app.")
    parser.add_argument("--no-build", action="store_true", help="skip PyInstaller; only seed data")
    parser.add_argument("--install", action="store_true", help="copy the .app into /Applications")
    parser.add_argument("--symlink-model", action="store_true",
                        help="dev: symlink the checkpoint instead of copying 3.4GB")
    parser.add_argument("--stable-sign", action="store_true",
                        help="sign with the stable cert so TCC grants persist "
                        "across rebuilds (prompts for your login password; "
                        "needs scripts/make_signing_cert.sh). Default: ad-hoc.")
    args = parser.parse_args()

    if not args.no_build:
        build()
    seed_data(args.symlink_model)
    if args.install:
        install(args.stable_sign)
    print("\nDone. Remember to grant permissions to the new bundle "
          "(System Settings → Privacy & Security).")


if __name__ == "__main__":
    main()
