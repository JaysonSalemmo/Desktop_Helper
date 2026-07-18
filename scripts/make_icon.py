"""
Generate assets/AppIcon.icns — a rendered 🤖 on a rounded-rect background.

No design tools needed: AppKit draws each size into a PNG, iconutil packs the
.icns. Rerun any time to change the glyph/colors; scripts/build_app.py copies
the result into the bundle.
"""
import subprocess
import tempfile
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSFont,
                    NSGraphicsContext, NSImage, NSMakeRect, NSPNGFileType)
from Foundation import NSString

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "assets" / "AppIcon.icns"
GLYPH = "🤖"


def _render(size: int) -> bytes:
    image = NSImage.alloc().initWithSize_((size, size))
    image.lockFocus()

    # rounded-rect background, macOS-icon style (~80% canvas, dark slate)
    inset = size * 0.1
    radius = size * 0.18
    rect = NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.16, 0.17, 0.21, 1.0).set()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius).fill()

    glyph = NSString.stringWithString_(GLYPH)
    font = NSFont.systemFontOfSize_(size * 0.52)
    attrs = {"NSFont": font}
    bounds = glyph.sizeWithAttributes_(attrs)
    glyph.drawAtPoint_withAttributes_(
        ((size - bounds.width) / 2, (size - bounds.height) / 2), attrs)

    image.unlockFocus()
    tiff = image.TIFFRepresentation()
    rep = NSBitmapImageRep.imageRepWithData_(tiff)
    return bytes(rep.representationUsingType_properties_(NSPNGFileType, None))


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            (iconset / f"icon_{size}x{size}.png").write_bytes(_render(size))
            (iconset / f"icon_{size}x{size}@2x.png").write_bytes(_render(size * 2))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUTPUT)],
                       check=True)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
