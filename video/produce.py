import argparse
import json
import math
import posixpath
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def extract_assets(deck_path: Path) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(deck_path), "slides": [], "images": []}
    with zipfile.ZipFile(deck_path) as archive:
        slide_names = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide")
                and name.endswith(".xml")
                and "/_rels/" not in name
            ),
            key=lambda name: int(Path(name).stem.replace("slide", "")),
        )
        for slide_name in slide_names:
            slide_root = ET.fromstring(archive.read(slide_name))
            relationships_name = (
                f"ppt/slides/_rels/{Path(slide_name).name}.rels"
            )
            relationships = {}
            if relationships_name in archive.namelist():
                for relationship in ET.fromstring(archive.read(relationships_name)):
                    relationships[relationship.attrib["Id"]] = posixpath.normpath(
                        posixpath.join("ppt/slides", relationship.attrib["Target"])
                    )
            pictures = []
            for picture in slide_root.findall(".//p:pic", NAMESPACES):
                blip = picture.find(".//a:blip", NAMESPACES)
                if blip is not None:
                    relationship_id = blip.attrib.get(
                        f"{{{NAMESPACES['r']}}}embed"
                    )
                    if relationship_id in relationships:
                        pictures.append(Path(relationships[relationship_id]).name)
            manifest["slides"].append(
                {
                    "slide": Path(slide_name).stem,
                    "text": [
                        node.text
                        for node in slide_root.findall(".//a:t", NAMESPACES)
                        if node.text
                    ],
                    "pictures": pictures,
                }
            )
        for entry in archive.infolist():
            if not entry.filename.startswith("ppt/media/") or entry.is_dir():
                continue
            asset_path = ASSETS / Path(entry.filename).name
            with archive.open(entry) as source, asset_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            try:
                with Image.open(asset_path) as image:
                    dimensions = image.size
                manifest["images"].append(
                    {
                        "file": asset_path.name,
                        "width": dimensions[0],
                        "height": dimensions[1],
                        "slides": [
                            slide["slide"]
                            for slide in manifest["slides"]
                            if asset_path.name in slide["pictures"]
                        ],
                    }
                )
            except OSError:
                continue
    (ASSETS / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    candidates = [
        image
        for image in manifest["images"]
        if image["width"] >= 500 and image["height"] >= 200
    ]
    assert candidates, "No usable screenshots were extracted."
    sheet = Image.new("RGB", (1800, math.ceil(len(candidates) / 3) * 400), "#0A101D")
    drawing = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    for index, item in enumerate(candidates):
        column, row = index % 3, index // 3
        with Image.open(ASSETS / item["file"]) as original:
            thumbnail = original.convert("RGB")
            thumbnail.thumbnail((580, 340), Image.Resampling.LANCZOS)
        left = column * 600 + (600 - thumbnail.width) // 2
        top = row * 400 + 10
        sheet.paste(thumbnail, (left, top))
        label = (
            f"{item['file']} | {item['width']}x{item['height']} | "
            f"{', '.join(item['slides'])}"
        )
        drawing.text((column * 600 + 12, row * 400 + 363), label, font=font, fill="white")
    sheet.save(ASSETS / "contact_sheet.jpg", quality=92)
    print(f"Extracted {len(manifest['slides'])} slides and {len(manifest['images'])} images.")
    for item in manifest["images"]:
        print(f"{item['file']}: {item['width']}x{item['height']} ({', '.join(item['slides'])})")
    print(f"Contact sheet: {ASSETS / 'contact_sheet.jpg'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["extract"])
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    extract_assets(arguments.source)


if __name__ == "__main__":
    main()