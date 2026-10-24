
import json
import base64
from pathlib import Path

# Project root (where this script is saved)
ROOT = Path(__file__).resolve().parent

# Read notebooks from notebooks folder
NOTEBOOK_DIR = ROOT / "notebooks"

# Save all graphs outside notebooks folder
OUTPUT_DIR = ROOT / "All_Graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0

for notebook in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
    data = json.loads(notebook.read_text(encoding="utf-8"))
    count = 0

    for cell in data["cells"]:
        for result in cell.get("outputs", []):
            image = result.get("data", {}).get("image/png")

            if image:
                count += 1
                filename = f"{notebook.stem}_{count:03d}.png"

                (OUTPUT_DIR / filename).write_bytes(
                    base64.b64decode("".join(image))
                )

    total += count
    print(f"{notebook.name}: {count} graphs")

print(f"\nTotal saved: {total} graphs")
print(f"Location: {OUTPUT_DIR}")
