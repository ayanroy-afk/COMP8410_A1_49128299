
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import random
import math

# Set paths and random seed
base = Path(__file__).resolve().parent / "datasets"
source = base / "ABO/images/small"
images_dir = base / "images"
masks_dir = base / "masks"

images_dir.mkdir(exist_ok=True)
masks_dir.mkdir(exist_ok=True)

random.seed(42)

# Select 100 random images
files = [p for p in source.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
selected = random.sample(files, 100)

rows = []
sizes = {"small": 0.10, "medium": 0.25, "large": 0.40}

for i, file in enumerate(selected, 1):
    # Resize image to 512x512
    image = Image.open(file).convert("RGB").resize((512, 512))
    name = f"img_{i:04d}"
    image.save(images_dir / f"{name}.png")

    for mask_type, coverage in sizes.items():
        # Create a random rectangular mask
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)

        side = round(math.sqrt(coverage) * 512)
        x = random.randint(0, 512 - side)
        y = random.randint(0, 512 - side)

        draw.rectangle((x, y, x + side - 1, y + side - 1), fill=255)

        mask_name = f"{name}_{mask_type}.png"
        mask.save(masks_dir / mask_name)

        rows.append({
            "image_id": name,
            "source": str(file.relative_to(source)),
            "image_path": f"images/{name}.png",
            "mask_path": f"masks/{mask_name}",
            "mask_type": mask_type,
            "mask_coverage": coverage
        })

# Save test cases for both models
pd.DataFrame(rows).to_csv(base / "test_cases.csv", index=False)

print(f"Saved {len(selected)} images and {len(rows)} masks")
