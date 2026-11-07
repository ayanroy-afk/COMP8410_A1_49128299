import json
import gzip
import random
import csv
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("Models/Pair_04/datasets/ABO/listings/metadata")
OUTPUT = Path("Models/Pair_04/datasets/products.csv")

N = 30
MAX_PER_CATEGORY = 2
SEED = 42

random.seed(SEED)


def get_text(data):
    if not data:
        return ""

    return " ".join(
        x["value"] for x in data
        if isinstance(x, dict)
        and isinstance(x.get("value"), str)
        and x.get("language_tag", "en_US").startswith("en")
    ).strip()


def get_category(data):
    if not data:
        return "UNKNOWN"
    return str(data[0].get("value", "UNKNOWN"))


# Read products from all metadata files
products_by_category = defaultdict(list)
category_counts = defaultdict(int)
seen_ids = set()

files = sorted(DATA_DIR.glob("listings_*.json.gz"))

for file in files:
    print(f"Reading {file.name}")

    with gzip.open(file, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            item_id = item.get("item_id", "")
            name = get_text(item.get("item_name", []))
            description = get_text(item.get("bullet_point", []))
            category = get_category(item.get("product_type", []))

            if not item_id or not name or not description:
                continue

            if item_id in seen_ids or category == "UNKNOWN":
                continue

            seen_ids.add(item_id)

            product = {
                "item_id": item_id,
                "product_name": name,
                "description": description,
                "product_type": category,
                "source_file": file.name
            }

            # Keep a random sample of up to 2 products per category
            group = products_by_category[category]

            if len(group) < MAX_PER_CATEGORY:
                group.append(product)
            else:
                # Reservoir sampling within each category
                count = category_counts[category]
                index = random.randrange(count)

                if index < MAX_PER_CATEGORY:
                    group[index] = product

            category_counts[category] += 1


# Randomly select products across categories
categories = list(products_by_category.keys())
random.shuffle(categories)

selected = []

for category in categories:
    group = products_by_category[category]
    random.shuffle(group)

    for product in group:
        if len(selected) < N:
            selected.append(product)


if len(selected) < N:
    raise ValueError(f"Only found {len(selected)} suitable products.")


# Shuffle final dataset
random.shuffle(selected)

# Save CSV
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "item_id",
            "product_name",
            "description",
            "product_type",
            "source_file"
        ]
    )
    writer.writeheader()
    writer.writerows(selected)


print(f"\nSaved {len(selected)} products to {OUTPUT}")
print(f"Categories represented: {len(set(p['product_type'] for p in selected))}")