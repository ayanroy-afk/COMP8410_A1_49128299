from pathlib import Path
import pandas as pd

# Set paths
base = Path(__file__).resolve().parent / "datasets"
examples_file = next(base.rglob("shopping_queries_dataset_examples.parquet"))
products_file = next(base.rglob("shopping_queries_dataset_products.parquet"))

# Load and merge the datasets
examples = pd.read_parquet(examples_file)
products = pd.read_parquet(products_file)

df = examples.merge(products, on=["product_id", "product_locale"], how="left")

# Use the English test data
df = df[(df["product_locale"] == "us") & (df["split"] == "test")].copy()

# Clean product information
text_cols = ["query", "product_title", "product_description"]
df[text_cols] = df[text_cols].fillna("").astype(str)

df = df[df["query"].str.strip().ne("") & df["product_title"].str.strip().ne("")]
df = df.drop_duplicates(subset=["query_id", "product_id"])

# Classification: 100 examples from each ESCI class
classification = df[df["large_version"] == 1].copy()
classification = pd.concat([
    group.sample(n=min(100, len(group)), random_state=42)
    for _, group in classification.groupby("esci_label")
]).sample(frac=1, random_state=42)

classification = classification[
    ["example_id", "query_id", "query", "product_id",
     "product_title", "product_description", "esci_label"]
]
classification.to_csv(base / "classification_cases.csv", index=False)

# Ranking: 50 queries with 10 candidate products each
ranking = df[df["small_version"] == 1].copy()
ranking = ranking.groupby("query_id").filter(lambda x: len(x) >= 10)

# Select queries containing both relevant and irrelevant products
ranking = ranking.groupby("query_id").filter(
    lambda x: ("E" in x["esci_label"].values)
    and (x["esci_label"] != "E").any()
)

query_ids = ranking["query_id"].drop_duplicates().sample(n=50, random_state=42)
ranking = ranking[ranking["query_id"].isin(query_ids)]

# Keep 10 products per query, including at least one exact match
ranking = pd.concat([
    pd.concat([
        group[group["esci_label"] == "E"].sample(n=1, random_state=42),
        group[group["esci_label"] != "E"].sample(
            n=9, random_state=42
        )
    ]).sample(frac=1, random_state=42)
    for _, group in ranking.groupby("query_id")
    if (group["esci_label"] != "E").sum() >= 9
])

ranking = ranking[
    ["query_id", "query", "product_id", "product_title",
     "product_description", "esci_label"]
]
ranking.to_csv(base / "ranking_cases.csv", index=False)

print("Classification examples:", len(classification))
print("Ranking queries:", ranking["query_id"].nunique())
print("Ranking examples:", len(ranking))
