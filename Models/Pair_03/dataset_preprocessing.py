#Prepare the Amazon Musical Instruments interactions for Pair 3.


import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent / "datasets"
SOURCE = BASE / "Amazon_Reviews_2023" / "Musical_Instruments"
OUTPUT = BASE / "processed"
SEED = 42
N_USERS = 10_000
MIN_RATING = 4


def load_split(name):
    file = SOURCE / f"Musical_Instruments.{name}.csv.gz"
    df = pd.read_csv(file, usecols=["user_id", "parent_asin", "rating"])
    df = df.dropna(subset=["user_id", "parent_asin", "rating"])
    df = df[df["rating"] >= MIN_RATING]
    return df.drop_duplicates(subset=["user_id", "parent_asin"]).copy()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    # Select users using positive training interactions only.
    available_users = train["user_id"].drop_duplicates()
    if len(available_users) < N_USERS:
        raise ValueError(f"Only {len(available_users)} eligible users; need {N_USERS}.")
    chosen_users = set(available_users.sample(n=N_USERS, random_state=SEED))

    train = train[train["user_id"].isin(chosen_users)].copy()
    valid = valid[valid["user_id"].isin(chosen_users)].copy()
    test = test[test["user_id"].isin(chosen_users)].copy()

    # Create one shared user/item mapping from the training graph only.
    user_ids = sorted(train["user_id"].unique())
    item_ids = sorted(train["parent_asin"].unique())
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}

    pd.DataFrame({"user_id": user_ids, "user_idx": range(len(user_ids))}).to_csv(
        OUTPUT / "user_mapping.csv", index=False
    )
    pd.DataFrame({"parent_asin": item_ids, "item_idx": range(len(item_ids))}).to_csv(
        OUTPUT / "item_mapping.csv", index=False
    )

    stats = {
        "seed": SEED,
        "sampled_users": N_USERS,
        "minimum_positive_rating": MIN_RATING,
        "num_users": len(user_ids),
        "num_items": len(item_ids),
    }

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        total = len(data)
        # Exclude held-out products that the model cannot represent.
        data = data[
            data["user_id"].isin(user_map) & data["parent_asin"].isin(item_map)
        ].copy()
        data["user_idx"] = data["user_id"].map(user_map)
        data["item_idx"] = data["parent_asin"].map(item_map)
        data = data[["user_idx", "item_idx"]].sort_values(
            ["user_idx", "item_idx"]
        ).reset_index(drop=True)
        data.to_csv(OUTPUT / f"{name}.csv", index=False)

        stats[name] = {
            "positive_interactions_before_item_filter": total,
            "saved_interactions": len(data),
            "excluded_unseen_items": total - len(data),
            "item_coverage": round(len(data) / total, 4) if total else None,
            "users_with_saved_interactions": int(data["user_idx"].nunique()),
        }
        print(f"{name.upper()}: {len(data):,} interactions ({total - len(data):,} excluded)")

    with open(OUTPUT / "dataset_stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(f"Users: {len(user_ids):,} | Items: {len(item_ids):,}")
    print(f"Saved files to: {OUTPUT}")


if __name__ == "__main__":
    main()
