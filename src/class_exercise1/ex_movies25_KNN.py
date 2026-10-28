"""Class exercise 1: predict Wolverine's rating for Titanic with user-based KNN (k=6).

Group members:
- Stefan Rybinski
- Tim Stainer
- Rohin Azizian
"""

from pathlib import Path

import pandas as pd
from surprise import Dataset, KNNWithMeans, Reader


def load_csv():
    """Load ratings from a matrix CSV and return a long-format DataFrame."""

    csv_file = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "data.csv",
        delimiter=";",
        dtype=str,
    )

    # First column = item IDs, remaining column headers = user IDs
    df = csv_file.iloc[:, 1:].copy()
    df.insert(0, "itemID", csv_file.iloc[:, 0])

    # Matrix -> long format, without the empty cells
    df = df.melt(id_vars="itemID", var_name="userID", value_name="rating")
    df = df.dropna(subset=["rating"])
    df["rating"] = pd.to_numeric(df["rating"])

    return df[["userID", "itemID", "rating"]]


# No train/test split: the file already has 25% missing ratings, so we train on all of them
df = load_csv()
trainset = Dataset.load_from_df(df, Reader(rating_scale=(1, 5))).build_full_trainset()

# Pearson over co-rated movies is the same as CORREL in the Excel file.
# KNNWithMeans = user mean + similarity-weighted average of the neighbours' (rating - mean).
k = 6
algo = KNNWithMeans(k, sim_options={"name": "pearson", "user_based": True}, verbose=False)
algo.fit(trainset)

target_user = "Wolverine"
target_item = "Titanic"

prediction = algo.predict(target_user, target_item)
print(f"Predicted rating of {target_user} for {target_item}: {prediction.est:.4f}")
print("Neighbours used:", prediction.details["actual_k"])

# Neighbour table, like the "Wolverine - 25%" Excel sheet.
# trainset.ir[item] is a list of (inner user id, rating) for the users who rated that item.
user_inner = trainset.to_inner_uid(target_user)
raters = trainset.ir[trainset.to_inner_iid(target_item)]
neighbours = sorted(raters, key=lambda r: algo.sim[user_inner, r[0]], reverse=True)[:k]

print(f"\n{'Neighbour':<20}{'Correlation':>12}{'Rating':>8}{'Average':>9}{'Normalized':>12}")
for other_inner, rating in neighbours:
    average = algo.means[other_inner]
    print(
        f"{trainset.to_raw_uid(other_inner):<20}{algo.sim[user_inner, other_inner]:>12.3f}"
        f"{rating:>8.0f}{average:>9.3f}{rating - average:>12.3f}"
    )

print(f"\n{target_user}'s average rating: {algo.means[user_inner]:.4f}")
