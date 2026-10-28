"""Class exercise 1: predict Wolverine's rating for Titanic with user-based KNN.

Group members:
- Stefan Rybinski
- Tim Stainer
- Rohin Azizian

The dataset already has 25% missing ratings, so there is no train/test split:
we train on every known rating and predict the missing Wolverine/Titanic one.
This reproduces the classroom Excel file (examples/DatasetFinal.xlsx):
Pearson correlation between users, the 6 most similar users who rated Titanic,
and a prediction based on mean-centred ratings.
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

    # First column = item IDs
    item_ids = csv_file.iloc[:, 0]

    # Remaining column headers = user IDs
    ratings_matrix = csv_file.iloc[:, 1:]

    # Add item IDs to the ratings matrix
    df = ratings_matrix.copy()
    df.insert(0, "itemID", item_ids)

    # Convert matrix format to long format
    df = df.melt(id_vars="itemID", var_name="userID", value_name="rating")

    # Remove missing ratings (the 25% empty cells)
    df = df.dropna(subset=["rating"])

    # Convert ratings to numeric. keep IDs as strings
    df["rating"] = pd.to_numeric(df["rating"])

    # Ensure the column order required by Surprise
    return df[["userID", "itemID", "rating"]]


# ---------------------------------------------------------
# Load ratings and train on all of them (no train/test split)
# ---------------------------------------------------------

df = load_csv()

print("Available ratings:", len(df))

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df, reader)

trainset = data.build_full_trainset()

# ---------------------------------------------------------
# User-based KNN with 6 neighbours, as in the Excel file
# ---------------------------------------------------------

# Pearson correlation is the same as Excel's CORREL over co-rated movies.
# KNNWithMeans predicts: user mean + weighted average of the neighbours'
# (rating - their mean), weighted by the correlation.
sim_options = {"name": "pearson", "user_based": True}

k = 6

algo = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)

algo.fit(trainset)

# ---------------------------------------------------------
# Predict Wolverine's rating for Titanic
# ---------------------------------------------------------

target_user = "Wolverine"
target_item = "Titanic"

prediction = algo.predict(target_user, target_item)

print(f"\nPredicted rating of {target_user} for {target_item}: {prediction.est:.4f}")
print("Neighbours used:", prediction.details["actual_k"])

# ---------------------------------------------------------
# Show the 6 neighbours, like the "Wolverine - 25%" Excel sheet
# ---------------------------------------------------------

user_inner = trainset.to_inner_uid(target_user)
item_inner = trainset.to_inner_iid(target_item)

# Users who rated Titanic, as (inner user id, rating) pairs
raters = trainset.ir[item_inner]

# Sort them by correlation with Wolverine, highest first, and keep the top k
neighbours = sorted(
    raters, key=lambda pair: algo.sim[user_inner, pair[0]], reverse=True
)[:k]

print(
    f"\n{'Neighbour':<20}{'Correlation':>12}"
    f"{'Rating':>8}{'Average':>9}{'Normalized':>12}"
)

for other_inner, rating in neighbours:
    name = trainset.to_raw_uid(other_inner)
    correlation = algo.sim[user_inner, other_inner]
    average = algo.means[other_inner]
    print(
        f"{name:<20}{correlation:>12.3f}"
        f"{rating:>8.0f}{average:>9.3f}{rating - average:>12.3f}"
    )

print(f"\n{target_user}'s average rating: {algo.means[user_inner]:.4f}")
