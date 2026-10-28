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


# No train/test split: the file already has 25% missing ratings,
# so we train on all the ratings we have
df = load_csv()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()

# Pearson over co-rated movies is the same as CORREL in the Excel file.
# KNNWithMeans = user mean + similarity-weighted average of the
# neighbours' (rating - their mean).
k = 6
sim_options = {"name": "pearson", "user_based": True}
algo = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)
algo.fit(trainset)

target_user = "Wolverine"
target_item = "Titanic"

prediction = algo.predict(target_user, target_item)
print("Predicted rating:", round(prediction.est, 4))
print("Neighbours used:", prediction.details["actual_k"])

# Neighbour table, like the "Wolverine - 25%" Excel sheet
user_means = df.groupby("userID")["rating"].mean()
titanic_ratings = df[df["itemID"] == target_item]

# Surprise numbers the users internally; the correlation matrix uses those numbers
wolverine_number = trainset.to_inner_uid(target_user)

rows = []
for _, row in titanic_ratings.iterrows():
    other_user = row["userID"]
    other_number = trainset.to_inner_uid(other_user)

    correlation = algo.sim[wolverine_number, other_number]
    average = user_means[other_user]

    rows.append(
        {
            "Neighbour": other_user,
            "Correlation": correlation,
            "Rating": row["rating"],
            "Average": average,
            "Normalized": row["rating"] - average,
        }
    )

# Keep the 6 users with the highest correlation to Wolverine
neighbours = pd.DataFrame(rows)
neighbours = neighbours.sort_values("Correlation", ascending=False)
neighbours = neighbours.head(k)

print()
print(neighbours.round(3).to_string(index=False))

print()
print("Wolverine's average rating:", round(user_means[target_user], 4))
