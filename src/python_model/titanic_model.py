# Titanic survival prediction - Python version
# Plan: load data -> poke around -> clean up a bit -> logistic regression -> check accuracy -> predict test.csv

import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# where the csv files live. inside docker this is /app/data (set in the Dockerfile),
# if you run the script on your own machine it just looks in src/data
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# ---------------------------------------------------------------
print("=== Step 1: loading the data ===")
print(f"Looking for data in: {DATA_DIR}")

if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
    print("Oops, can't find train.csv and/or test.csv.")
    print("Download them from Kaggle and drop them in src/data (the README shows how).")
    sys.exit(1)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"train.csv has {train_df.shape[0]} rows and {train_df.shape[1]} columns")
print(f"test.csv has {test_df.shape[0]} rows and {test_df.shape[1]} columns")
print("\nFirst few rows of train.csv:")
print(train_df.head())

# ---------------------------------------------------------------
print("\n=== Step 2: quick look at the data ===")
missing = train_df.isna().sum()
print("Columns with missing values in train.csv:")
print(missing[missing > 0])

print(f"\nOverall survival rate: {train_df['Survived'].mean():.1%}")
print("\nSurvival rate by sex:")
print(train_df.groupby("Sex")["Survived"].mean().round(3))
print("\nSurvival rate by passenger class:")
print(train_df.groupby("Pclass")["Survived"].mean().round(3))
print("\n-> Women and first class folks did way better. Makes sense, 'women and children first'.")

# ---------------------------------------------------------------
print("\n=== Step 3: splitting train.csv ===")
# test.csv from Kaggle has NO Survived column, so I can't check accuracy on it.
# So I hold out 20% of train.csv and use that as my "real" test accuracy.
train_raw, holdout_raw = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["Survived"]
)
print(f"Training part: {len(train_raw)} rows, hold-out part: {len(holdout_raw)} rows")

# ---------------------------------------------------------------
print("\n=== Step 4: cleaning up + making new features ===")
# fill missing stuff using ONLY the training part (so the hold-out stays "unseen")
age_fill = train_raw["Age"].median()
fare_fill = train_raw["Fare"].median()
embarked_fill = train_raw["Embarked"].mode()[0]
print(f"Missing Age -> filling with median age: {age_fill}")
print(f"Missing Fare -> filling with median fare: {fare_fill:.2f}")
print(f"Missing Embarked -> filling with most common port: {embarked_fill}")

FEATURE_COLS = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "IsChild"]


def clean_up(df):
    df = df.copy()
    df["Age"] = df["Age"].fillna(age_fill)
    df["Fare"] = df["Fare"].fillna(fare_fill)
    df["Embarked"] = df["Embarked"].fillna(embarked_fill)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1        # me + siblings/spouse + parents/kids
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)     # traveling solo?
    df["IsChild"] = (df["Age"] < 12).astype(int)            # kids under 12
    return df


def to_features(df):
    feats = df[FEATURE_COLS].copy()
    feats["Pclass"] = feats["Pclass"].astype(str)           # class is a category, not a real number
    return pd.get_dummies(feats, columns=["Pclass", "Sex", "Embarked"], drop_first=True, dtype=int)


train_clean = clean_up(train_raw)
X_train = to_features(train_clean)
y_train = train_clean["Survived"]
print("New features added: FamilySize, IsAlone, IsChild")
print(f"Missing values left in the features: {int(train_clean[FEATURE_COLS].isna().sum().sum())}")
print(f"Final feature columns: {list(X_train.columns)}")

# ---------------------------------------------------------------
print("\n=== Step 5: training the logistic regression ===")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Done training. Coefficients (positive = more likely to survive):")
coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
print(coefs.round(3))

# ---------------------------------------------------------------
print("\n=== Step 6: accuracy on the training set ===")
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Training accuracy: {train_acc:.1%}")

# ---------------------------------------------------------------
print("\n=== Step 7: predicting on test.csv ===")
test_clean = clean_up(test_df)
X_test = to_features(test_clean).reindex(columns=X_train.columns, fill_value=0)
test_pred = model.predict(X_test)
print(f"Predicted {len(test_pred)} passengers: {int(test_pred.sum())} survive, {int((test_pred == 0).sum())} don't")
print(f"Predicted survival rate on test.csv: {test_pred.mean():.1%}")

out_path = os.path.join(DATA_DIR, "predictions_python.csv")
pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_pred}).to_csv(out_path, index=False)
print(f"Saved predictions to {out_path} (Kaggle submission format)")

# ---------------------------------------------------------------
print("\n=== Step 8: accuracy on the test side ===")
holdout_clean = clean_up(holdout_raw)
X_holdout = to_features(holdout_clean).reindex(columns=X_train.columns, fill_value=0)
y_holdout = holdout_clean["Survived"]
holdout_acc = accuracy_score(y_holdout, model.predict(X_holdout))
baseline = 1 - y_holdout.mean()
print("(test.csv has no answers, so this uses the 20% hold-out from train.csv)")
print(f"Hold-out accuracy: {holdout_acc:.1%}")
print(f"For comparison, just guessing 'everyone dies' gets: {baseline:.1%}")
print("\nAll done!")
