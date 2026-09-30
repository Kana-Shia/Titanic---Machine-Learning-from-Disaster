from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "Survived"
REQUIRED_TRAIN_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}
REQUIRED_TEST_COLUMNS = REQUIRED_TRAIN_COLUMNS - {"Survived"}


class DataValidationError(RuntimeError):
    """Raised when input data does not satisfy required schema."""


def load_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    gender_submission_path = data_dir / "gender_submission.csv"

    for path in (train_path, test_path, gender_submission_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    gender_submission_df = pd.read_csv(gender_submission_path)

    validate_schema(train_df, test_df)
    return train_df, test_df, gender_submission_df


def validate_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    if TARGET_COLUMN not in train_cols:
        raise DataValidationError("train.csv must include Survived column")
    if TARGET_COLUMN in test_cols:
        raise DataValidationError("test.csv must not include Survived column")

    missing_train = REQUIRED_TRAIN_COLUMNS - train_cols
    missing_test = REQUIRED_TEST_COLUMNS - test_cols
    if missing_train:
        raise DataValidationError(f"train.csv missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise DataValidationError(f"test.csv missing required columns: {sorted(missing_test)}")


def summarize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": name,
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "row_count": len(df),
        }
    )


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    title = name.split(",", 1)[-1].split(".", 1)[0].strip()
    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    title = title_mapping.get(title, title)
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    if title not in common_titles:
        title = "Rare"
    return title


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["FamilySize"] = output["SibSp"] + output["Parch"] + 1
    output["IsAlone"] = (output["FamilySize"] == 1).astype(int)
    output["Title"] = output["Name"].apply(extract_title)

    deck = output["Cabin"].fillna("Unknown").astype(str).str[0]
    output["Deck"] = np.where(deck.str.isalpha(), deck, "Unknown")

    fare = output["Fare"].fillna(output["Fare"].median())
    output["FarePerPerson"] = fare / output["FamilySize"].replace(0, 1)

    age_bins = [-1, 12, 19, 59, 200]
    age_labels = ["Child", "Teenager", "Adult", "Senior"]
    age_band = pd.cut(output["Age"], bins=age_bins, labels=age_labels)
    output["AgeBand"] = age_band.astype(object).fillna("Unknown")

    return output


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
    ]
    categorical_features = ["Sex", "Embarked", "Title", "Deck", "AgeBand"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_model_pipelines() -> Dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, str, Pipeline]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results: List[Dict[str, float]] = []
    best_model_name = ""
    best_pipeline: Pipeline | None = None
    best_key = (-np.inf, np.inf)

    for model_name, pipeline in get_model_pipelines().items():
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_elapsed = time.perf_counter() - start

        val_pred = pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=None,
        )

        feature_count = len(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        fold_scores = [round(float(score), 6) for score in cv_scores]

        results.append(
            {
                "model": model_name,
                "validation_accuracy": val_accuracy,
                "cv_mean_accuracy": float(np.mean(cv_scores)),
                "cv_std_accuracy": float(np.std(cv_scores)),
                "cv_fold_accuracies": "|".join(map(str, fold_scores)),
                "training_time": fit_elapsed,
                "feature_count": feature_count,
            }
        )

        selection_key = (float(np.mean(cv_scores)), float(np.std(cv_scores)))
        if selection_key[0] > best_key[0] or (
            np.isclose(selection_key[0], best_key[0]) and selection_key[1] < best_key[1]
        ):
            best_key = selection_key
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was evaluated successfully")

    results_df = pd.DataFrame(results).sort_values(
        ["cv_mean_accuracy", "cv_std_accuracy"], ascending=[False, True]
    )

    return results_df, best_model_name, best_pipeline


def fit_best_model_and_predict(
    best_pipeline: Pipeline, X_full: pd.DataFrame, y_full: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    best_pipeline.fit(X_full, y_full)
    predictions = best_pipeline.predict(X_test)
    return predictions.astype(int)


def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame) -> None:
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == len(test_df)
    assert len(submission) == 418
    assert submission["PassengerId"].is_unique
    assert submission["Survived"].isin([0, 1]).all()
    assert submission["Survived"].notna().all()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "outputs"
    reports_dir = project_root / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, gender_submission_df = load_data(project_root)

    train_summary = summarize_dataframe(train_df, "train")
    test_summary = summarize_dataframe(test_df, "test")
    gender_summary = summarize_dataframe(gender_submission_df, "gender_submission")
    data_summary = pd.concat([train_summary, test_summary, gender_summary], ignore_index=True)
    data_summary.to_csv(reports_dir / "data_summary.csv", index=False)

    engineered_train = add_features(train_df)
    engineered_test = add_features(test_df)

    X = engineered_train.drop(columns=[TARGET_COLUMN, "PassengerId", "Name", "Ticket", "Cabin"])
    y = engineered_train[TARGET_COLUMN]
    X_test = engineered_test.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    results_df, best_model_name, best_pipeline = evaluate_models(X, y)
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    predictions = fit_best_model_and_predict(best_pipeline, X, y, X_test)
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )

    validate_submission(submission, test_df)
    submission.to_csv(outputs_dir / "submission.csv", index=False)

    print("Data summary saved to reports/data_summary.csv")
    print("Model comparison saved to reports/model_comparison.csv")
    print(f"Best model: {best_model_name}")
    print("Submission saved to outputs/submission.csv")


if __name__ == "__main__":
    main()
