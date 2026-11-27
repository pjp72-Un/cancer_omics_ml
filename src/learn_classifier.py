"""Learn a breast-cancer subtype classifier with GS-BRCA data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "data/Cancer-Multi-Omics-Benchmark/datasets"
    / "Classification_datasets/GS-BRCA/Original"
)

RANDOM_STATE = 6
TEST_FRACTION = 0.2
CV_FOLDS = 5
CANDIDATE_C = [0.001, 0.01, 0.1, 1.0, 10.0]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return samples-by-genes features and one integer label per sample."""
    # Gene IDs are in the first column, so index_col=0 keeps them out of
    # the measurements.
    expression = pd.read_csv(DATA_DIR / "BRCA_mRNA.csv", index_col=0)
    labels = pd.read_csv(DATA_DIR / "BRCA_label_num.csv")

    # Models expect one row per sample.
    features = expression.T
    targets = labels["Label"]

    if len(features) != len(targets):
        raise ValueError("The sample and label counts do not match.")

    # Labels carry no sample IDs, so they rely on file order. Give them
    # the feature index so that rows stay paired when we select subsets.
    targets.index = features.index
    return features, targets


def inspect_data(features: pd.DataFrame, targets: pd.Series) -> None:
    """Print shapes, class balance, value range, and scaling checks."""
    print(f"Features (samples, genes): {features.shape}")
    print(f"Targets: {targets.shape}, dtype {targets.dtype}")
    print(f"Missing measurements: {features.isna().sum().sum()}")
    print(f"Missing targets: {targets.isna().sum()}")

    counts = targets.value_counts().sort_index()
    fractions = targets.value_counts(normalize=True).sort_index()
    balance = pd.DataFrame({"count": counts, "fraction": fractions})
    print("\nClass balance:")
    print(balance.round(3).to_string())
    print(
        f"\nValue range: {features.min().min():.3f} "
        f"to {features.max().max():.3f}"
    )
    # A TCGA patient ID is the first 12 characters of the barcode. The
    # last two digits give the sample type; 01 means primary tumour.
    patient_ids = features.index.str[:12]
    print(
        f"Unique patients: {patient_ids.nunique()} "
        f"of {len(patient_ids)} samples"
    )
    print(f"Repeated patients: {patient_ids.duplicated().sum()}")
    sample_types = features.index.str[-2:].value_counts()
    print(f"Sample types: {sample_types.to_dict()}")

    # If each gene was z-scored upstream, every gene mean is about 0 and
    # every gene standard deviation is about 1.
    gene_means = features.mean()
    gene_stds = features.std()
    print("\nPer-gene mean, summarised over all genes:")
    print(gene_means.describe().round(3).to_string())
    print("\nPer-gene std, summarised over all genes:")
    print(gene_stds.describe().round(3).to_string())

    print(f"\nFirst sample IDs: {list(features.index[:5])}")


def split_data(
    features: pd.DataFrame,
    targets: pd.Series,
    holdout_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split samples into stratified training and test sets."""
    # stratify=targets keeps each class's fraction about the same in
    # both sets. random_state makes the shuffle repeatable.
    train_features, test_features, train_targets, test_targets = (
        train_test_split(
            features,
            targets,
            test_size=holdout_fraction,
            stratify=targets,
            random_state=RANDOM_STATE,
        )
    )
    return train_features, test_features, train_targets, test_targets


def check_split(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
) -> None:
    """Confirm the split is disjoint, paired, and class-balanced."""
    shared = train_features.index.intersection(test_features.index)
    if len(shared) > 0:
        raise ValueError(f"{len(shared)} samples are in both sets.")
    if not train_features.index.equals(train_targets.index):
        raise ValueError("Training features and labels are not paired.")
    if not test_features.index.equals(test_targets.index):
        raise ValueError("Test features and labels are not paired.")

    print(f"\nTraining samples: {len(train_targets)}")
    print(f"Test samples: {len(test_targets)}")
    balance = pd.DataFrame(
        {
            "train_count": train_targets.value_counts(),
            "test_count": test_targets.value_counts(),
            "train_fraction": train_targets.value_counts(normalize=True),
            "test_fraction": test_targets.value_counts(normalize=True),
        }
    ).sort_index()
    print("\nClass balance by split:")
    print(balance.round(3).to_string())


def score_predictions(
    true_targets: pd.Series, predicted_targets: np.ndarray
) -> dict[str, float]:
    """Return accuracy, balanced accuracy, and macro F1."""
    # Accuracy: fraction of samples predicted correctly.
    # Balanced accuracy: the mean of each class's recall, so every class
    # counts equally however many samples it has.
    # Macro F1: the mean of each class's F1 score, also equally weighted.
    # zero_division=0 scores a never-predicted class as 0, not a warning.
    return {
        "accuracy": float(accuracy_score(true_targets, predicted_targets)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_targets, predicted_targets)
        ),
        "macro_f1": float(
            f1_score(
                true_targets,
                predicted_targets,
                average="macro",
                zero_division=0,
            )
        ),
    }


def print_scores(model_name: str, scores: dict[str, float]) -> None:
    """Print one model's scores on a single labelled block."""
    print(f"\n{model_name}:")
    for metric_name, value in scores.items():
        print(f"  {metric_name:<18} {value:.3f}")


def make_logistic_pipeline(inverse_penalty: float = 1.0) -> Pipeline:
    """Return a scaler followed by multinomial L2 logistic regression."""
    # Pipeline.fit fits the scaler on the rows it is given, transforms
    # them, then fits the classifier. Pipeline.predict reuses the
    # fitted scaler, so scaling never learns from evaluation rows.
    # C is the inverse of penalty strength: smaller C, simpler model.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=inverse_penalty,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_logistic(
    train_features: pd.DataFrame, train_targets: pd.Series
) -> GridSearchCV:
    """Choose C by stratified cross-validated macro F1, then refit."""
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    # "classifier__C" means: the C argument of the pipeline step named
    # "classifier". Each fold refits the scaler on its own rows.
    search = GridSearchCV(
        make_logistic_pipeline(),
        param_grid={"classifier__C": CANDIDATE_C},
        scoring="f1_macro",
        cv=folds,
        return_train_score=True,
    )
    search.fit(train_features, train_targets)
    return search


def print_search_results(search: GridSearchCV) -> None:
    """Print mean fitting and validation scores for each candidate C."""
    # scikit-learn calls the held-out fold "test"; here it is our
    # validation fold, not the final test set.
    results = pd.DataFrame(search.cv_results_)
    table = results[
        [
            "param_classifier__C",
            "mean_train_score",
            "mean_test_score",
            "std_test_score",
        ]
    ].rename(
        columns={
            "param_classifier__C": "C",
            "mean_train_score": "fit_f1",
            "mean_test_score": "validation_f1",
            "std_test_score": "validation_f1_std",
        }
    )
    print("\nCross-validated macro F1 by C:")
    print(table.round(3).to_string(index=False))
    print(f"Chosen C: {search.best_params_['classifier__C']}")


def main() -> None:
    """Load the GS-BRCA mRNA data, inspect it, and split it."""
    features, targets = load_data()
    inspect_data(features, targets)
    train_features, test_features, train_targets, test_targets = split_data(
        features, targets
    )
    check_split(train_features, test_features, train_targets, test_targets)
    # The majority model learns one thing from the training labels: the
    # most common class. It ignores the features entirely.
    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(train_features, train_targets)
    majority_predictions = majority_model.predict(test_features)
    majority_scores = score_predictions(test_targets, majority_predictions)
    print_scores("Majority class", majority_scores)
    # Every choice is made inside the training set.
    search = tune_logistic(train_features, train_targets)
    print_search_results(search)

    # Final evaluation: the test set is used once, after C is chosen.
    test_predictions = search.predict(test_features)
    test_scores = score_predictions(test_targets, test_predictions)
    print_scores("Logistic regression, test", test_scores)
    print("\nPer-class results on the test set:")
    print(
        classification_report(
            test_targets, test_predictions, digits=3, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
