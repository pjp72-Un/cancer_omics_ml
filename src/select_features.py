"""Choose genes inside cross-validation, and see what leakage does."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learn_classifier import (
    CV_FOLDS,
    RANDOM_STATE,
    load_data,
    make_logistic_pipeline,
    split_data,
)

CHOSEN_C = 0.1
GENE_COUNTS = [20, 100, 500, 2000]


def make_selected_pipeline(gene_count: int) -> Pipeline:
    """Return scaler, top-k gene selector, then logistic regression."""
    # SelectKBest ranks genes by an ANOVA F-test: how strongly each
    # gene's mean differs between classes. It looks at the labels, so
    # it must learn only from the training rows of each fold.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=gene_count)),
            (
                "classifier",
                LogisticRegression(
                    C=CHOSEN_C,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def honest_scores(
    features: pd.DataFrame,
    targets: pd.Series,
    gene_count: int,
    folds: StratifiedKFold,
) -> pd.Series:
    """Select genes inside every fold, as part of the pipeline."""
    scores = cross_val_score(
        make_selected_pipeline(gene_count),
        features,
        targets,
        cv=folds,
        scoring="f1_macro",
    )
    return pd.Series(scores)


def leaky_scores(
    features: pd.DataFrame,
    targets: pd.Series,
    gene_count: int,
    folds: StratifiedKFold,
) -> pd.Series:
    """The wrong way: select genes once using every row, then score."""
    # The selector sees the labels of rows that later act as
    # validation folds. This is the mistake to recognise and avoid.
    selector = SelectKBest(f_classif, k=gene_count).fit(features, targets)
    selected_features = features.loc[:, selector.get_support()]
    scores = cross_val_score(
        make_logistic_pipeline(inverse_penalty=CHOSEN_C),
        selected_features,
        targets,
        cv=folds,
        scoring="f1_macro",
    )
    return pd.Series(scores)


def compare_methods(
    features: pd.DataFrame,
    targets: pd.Series,
    folds: StratifiedKFold,
) -> pd.DataFrame:
    """Return honest and leaky macro F1 for every gene count."""
    rows = []
    for gene_count in GENE_COUNTS:
        honest = honest_scores(features, targets, gene_count, folds)
        leaky = leaky_scores(features, targets, gene_count, folds)
        rows.append(
            {
                "genes": gene_count,
                "honest_f1": honest.mean(),
                "honest_std": honest.std(),
                "leaky_f1": leaky.mean(),
                "inflation": leaky.mean() - honest.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Compare honest and leaky selection on real and shuffled labels."""
    features, targets = load_data()
    train_features, _, train_targets, _ = split_data(features, targets)
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    print("Real labels (all genes, step 6, scored 0.793):")
    real_table = compare_methods(train_features, train_targets, folds)
    print(real_table.round(3).to_string(index=False))

    # Shuffling breaks every link between genes and labels while keeping
    # the class counts. An honest method should now score near chance.
    shuffled_targets = pd.Series(
        train_targets.sample(frac=1, random_state=RANDOM_STATE).to_numpy(),
        index=train_targets.index,
    )
    print("\nShuffled labels (no real signal exists):")
    shuffled_table = compare_methods(train_features, shuffled_targets, folds)
    print(shuffled_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
