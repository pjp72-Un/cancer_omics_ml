"""Ask whether combining omics types beats mRNA alone."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learn_classifier import (
    CV_FOLDS,
    DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    load_data,
    split_data,
)

# plot_style sits at the repository root, outside src.
sys.path.insert(0, str(PROJECT_ROOT))
from plot_style.theme import (  # noqa: E402
    INK,
    MUTED,
    THEME_COLOURS,
    plot_theme,
    save_figure,
)

CHOSEN_C = 0.1
FEATURES_PER_MODEL = 500
OMICS_FILES = {
    "miRNA": "BRCA_miRNA.csv",
    "methylation": "BRCA_Methy.csv",
    "CNV": "BRCA_CNV.csv",
}
COMBINATIONS = {
    "mRNA": ["mRNA"],
    "miRNA": ["miRNA"],
    "methylation": ["methylation"],
    "CNV": ["CNV"],
    "mRNA+miRNA": ["mRNA", "miRNA"],
    "mRNA+methyl": ["mRNA", "methylation"],
    "all four": ["mRNA", "miRNA", "methylation", "CNV"],
}

FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
# A deeper blue than the theme's sky, so dots keep 3:1 contrast.
SINGLE_COLOUR = "#4A7FB8"
COMBINED_COLOUR = THEME_COLOURS["orange"]


def load_all_omics() -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Return one samples-by-features table per omics type, and labels."""
    mrna_features, targets = load_data()
    omics = {"mRNA": mrna_features}
    for omics_name, file_name in OMICS_FILES.items():
        table = pd.read_csv(DATA_DIR / file_name, index_col=0).T
        # The labels follow the mRNA sample order, so every other table
        # must list the same samples in the same order.
        if not table.index.equals(mrna_features.index):
            raise ValueError(f"{omics_name} samples differ from mRNA.")
        omics[omics_name] = table

    # Gene names repeat across types (A1BG is in mRNA, methylation, and
    # CNV), so prefix each column with its type before combining.
    return {
        omics_name: table.add_prefix(f"{omics_name}:")
        for omics_name, table in omics.items()
    }, targets


def describe_omics(omics: dict[str, pd.DataFrame]) -> None:
    """Print feature counts, missing values, and value ranges."""
    print(f"{'type':<12} {'features':>8} {'missing':>8} {'min':>8} {'max':>8}")
    for omics_name, table in omics.items():
        print(
            f"{omics_name:<12} {table.shape[1]:>8} "
            f"{table.isna().sum().sum():>8} "
            f"{table.min().min():>8.2f} {table.max().max():>8.2f}"
        )


def make_pipeline(feature_count: int) -> Pipeline:
    """Return scaler, top-k selection inside each fold, then logistic."""
    # miRNA has only 345 features, so keep all of them when k is larger.
    selected_count = min(FEATURES_PER_MODEL, feature_count)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=selected_count)),
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


def plot_omics_scores(by_fold: pd.DataFrame, path: Path) -> None:
    """Dot plot: mean macro F1 per combination, with its five folds."""
    means = by_fold.mean().sort_values()
    with plot_theme(base_size=10, grid="x", categorical_axis="y"):
        figure, axis = plt.subplots(figsize=(9, 5.2))
        for row, name in enumerate(means.index):
            single = len(COMBINATIONS[name]) == 1
            colour = SINGLE_COLOUR if single else COMBINED_COLOUR
            # Small, faded dots: one per fold.
            axis.scatter(
                by_fold[name], [row] * len(by_fold), s=30, color=colour,
                alpha=0.45, edgecolors="none", zorder=2,
            )  # fmt: skip
            # Large, outlined dot: the mean across folds.
            axis.scatter(
                means[name], row, s=110, color=colour, edgecolors=INK,
                linewidths=0.8, zorder=3,
            )  # fmt: skip
            # Means sit in a column right of the plot, clear of all dots.
            axis.text(
                1.01, row, f"{means[name]:.3f}", va="center", fontsize=9.5,
                color=INK, transform=axis.get_yaxis_transform(),
            )  # fmt: skip
        axis.axvline(
            means["mRNA"], color=MUTED, linestyle="--", linewidth=1, zorder=1
        )
        axis.set_yticks(range(len(means)), means.index)
        axis.set_xlabel("Macro F1, 5-fold cross-validation")
        axis.set_xlim(0.4, 0.9)

        legend_items = [
            Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SINGLE_COLOUR, markeredgecolor=INK,
                   label="One omics type"),
            Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=COMBINED_COLOUR, markeredgecolor=INK,
                   label="Combined types"),
        ]  # fmt: skip
        figure.legend(
            handles=legend_items, loc="upper right", ncols=2,
            bbox_to_anchor=(0.98, 0.9),
        )  # fmt: skip
        figure.suptitle(
            "Combining omics types adds little beyond mRNA alone",
            x=0.02, y=0.97, ha="left", weight="bold", fontsize=13,
        )  # fmt: skip
        figure.subplots_adjust(left=0.15, right=0.92, top=0.8, bottom=0.17)
        save_figure(figure, path)
        plt.close(figure)


def main() -> None:
    """Score each omics type and combination on the same five folds."""
    print("Loading four omics tables; this can take a minute.")
    omics, targets = load_all_omics()
    describe_omics(omics)

    # Split once, then use the same training samples, in the same
    # order, for every combination, so the folds are identical.
    train_mrna, _, train_targets, _ = split_data(omics["mRNA"], targets)
    train_samples = train_mrna.index
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    print("\nMean macro F1 across folds:")
    fold_scores = {}
    for combination_name, omics_names in COMBINATIONS.items():
        # axis=1 joins tables side by side: same rows, more columns.
        combined = pd.concat(
            [omics[name].loc[train_samples] for name in omics_names],
            axis=1,
        )
        scores = cross_val_score(
            make_pipeline(combined.shape[1]),
            combined,
            train_targets,
            cv=folds,
            scoring="f1_macro",
        )
        fold_scores[combination_name] = scores
        print(
            f"  {combination_name:<12} {combined.shape[1]:>6} features  "
            f"{scores.mean():.3f} +/- {scores.std():.3f}"
        )

    by_fold = pd.DataFrame(
        fold_scores, index=pd.RangeIndex(1, CV_FOLDS + 1, name="fold")
    )
    print("\nMacro F1 by fold:")
    print(by_fold.round(3).to_string())

    # Paired: each fold scores every combination on the same samples.
    differences = by_fold.sub(by_fold["mRNA"], axis=0).drop(columns="mRNA")
    print("\nDifference from mRNA alone, per fold:")
    print(differences.round(3).to_string())
    print("\nFolds that beat mRNA alone:")
    print((differences > 0).sum().to_string())
    # Save the raw fold scores, then draw the figure from them.
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(FIGURE_DIR / "multi_omics_folds.csv")
    plot_omics_scores(by_fold, FIGURE_DIR / "multi_omics_scores.png")
    print(f"\nSaved the scores and figure to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
