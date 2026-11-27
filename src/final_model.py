"""Nested cross-validation for an honest estimate, then one test score."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)

from inspect_model import FIGURE_DIR, SUBTYPE_NAMES
from learn_classifier import (
    CV_FOLDS,
    PROJECT_ROOT,
    RANDOM_STATE,
    load_data,
    score_predictions,
    split_data,
)
from select_features import make_selected_pipeline

# plot_style sits at the repository root, outside src.
sys.path.insert(0, str(PROJECT_ROOT))
from plot_style.theme import (  # noqa: E402
    GRID,
    INK,
    THEME_COLOURS,
    plot_theme,
    save_figure,
)

BASELINE_NAME = "Majority baseline"
TUNED_NAME = "Tuned logistic"
INNER_FOLDS = 3
# Every combination of these is tried inside each outer training fold.
PARAM_GRID = {
    "selector__k": [100, 500, 2000, "all"],
    "classifier__C": [0.01, 0.1, 1.0],
}
# zero_division=0 scores a never-predicted class as 0 without warnings,
# which matters for the majority baseline.
SCORING = {
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}
FOLD_BLUE = "#4A7FB8"
TEST_ORANGE = THEME_COLOURS["orange"]


def make_search() -> GridSearchCV:
    """Return a grid search over gene count and C, scored by macro F1."""
    inner_folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    # The starting k and C are placeholders; the grid replaces them.
    # n_jobs=-1 runs the candidate fits on all CPU cores.
    return GridSearchCV(
        make_selected_pipeline(500),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=inner_folds,
        n_jobs=-1,
    )


def nested_scores(features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
    """Score the baseline and the whole tuning procedure on outer folds."""
    outer_folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    models = {
        BASELINE_NAME: DummyClassifier(strategy="most_frequent"),
        TUNED_NAME: make_search(),
    }
    rows = []
    for model_name, model in models.items():
        # Each outer fold refits the model, including its whole grid
        # search, on the other four folds, then scores the held-out one.
        result = cross_validate(
            model,
            features,
            targets,
            cv=outer_folds,
            scoring=SCORING,
            return_estimator=True,
        )
        for fold_index, estimator in enumerate(result["estimator"]):
            row = {"model": model_name, "fold": fold_index + 1}
            for metric_name in SCORING:
                row[metric_name] = result[f"test_{metric_name}"][fold_index]
            if isinstance(estimator, GridSearchCV):
                row.update(estimator.best_params_)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_scores_panel(
    axis: plt.Axes, nested: pd.DataFrame, test_f1: float
) -> None:
    """Fold scores, their mean, and the single test score per model."""
    model_names = [BASELINE_NAME, TUNED_NAME]
    for row, model_name in enumerate(model_names):
        folds = nested.loc[nested["model"] == model_name, "macro_f1"]
        axis.scatter(
            folds, [row] * len(folds), s=34, color=FOLD_BLUE, alpha=0.45,
            edgecolors="none", zorder=2,
        )  # fmt: skip
        axis.scatter(
            folds.mean(), row, s=120, color=FOLD_BLUE, edgecolors=INK,
            linewidths=0.8, zorder=3,
        )  # fmt: skip
        axis.text(
            1.02, row, f"{folds.mean():.3f}", va="center", fontsize=10,
            color=INK, transform=axis.get_yaxis_transform(),
        )  # fmt: skip
    # The final model was scored on the test set exactly once.
    axis.scatter(
        test_f1, 1, s=130, marker="D", color=TEST_ORANGE, edgecolors=INK,
        linewidths=0.8, zorder=4,
    )  # fmt: skip

    axis.set_yticks([0, 1], model_names)
    axis.set_ylim(-0.5, 1.5)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Macro F1")
    axis.set_title("Nested 5-fold cross-validation", loc="left")
    legend_items = [
        Line2D([], [], marker="o", linestyle="", markersize=6,
               markerfacecolor=FOLD_BLUE, markeredgecolor="none",
               alpha=0.45, label="Outer fold"),
        Line2D([], [], marker="o", linestyle="", markersize=10,
               markerfacecolor=FOLD_BLUE, markeredgecolor=INK,
               label="Mean of folds"),
        Line2D([], [], marker="D", linestyle="", markersize=9,
               markerfacecolor=TEST_ORANGE, markeredgecolor=INK,
               label="Test set, final model"),
    ]  # fmt: skip
    axis.legend(handles=legend_items, loc="lower left", fontsize=9)


def plot_confusion_panel(axis: plt.Axes, counts: np.ndarray) -> None:
    """Test-set confusion matrix: counts, shaded by row percentage."""
    names = list(SUBTYPE_NAMES.values())
    # Each row is one true subtype, so row percentages are recalls.
    row_fractions = counts / counts.sum(axis=1, keepdims=True)
    colour_map = LinearSegmentedColormap.from_list(
        "paper_to_blue", ["#F4F2EC", FOLD_BLUE, "#1F4466"]
    )
    axis.imshow(row_fractions, cmap=colour_map, vmin=0, vmax=1)
    for true_index, predicted_index in np.ndindex(counts.shape):
        fraction = row_fractions[true_index, predicted_index]
        axis.text(
            predicted_index, true_index,
            f"{counts[true_index, predicted_index]}\n{fraction:.0%}",
            ha="center", va="center", fontsize=8.5,
            color="white" if fraction > 0.55 else INK,
        )  # fmt: skip
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted subtype")
    axis.set_ylabel("True subtype")
    axis.set_title("Test set (135 samples)", loc="left")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_final(
    nested: pd.DataFrame,
    test_f1: float,
    counts: np.ndarray,
    path: Path,
) -> None:
    """Two panels: nested scores beside the test confusion matrix."""
    tuned_mean = nested.loc[nested["model"] == TUNED_NAME, "macro_f1"].mean()
    with plot_theme(base_size=10, grid="x", categorical_axis="both"):
        figure, (score_axis, matrix_axis) = plt.subplots(
            1, 2, figsize=(12, 5.6), width_ratios=[1.2, 1]
        )
        plot_scores_panel(score_axis, nested, test_f1)
        plot_confusion_panel(matrix_axis, counts)
        figure.suptitle(
            f"Final model: macro F1 {tuned_mean:.2f} in nested "
            f"cross-validation and {test_f1:.2f} on the held-out test set",
            x=0.02, y=0.97, ha="left", weight="bold", fontsize=13,
        )  # fmt: skip
        figure.subplots_adjust(
            left=0.13, right=0.97, top=0.78, bottom=0.17, wspace=0.45
        )
        matrix_axis.spines["left"].set_color(GRID)
        save_figure(figure, path)
        plt.close(figure)


def main() -> None:
    """Estimate with nested CV, then fit once and score the test set."""
    features, targets = load_data()
    train_features, test_features, train_targets, test_targets = split_data(
        features, targets
    )

    started = time.perf_counter()
    nested = nested_scores(train_features, train_targets)
    print(f"Nested cross-validation took {time.perf_counter() - started:.0f}s")
    print("\nOuter-fold scores and the settings each fold chose:")
    print(nested.round(3).to_string(index=False))
    print("\nMean and standard deviation by model:")
    summary = nested.groupby("model")[list(SCORING)].agg(["mean", "std"])
    print(summary.round(3).to_string())

    # The final model repeats the same tuning on all training rows.
    # Its test score is computed once and never used to change anything.
    final_search = make_search().fit(train_features, train_targets)
    print(f"\nFinal settings: {final_search.best_params_}")
    test_predictions = final_search.predict(test_features)
    test_scores = score_predictions(test_targets, test_predictions)
    print("Test set:", {name: round(v, 3) for name, v in test_scores.items()})

    counts = confusion_matrix(
        test_targets, test_predictions, labels=list(SUBTYPE_NAMES)
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    nested.to_csv(FIGURE_DIR / "final_nested_folds.csv", index=False)
    plot_final(
        nested,
        test_scores["macro_f1"],
        counts,
        FIGURE_DIR / "final_model.png",
    )
    print(f"\nSaved the scores and figure to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
