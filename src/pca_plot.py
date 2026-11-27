"""Unsupervised view: do the subtypes separate without using labels?"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from inspect_model import FIGURE_DIR, SUBTYPE_NAMES
from learn_classifier import PROJECT_ROOT, RANDOM_STATE, load_data, split_data

# plot_style sits at the repository root, outside src.
sys.path.insert(0, str(PROJECT_ROOT))
from plot_style.theme import INK, plot_theme, save_figure  # noqa: E402

# Colour and marker per subtype. The colours were checked for
# colour-blind separation; the shapes mean colour is never the only cue.
SUBTYPE_STYLES = {
    "LumA": {"color": "#4A7FB8", "marker": "o"},
    "Her2": {"color": "#EB6834", "marker": "D"},
    "LumB": {"color": "#2BAA8E", "marker": "s"},
    "Normal": {"color": "#E3AA00", "marker": "^"},
    "Basal": {"color": "#7B2E8E", "marker": "v"},
}


def project_samples(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    """Return the first two principal components and variance shares."""
    # PCA never sees the labels: it only finds the directions along
    # which the samples vary most across all genes.
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    projected = pd.DataFrame(
        components, index=features.index, columns=["PC1", "PC2"]
    )
    return projected, list(pca.explained_variance_ratio_)


def plot_pca(
    projected: pd.DataFrame,
    names: pd.Series,
    variance: list[float],
    gene_count: int,
    path: Path,
) -> None:
    """Scatter PC1 against PC2, one colour and marker per subtype."""
    counts = names.value_counts()
    # Rank 0 is the largest group. Higher zorder draws on top, so the
    # small groups are not hidden under the large ones.
    draw_rank = {subtype: rank for rank, subtype in enumerate(counts.index)}
    with plot_theme(base_size=11, grid="none"):
        figure, axis = plt.subplots(figsize=(10, 6.2))
        for subtype in SUBTYPE_NAMES.values():
            style = SUBTYPE_STYLES[subtype]
            rows = projected[names == subtype]
            axis.scatter(
                rows["PC1"], rows["PC2"], s=34, alpha=0.8,
                color=style["color"], marker=style["marker"],
                edgecolors=INK, linewidths=0.4,
                zorder=2 + draw_rank[subtype],
                label=f"{subtype}  (n={counts[subtype]})",
            )  # fmt: skip

        axis.set_xlabel(f"PC1 ({variance[0]:.0%} of variance)")
        axis.set_ylabel(f"PC2 ({variance[1]:.0%} of variance)")
        # PCA coordinates have no useful units, so hide the numbers.
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(
            title="PAM50 subtype", loc="center left",
            bbox_to_anchor=(1.02, 0.5), markerscale=1.4,
        )  # fmt: skip

        figure.suptitle(
            "Basal tumours separate without labels; the luminal "
            "subtypes overlap",
            x=0.03, y=0.96, ha="left", weight="bold", fontsize=13,
        )  # fmt: skip
        figure.subplots_adjust(left=0.07, right=0.78, top=0.88, bottom=0.14)
        save_figure(figure, path)
        plt.close(figure)


def main() -> None:
    """Project the training samples with PCA and save the figure."""
    features, targets = load_data()
    train_features, _, train_targets, _ = split_data(features, targets)

    projected, variance = project_samples(train_features)
    print(f"Variance explained: PC1 {variance[0]:.1%}, PC2 {variance[1]:.1%}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_pca(
        projected,
        train_targets.map(SUBTYPE_NAMES),
        variance,
        train_features.shape[1],
        FIGURE_DIR / "pca_subtypes.png",
    )
    print(f"Saved the figure to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
