"""Look inside the 500-gene model: which genes did it choose, and why?"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from learn_classifier import PROJECT_ROOT, load_data, split_data
from select_features import make_selected_pipeline

# plot_style sits at the repository root, outside src, so add the root
# to the import path before importing it.
sys.path.insert(0, str(PROJECT_ROOT))
from plot_style.theme import (  # noqa: E402
    GRID,
    INK,
    THEME_COLOURS,
    plot_theme,
    save_figure,
)

GENE_COUNT = 500
TOP_PER_SUBTYPE = 8
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"

# Label IDs from the benchmark's GS-Subtype_labels.csv.
SUBTYPE_NAMES = {0: "LumA", 1: "Her2", 2: "LumB", 3: "Normal", 4: "Basal"}

# The 50 genes of the PAM50 assay, which defines these subtypes.
PAM50_GENES = {
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B",
    "TYMS", "UBE2C", "UBE2T",
}  # fmt: skip


def fit_and_extract(
    features: pd.DataFrame, targets: pd.Series
) -> pd.DataFrame:
    """Fit the pipeline; return a genes-by-subtypes weight table."""
    pipeline = make_selected_pipeline(GENE_COUNT)
    pipeline.fit(features, targets)
    # Index a pipeline by step name to reach each fitted step.
    kept = pipeline["selector"].get_support()
    weights = pipeline["classifier"].coef_
    # coef_ is [5 subtypes, 500 genes]; transpose to one row per gene.
    return pd.DataFrame(
        weights.T,
        index=features.columns[kept],
        columns=[SUBTYPE_NAMES[label] for label in pipeline.classes_],
    )


def top_genes(weights: pd.DataFrame) -> pd.DataFrame:
    """Return the most positive genes for every subtype, in long form."""
    rows = []
    for subtype in weights.columns:
        strongest = weights[subtype].nlargest(TOP_PER_SUBTYPE)
        for rank, (gene, weight) in enumerate(strongest.items(), start=1):
            rows.append(
                {
                    "subtype": subtype,
                    "rank": rank,
                    "gene": gene,
                    "weight": weight,
                    "in_pam50": gene in PAM50_GENES,
                }
            )
    return pd.DataFrame(rows)


def report_pam50(selected_genes: pd.Index, gene_total: int) -> list[str]:
    """Print how many PAM50 genes were selected, against chance."""
    found = sorted(PAM50_GENES & set(selected_genes))
    missed = sorted(PAM50_GENES - set(selected_genes))
    expected = len(PAM50_GENES) * len(selected_genes) / gene_total
    print(
        f"\nPAM50 genes among the {len(selected_genes)} selected: "
        f"{len(found)} of {len(PAM50_GENES)}"
    )
    print(f"Expected by chance: {expected:.1f}")
    print(f"Not selected: {', '.join(missed)}")
    return found


def order_by_peak_subtype(
    features: pd.DataFrame, names: pd.Series, genes: list[str]
) -> tuple[list[str], pd.Series]:
    """Group genes by the subtype where their mean expression is highest."""
    subtypes = list(SUBTYPE_NAMES.values())
    # groupby(names) averages each gene within each subtype.
    means = features[genes].groupby(names).mean().reindex(subtypes)
    peak = means.idxmax()

    def sort_key(gene: str) -> tuple[int, float]:
        """Sort by peak subtype, then strongest peak first within it."""
        return subtypes.index(peak[gene]), -means[gene].max()

    ordered = sorted(genes, key=sort_key)
    return ordered, peak[ordered]


def plot_pam50_heatmap(
    features: pd.DataFrame,
    targets: pd.Series,
    genes: list[str],
    path: Path,
) -> None:
    """Heatmap of selected PAM50 genes across samples, grouped by subtype."""
    subtypes = list(SUBTYPE_NAMES.values())
    names = targets.map(SUBTYPE_NAMES)
    genes, peak = order_by_peak_subtype(features, names, genes)

    # Columns: samples sorted by subtype. A stable sort keeps the
    # original order within each subtype.
    sample_order = names.map(subtypes.index).sort_values(kind="stable").index
    matrix = features.loc[sample_order, genes].T.clip(-3, 3)

    group_sizes = names.value_counts().reindex(subtypes)
    column_ends = group_sizes.cumsum().to_numpy()
    column_starts = np.concatenate([[0], column_ends[:-1]])
    row_ends = peak.value_counts().reindex(subtypes).fillna(0).cumsum()

    # Diverging scale: blue below average, neutral at 0, orange above.
    colour_map = LinearSegmentedColormap.from_list(
        "blue_neutral_orange",
        ["#2F5F8A", THEME_COLOURS["sky"], "#ECEAE3",
         THEME_COLOURS["orange"], "#A3401A"],
    )  # fmt: skip
    with plot_theme(base_size=10, grid="none"):
        figure, axis = plt.subplots(figsize=(12, 8))
        image = axis.imshow(
            matrix, aspect="auto", cmap=colour_map, vmin=-3, vmax=3,
            interpolation="nearest",
        )  # fmt: skip
        # White gaps separate the subtype groups in both directions.
        for column_end in column_ends[:-1]:
            axis.axvline(column_end - 0.5, color="white", linewidth=2.5)
        for row_end in row_ends.unique()[:-1]:
            axis.axhline(row_end - 0.5, color="white", linewidth=2.5)

        centres = (column_starts + column_ends) / 2 - 0.5
        axis.set_xticks(
            centres,
            [f"{name}\n(n={size})" for name, size in group_sizes.items()],
        )
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", colors=INK, labelsize=10)
        axis.set_yticks(range(len(genes)), genes, fontsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)

        colour_bar = figure.colorbar(image, ax=axis, shrink=0.45, pad=0.02)
        colour_bar.set_label("Expression z-score (clipped at ±3)")
        colour_bar.outline.set_edgecolor(GRID)

        figure.suptitle(
            f"The model chose {len(genes)} of the 50 PAM50 subtype genes "
            f"without being told about them",
            x=0.01, y=0.975, ha="left", weight="bold", fontsize=13,
        )  # fmt: skip
        figure.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.05)
        save_figure(figure, path)
        plt.close(figure)


def main() -> None:
    """Fit on the training set, report the genes, and save a figure."""
    features, targets = load_data()
    train_features, _, train_targets, _ = split_data(features, targets)
    gene_total = features.shape[1]

    weights = fit_and_extract(train_features, train_targets)
    top = top_genes(weights)
    print("Largest positive weights per subtype:")
    print(top.round(3).to_string(index=False))
    found = report_pam50(weights.index, gene_total)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(FIGURE_DIR / "top_gene_weights.csv", index=False)
    plot_pam50_heatmap(
        train_features,
        train_targets,
        found,
        FIGURE_DIR / "pam50_heatmap.png",
    )
    print(f"\nSaved the table and figure to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
