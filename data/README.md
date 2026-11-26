# Local data

The input data come from
[MLOmics](https://huggingface.co/datasets/AIBIC/MLOmics) and the accompanying
[Cancer Multi-Omics Benchmark](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark).
The dataset card declares CC BY 4.0. Cite the
[MLOmics paper](https://doi.org/10.1038/s41597-025-05235-x) and follow the upstream
attribution requirements.

## Local arrangement

The existing benchmark checkout is stored at `data/Cancer-Multi-Omics-Benchmark/`.
Its task-organized CSV inputs are under `data/Cancer-Multi-Omics-Benchmark/datasets/`.
This directory and its nested repositories remain local. Only this README is
tracked under `data/`.

For a new checkout, download data from the linked upstream sources and place
them under `data/`. Follow their current download instructions. Upstream releases
may use `Main_Dataset/` instead of `datasets/`, so record the actual input path
when configuring an analysis. Cloning this project does not download the data.

## Checks before modelling

- Record the upstream revision or dataset release, input paths, and file hashes.
- Upstream describes omics CSVs as features in rows and samples in columns.
  Confirm this for each selected file before transposing for PyTorch.
- Align modalities and labels by explicit sample identifiers. Validate uniqueness
  and matching sample sets rather than relying on column order.
- Check missing values, nonfinite values, target definitions, and class balance.
- Upstream describes z-score-normalized values and ANOVA-selected `Top` features.
  Establish whether those operations used the full cohort or labels from the
  evaluation set. Prefer source features that allow preprocessing and feature
  selection to be fitted within the training split where available. Even an
  `Original` feature set may already have undergone normalization.
- Keep raw inputs unchanged. Save derived data and run metadata locally.
