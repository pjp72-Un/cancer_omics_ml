# Cancer omics machine learning

Cancer multi-omics analysis using PyTorch and the
[MLOmics benchmark](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark).
The aim is to build reproducible models from mRNA expression, miRNA expression,
DNA methylation, and copy number variation data.

This repository currently provides the project structure and Python environment.
Data loading, model training, and evaluation will be developed after choosing the
first cohort and prediction target. No analysis results are reported yet.

## Setup

Use Python 3.11 to 3.13. Python 3.12 is used in CI. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the CPU build of PyTorch on Windows or Linux, followed by the project:

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,notebooks]"
```

For macOS or an NVIDIA GPU, use the appropriate installation command from
[PyTorch](https://pytorch.org/get-started/locally/) in place of the CPU command.
Then run the editable project installation above.

Check the environment:

```bash
python -c "import cancer_omics_ml, torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m ruff check src
python -m ruff format --check src
```

Open notebooks with `python -m jupyterlab`. If needed, register this environment:

```bash
python -m ipykernel install --user --name cancer-omics-ml --display-name "Python (cancer-omics-ml)"
```

## Repository layout

| Location | Purpose |
| --- | --- |
| [src/cancer_omics_ml/](src/cancer_omics_ml/) | Reusable data, modelling, and evaluation code |
| [notebooks/](notebooks/) | Numbered exploratory and analysis notebooks |
| [data/](data/README.md) | Local input data and source documentation |
| `outputs/` | Generated figures, metrics, and run metadata, created as needed |
| [pyproject.toml](pyproject.toml) | Package metadata and Python dependencies |

Datasets, downloaded benchmark code, trained models, and generated outputs are
excluded from Git. Keep notebook outputs cleared before committing, especially
when they contain sample identifiers or clinical information.

## Analysis plan

1. Select a cohort, task, target, and evaluation metric.
2. Audit sample identifiers, feature orientation, missingness, and class balance.
3. Split by patient before fitting preprocessing or selecting features. Keep the
   final test set separate from model selection.
4. Establish a simple baseline, then train a small PyTorch model.
5. Compare held-out performance and record uncertainty, seeds, split membership,
   exact dependency versions, data provenance, and hardware settings locally.

The benchmark includes preprocessed feature sets. Review normalization and
feature-selection provenance before interpreting held-out performance. See the
[data notes](data/README.md).

## Data source

MLOmics is maintained by the upstream authors. Its dataset card declares a
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Consult the
upstream terms and retain attribution when reusing data:

- [MLOmics paper](https://doi.org/10.1038/s41597-025-05235-x)
- [Benchmark code](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark)
- [Dataset distribution](https://huggingface.co/datasets/AIBIC/MLOmics)

Research use only. Models developed here are not clinically validated.
