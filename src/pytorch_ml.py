"""Learn PyTorch with GS-BRCA mRNA data in eight numbered stages.

Run: python src/pytorch_ml.py --epochs 30
Read the README learning path alongside this script. Importing this
module does not load data, train a model, or create output files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "data/Cancer-Multi-Omics-Benchmark/datasets"
    / "Classification_datasets/GS-BRCA/Original"
)
SEED = 26
BATCH_SIZE = 64
HIDDEN_DIM = 32
LEARNING_RATE = 0.001
THREADS = 2


class EvaluationResult(NamedTuple):
    """Expose evaluation values by name and through tuple unpacking.

    Attributes:
        loss: Sample-weighted mean cross-entropy.
        targets: True class IDs in loader order.
        predictions: Predicted class IDs in the same order.
    """

    loss: float
    targets: NDArray[np.int64]
    predictions: NDArray[np.int64]


# Stage 1: Establish what each row, column, and label represents.
def load_data(data_dir: Path) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Read the original mRNA matrix and positional class labels.

    Args:
        data_dir: Directory containing the two GS-BRCA CSV files.

    Returns:
        Sample-by-gene measurements and integer class IDs.

    Raises:
        ValueError: Inputs have invalid identifiers, values, or shapes.
    """
    expression_path = data_dir / "BRCA_mRNA.csv"
    # Check before pandas can rename duplicate column identifiers.
    with expression_path.open(encoding="utf-8-sig", newline="") as stream:
        sample_ids = next(csv.reader(stream), [])[1:]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample identifiers must be nonempty and unique.")
    if any(not sample_id.strip() for sample_id in sample_ids):
        raise ValueError("A sample identifier is blank.")
    # TCGA aliquots from one patient must never cross split boundaries.
    if all(sample_id.startswith("TCGA-") for sample_id in sample_ids):
        patient_ids = [sample_id[:12] for sample_id in sample_ids]
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError("Repeated TCGA patients need grouped splits.")
    features = pd.read_csv(expression_path, index_col=0).T
    label_frame = pd.read_csv(data_dir / "BRCA_label_num.csv")
    if list(label_frame.columns) != ["Label"]:
        raise ValueError("Expected one numeric Label column.")
    if features.empty or len(features) != len(label_frame):
        raise ValueError("Features and labels need matching nonempty rows.")
    if not features.columns.is_unique or features.columns.hasnans:
        raise ValueError("Gene identifiers must be present and unique.")
    values = features.to_numpy(dtype=np.float32)
    raw_targets = label_frame["Label"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(raw_targets).all():
        raise ValueError("Inputs contain missing or nonfinite values.")
    if not np.equal(raw_targets, np.floor(raw_targets)).all():
        raise ValueError("Class labels must be integers.")
    targets = raw_targets.astype(np.int64)
    if not np.array_equal(np.unique(targets), np.arange(5)):
        raise ValueError("This lesson expects GS-BRCA labels 0 through 4.")
    warnings.warn(
        "Learning dataset: labels have no sample IDs, so alignment "
        "assumes upstream row order. Upstream normalization is also "
        "unverified. Scores are not validated biological results.",
        UserWarning,
        stacklevel=2,
    )
    return features, targets


# Stage 2: Split before fitting any new preprocessing.
def split_indices(
    targets: NDArray[np.int64], seed: int
) -> dict[str, NDArray[np.int64]]:
    """Make reproducible, stratified train/validation/test partitions.

    Args:
        targets: One class ID per sample.
        seed: Random seed for both split operations.

    Returns:
        Row indices for approximately 60/20/20 percent of the samples.
    """
    train_val, test = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=targets[train_val],
        random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def scale_splits(
    features: pd.DataFrame, splits: dict[str, NDArray[np.int64]]
) -> tuple[dict[str, NDArray[np.float32]], StandardScaler]:
    """Fit scaling on training rows and apply it to all partitions.

    Args:
        features: Sample-by-gene measurements in original row order.
        splits: Train, validation, and test row indices.

    Returns:
        Float32 arrays for each partition and the fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(features.iloc[splits["train"]])
    scaled = {
        name: scaler.transform(features.iloc[indices]).astype(np.float32)
        for name, indices in splits.items()
    }
    return scaled, scaler


# Stage 3: A Dataset returns one sample; a DataLoader builds batches.
class OmicsDataset(Dataset):
    """Pair float32 features with int64 multiclass targets."""

    def __init__(
        self, features: NDArray[np.float32], targets: NDArray[np.int64]
    ) -> None:
        """Create tensors from aligned feature rows and class IDs.

        Args:
            features: Two-dimensional sample-by-feature array.
            targets: One-dimensional integer labels for those samples.
        """
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("Expected 2D features and 1D targets.")
        if len(features) != len(targets) or len(targets) == 0:
            raise ValueError("Features and targets need equal nonzero rows.")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.int64)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the feature vector and class ID at the given index."""
        return self.features[index], self.targets[index]


# Stage 4: nn.Module owns parameters; forward defines the computation.
class NeuralNetwork(nn.Module):
    """Map gene measurements to five unnormalized class scores."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        """Build a hidden layer, ReLU activation, and output layer.

        Args:
            input_dim: Number of measured genes.
            hidden_dim: Number of hidden neurons.
            output_dim: Number of target classes.
        """
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits shaped [batch_size, output_dim].

        Args:
            features: Float tensor shaped [batch_size, input_dim].

        Returns:
            Raw class scores, without sigmoid or softmax.
        """
        hidden = self.activation(self.hidden(features))
        return self.output(hidden)


# Stage 5: Keep the gradient-update operations visible.
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Update parameters once per batch and return mean sample loss.

    Args:
        model: Network to train on CPU.
        loader: Nonempty training batches.
        loss_fn: A loss using mean reduction, such as CrossEntropyLoss.
        optimizer: Optimizer bound to the model parameters.

    Returns:
        Sample-weighted mean loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        # Gradients accumulate unless we explicitly clear them.
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("Training loss is nonfinite; inspect inputs.")
        loss.backward()
        optimizer.step()
        # Weight by sample count because the final batch may be smaller.
        total_loss += loss.item() * len(targets)
        total_samples += len(targets)
    if total_samples == 0:
        raise ValueError("Training loader is empty.")
    return total_loss / total_samples


# Stage 6: Collect targets alongside their predictions.
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module
) -> EvaluationResult:
    """Evaluate without gradients, preserving label alignment.

    Args:
        model: Network to evaluate on CPU.
        loader: Nonempty validation or test batches.
        loss_fn: Classification loss using mean reduction.

    Returns:
        Mean sample loss, true labels, and predicted class IDs.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features)
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Evaluation loss is nonfinite.")
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
            predictions.extend(logits.argmax(dim=1).tolist())
            true_labels.extend(targets.tolist())
    if total_samples == 0:
        raise ValueError("Evaluation loader is empty.")
    return EvaluationResult(
        loss=total_loss / total_samples,
        targets=np.asarray(true_labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
    )


def main(epochs: int = 30, output_dir: Path | None = None) -> None:
    """Run the numbered learning stages and save local results.

    Args:
        epochs: Positive number of passes through the training data.
        output_dir: Destination for generated files.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    output_dir = output_dir or PROJECT_ROOT / "outputs/pytorch_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)

    features, targets = load_data(DATA_DIR)
    splits = split_indices(targets, seed=SEED)
    scaled, scaler = scale_splits(features, splits)
    generator = torch.Generator().manual_seed(SEED)
    loaders = {
        name: DataLoader(
            OmicsDataset(scaled[name], targets[indices]),
            batch_size=BATCH_SIZE,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
        )
        for name, indices in splits.items()
    }
    model = NeuralNetwork(
        input_dim=features.shape[1], hidden_dim=HIDDEN_DIM, output_dim=5
    )
    print(f"Samples x genes: {features.shape}")
    sizes = {name: len(indices) for name, indices in splits.items()}
    print(f"Split sizes: {sizes}")
    # Index the Dataset without consuming the loader's shuffle state.
    sample_features, sample_target = loaders["train"].dataset[0]
    print("One sample:", sample_features.shape)
    print("One target:", sample_target.dtype)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], loss_fn, optimizer)
        validation = evaluate(model, loaders["validation"], loss_fn)
        val_accuracy = accuracy_score(
            validation.targets, validation.predictions
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation.loss,
                "validation_accuracy": val_accuracy,
            }
        )
        if validation.loss < best_loss:
            best_loss, best_epoch = validation.loss, epoch
            # Freeze this epoch's weights while training continues.
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:02d}: train={train_loss:.4f}, "
            f"validation={validation.loss:.4f}, "
            f"val accuracy={val_accuracy:.3f}"
        )

    # Stage 7: Restore the validation-selected model, then test once.
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": features.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "output_dim": 5,
            "feature_names": features.columns.tolist(),
            "scaler_mean": torch.from_numpy(scaler.mean_),
            "scaler_scale": torch.from_numpy(scaler.scale_),
        },
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_true, test_pred = evaluate(model, loaders["test"], loss_fn)
    majority_class = int(np.bincount(targets[splits["train"]]).argmax())
    baseline_pred = np.full_like(test_true, majority_class)
    class_ids = list(range(5))
    metrics = {
        "best_epoch": best_epoch,
        "test_samples": len(test_true),
        "test_loss": test_loss,
        "accuracy": accuracy_score(test_true, test_pred),
        "balanced_accuracy": balanced_accuracy_score(test_true, test_pred),
        "macro_f1": f1_score(
            test_true, test_pred, average="macro", zero_division=0
        ),
        "majority_baseline_accuracy": accuracy_score(test_true, baseline_pred),
        "classification_report": classification_report(
            test_true,
            test_pred,
            labels=class_ids,
            output_dict=True,
            zero_division=0,
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": features.index[splits["test"]],
            "true_class": test_true,
            "predicted_class": test_pred,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(
        [
            pd.DataFrame({"sample_id": features.index[indices], "split": name})
            for name, indices in splits.items()
        ],
        ignore_index=True,
    ).to_csv(output_dir / "splits.csv", index=False)

    # Stage 8: Curves reveal overfitting hidden by final accuracy.
    figure = Figure(figsize=(10, 4), layout="constrained")
    loss_axis, matrix_axis = figure.subplots(1, 2)
    table = pd.DataFrame(history)
    loss_axis.plot(table["epoch"], table["train_loss"], label="Training")
    loss_axis.plot(
        table["epoch"], table["validation_loss"], label="Validation"
    )
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.axvline(
        best_epoch, color="0.5", linestyle=":", label="Selected epoch"
    )
    loss_axis.legend()
    matrix = confusion_matrix(test_true, test_pred, labels=class_ids)
    matrix_axis.imshow(matrix, cmap="Blues")
    for (row, column), count in np.ndenumerate(matrix):
        matrix_axis.text(
            column,
            row,
            str(count),
            ha="center",
            va="center",
            color="white" if count > matrix.max() / 2 else "black",
        )
    matrix_axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        xticks=class_ids,
        yticks=class_ids,
        title="Test confusion matrix",
    )
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    input_hashes = {}
    for name in ("BRCA_mRNA.csv", "BRCA_label_num.csv"):
        with (DATA_DIR / name).open("rb") as stream:
            input_hashes[name] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    provenance = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "device": "cpu",
        "threads": THREADS,
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("torch", "numpy", "pandas", "scikit-learn")
        },
        "alignment": "Assumed upstream positional label order; unverified",
        "preprocessing": "Upstream normalization provenance unverified",
        "input_sha256": input_hashes,
    }
    (output_dir / "run.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Selected epoch: {best_epoch}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Majority baseline: {metrics['majority_baseline_accuracy']:.3f}")
    print(f"Saved learning outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    main(epochs=arguments.epochs, output_dir=arguments.output_dir)
