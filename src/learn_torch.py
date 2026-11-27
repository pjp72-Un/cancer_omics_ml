"""Rebuild logistic regression by hand as a linear PyTorch model."""

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learn_classifier import (
    RANDOM_STATE,
    load_data,
    print_scores,
    score_predictions,
    split_data,
)

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 5
HIDDEN_SIZE = 32
DROPOUT = 0.5
PATIENCE = 5


def to_dataset(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    """Wrap features and labels as a dataset of (row, label) tensors."""
    # Layer weights are float32, so inputs must be float32 too.
    # Cross-entropy expects class IDs as int64 integers.
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.int64),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Update the weights once per batch; return mean loss per sample."""
    model.train()
    total_loss = 0.0
    for batch_features, batch_targets in loader:
        # 1. Clear gradients left over from the previous batch.
        optimizer.zero_grad()
        # 2. Forward pass: [batch, 18206] becomes [batch, 5] logits.
        logits = model(batch_features)
        # 3. One number: how wrong the logits are for these labels.
        loss = loss_function(logits, batch_targets)
        # 4. Backward pass: the gradient of the loss for every weight.
        loss.backward()
        # 5. Move every weight a small step downhill.
        optimizer.step()
        # Weight by batch size so a small final batch is not overcounted.
        total_loss += loss.item() * len(batch_targets)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> tuple[float, np.ndarray]:
    """Return mean loss and predicted class IDs, without learning."""
    # eval() switches layers to evaluation behaviour; no_grad() skips
    # gradient bookkeeping, since nothing is learned here.
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_function(logits, targets).item()
    predictions = logits.argmax(dim=1).numpy()
    return loss, predictions


def make_model(input_size: int, hidden_size: int) -> nn.Module:
    """Return a linear model, or one with a hidden layer if size > 0."""
    if hidden_size == 0:
        return nn.Linear(input_size, NUM_CLASSES)
    # ReLU lets the network combine genes non-linearly. Dropout zeroes
    # a random half of the hidden units on every training batch, so the
    # network cannot lean on a few memorised patterns.
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(hidden_size, NUM_CLASSES),
    )


def train_with_early_stopping(
    model: nn.Module,
    loader: DataLoader,
    validation_inputs: torch.Tensor,
    validation_labels: torch.Tensor,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    verbose: bool = True,
) -> int:
    """Train until validation loss stops improving; keep the best weights.

    Returns the epoch whose weights were kept.
    """
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    if verbose:
        print(f"\n{'epoch':>5} {'fit_loss':>9} {'val_loss':>9} {'val_f1':>7}")
    for epoch in range(1, EPOCHS + 1):
        fit_loss = train_one_epoch(model, loader, loss_function, optimizer)
        validation_loss, validation_predictions = evaluate(
            model, validation_inputs, validation_labels, loss_function
        )
        validation_f1 = score_predictions(
            validation_labels.numpy(), validation_predictions
        )["macro_f1"]

        improved = validation_loss < best_loss
        marker = " *" if improved else ""
        if verbose:
            print(
                f"{epoch:>5} {fit_loss:>9.3f} {validation_loss:>9.3f} "
                f"{validation_f1:>7.3f}{marker}"
            )
        elif epoch - best_epoch >= PATIENCE:
            if verbose:
                print(f"Stopping: no improvement for {PATIENCE} epochs.")
            break

        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            # deepcopy takes a snapshot; without it, best_state would
            # keep changing as training updates the same tensors.
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            print(f"Stopping: no improvement for {PATIENCE} epochs.")
            break

    model.load_state_dict(best_state)
    return best_epoch


def main() -> None:
    """Train a linear classifier and report it on the validation set."""
    torch.manual_seed(RANDOM_STATE)

    features, targets = load_data()
    train_features, _, train_targets, _ = split_data(features, targets)
    fit_features, validation_features, fit_targets, validation_targets = (
        split_data(train_features, train_targets, holdout_fraction=0.25)
    )

    # The scaler learns from fitting rows only, as the Pipeline did.
    scaler = StandardScaler().fit(fit_features)
    fit_data = to_dataset(
        scaler.transform(fit_features), fit_targets.to_numpy()
    )
    validation_data = to_dataset(
        scaler.transform(validation_features), validation_targets.to_numpy()
    )
    validation_inputs, validation_labels = validation_data.tensors

    # shuffle=True gives new batches each epoch; the generator makes
    # that shuffle repeatable.
    loader = DataLoader(
        fit_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    model = make_model(fit_features.shape[1], HIDDEN_SIZE)
    print(model)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Trainable parameters: {parameter_count:,}")

    # CrossEntropyLoss applies softmax itself, so the model outputs raw
    # logits. weight_decay adds an L2 penalty, like 1/C in scikit-learn.
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_epoch = train_with_early_stopping(
        model,
        loader,
        validation_inputs,
        validation_labels,
        loss_function,
        optimizer,
    )
    print(f"\nKept weights from epoch {best_epoch}.")

    _, validation_predictions = evaluate(
        model, validation_inputs, validation_labels, loss_function
    )
    final_scores = score_predictions(
        validation_targets, validation_predictions
    )
    print_scores(
        f"PyTorch, hidden size {HIDDEN_SIZE}, validation", final_scores
    )


if __name__ == "__main__":
    main()
