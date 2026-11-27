"""Compare logistic regression and PyTorch models on shared folds."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from learn_classifier import (
    CV_FOLDS,
    RANDOM_STATE,
    load_data,
    make_logistic_pipeline,
    score_predictions,
    split_data,
)
from learn_torch import (
    BATCH_SIZE,
    LEARNING_RATE,
    make_model,
    to_dataset,
    train_with_early_stopping,
)

CHOSEN_C = 0.1
TORCH_SETTINGS = {
    "torch_linear": {"hidden_size": 0, "weight_decay": 0.025},
    "torch_hidden_32": {"hidden_size": 32, "weight_decay": 0.0001},
}


def fit_torch_and_predict(
    train_features: pd.DataFrame,
    train_targets: pd.Series,
    evaluation_features: pd.DataFrame,
    hidden_size: int,
    weight_decay: float,
) -> tuple[np.ndarray, int]:
    """Train on one fold's training rows, then predict its evaluation rows.

    Returns the predictions and the epoch whose weights were kept.
    """
    # Early stopping needs its own validation rows. They come from this
    # fold's training rows, so the evaluation fold stays unseen.
    fit_features, stop_features, fit_targets, stop_targets = split_data(
        train_features, train_targets, holdout_fraction=0.25
    )
    scaler = StandardScaler().fit(fit_features)
    fit_data = to_dataset(
        scaler.transform(fit_features), fit_targets.to_numpy()
    )
    stop_inputs, stop_labels = to_dataset(
        scaler.transform(stop_features), stop_targets.to_numpy()
    ).tensors

    # Reseed so every fit starts from repeatable weights and batches.
    torch.manual_seed(RANDOM_STATE)
    loader = DataLoader(
        fit_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    model = make_model(fit_features.shape[1], hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )
    best_epoch = train_with_early_stopping(
        model,
        loader,
        stop_inputs,
        stop_labels,
        loss_function,
        optimizer,
        verbose=False,
    )

    evaluation_inputs = torch.tensor(
        scaler.transform(evaluation_features), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        predictions = model(evaluation_inputs).argmax(dim=1).numpy()
    return predictions, best_epoch


def main() -> None:
    """Score every model on the same five folds of the training set."""
    features, targets = load_data()
    train_features, _, train_targets, _ = split_data(features, targets)

    # Same seed and data as step 6, so these are the same five folds
    # that GridSearchCV used to choose C.
    folds = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    rows = []
    fold_splits = folds.split(train_features, train_targets)
    for fold_number, (train_rows, evaluation_rows) in enumerate(
        fold_splits, start=1
    ):
        # split() returns row positions, so select with iloc.
        fold_train_features = train_features.iloc[train_rows]
        fold_train_targets = train_targets.iloc[train_rows]
        evaluation_features = train_features.iloc[evaluation_rows]
        evaluation_targets = train_targets.iloc[evaluation_rows]

        logistic_model = make_logistic_pipeline(inverse_penalty=CHOSEN_C)
        logistic_model.fit(fold_train_features, fold_train_targets)
        predictions = logistic_model.predict(evaluation_features)
        scores = score_predictions(evaluation_targets, predictions)
        rows.append(
            {"fold": fold_number, "model": "sklearn_logistic", **scores}
        )

        for model_name, settings in TORCH_SETTINGS.items():
            predictions, best_epoch = fit_torch_and_predict(
                fold_train_features,
                fold_train_targets,
                evaluation_features,
                **settings,
            )
            scores = score_predictions(evaluation_targets, predictions)
            rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "best_epoch": best_epoch,
                    **scores,
                }
            )
        print(f"Finished fold {fold_number} of {CV_FOLDS}")

    results = pd.DataFrame(rows)
    f1_by_fold = results.pivot(
        index="fold", columns="model", values="macro_f1"
    )
    print("\nMacro F1 by fold (each row is the same samples):")
    print(f1_by_fold.round(3).to_string())
    print("\nMean and standard deviation across folds:")
    print(f1_by_fold.agg(["mean", "std"]).round(3).to_string())

    torch_results = results[results["model"] != "sklearn_logistic"]
    epochs = torch_results.pivot(
        index="fold", columns="model", values="best_epoch"
    )
    print("\nEpochs kept by early stopping:")
    print(epochs.astype(int).to_string())

    differences = (
        f1_by_fold["torch_hidden_32"] - f1_by_fold["sklearn_logistic"]
    )
    print("\nHidden network minus logistic regression, per fold:")
    print(differences.round(3).to_string())
    wins = (differences > 0).sum()
    print(f"Hidden network wins {wins} of {CV_FOLDS} folds.")


if __name__ == "__main__":
    main()
