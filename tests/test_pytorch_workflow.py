"""Check workflow invariants without downloading cancer data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ml import (
    NeuralNetwork,
    OmicsDataset,
    evaluate,
    load_data,
    scale_splits,
    split_indices,
    train_epoch,
)


class WorkflowTests(unittest.TestCase):
    """Protect data splits, scaling, and multiclass evaluation."""

    def test_named_evaluation_preserves_tuple_unpacking(self) -> None:
        """Support named access without breaking existing callers."""
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=1)
        result = evaluate(nn.Identity(), loader, nn.CrossEntropyLoss())
        loss, actual, predicted = result
        self.assertIsInstance(result, tuple)
        self.assertEqual(result.loss, loss)
        self.assertIs(result.targets, actual)
        self.assertIs(result.predictions, predicted)
        np.testing.assert_array_equal(result.predictions, [0, 1])

    def test_splits_are_disjoint_complete_and_repeatable(self) -> None:
        """Partition every row exactly once and reproducibly."""
        targets = np.repeat(np.arange(5), 30)
        splits = split_indices(targets, seed=26)
        repeated = split_indices(targets, seed=26)
        combined = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(combined), np.arange(150))
        for name, indices in splits.items():
            np.testing.assert_array_equal(indices, repeated[name])
            np.testing.assert_array_equal(
                np.unique(targets[indices]), np.arange(5)
            )

    def test_scaling_uses_training_rows_only(self) -> None:
        """Keep held-out observations out of training scaling."""
        features = pd.DataFrame({"gene": [1.0, 3.0, 101.0, -99.0]})
        splits = {
            "train": np.array([0, 1]),
            "validation": np.array([2]),
            "test": np.array([3]),
        }
        scaled, scaler = scale_splits(features, splits)
        np.testing.assert_allclose(scaler.mean_, [2.0])
        np.testing.assert_allclose(scaler.scale_, [1.0])
        np.testing.assert_allclose(scaled["train"].ravel(), [-1.0, 1.0])
        np.testing.assert_allclose(scaled["test"].ravel(), [-101.0])

    def test_evaluation_handles_shuffle_and_partial_batches(self) -> None:
        """Check class selection, label alignment, and weighted loss."""
        sample_ids = torch.arange(135)
        expected = sample_ids % 5
        targets = (expected + (sample_ids % 4 == 0).long()) % 5
        logits = torch.full((135, 5), -2.0)
        logits[sample_ids, expected] = 2.0
        loader = DataLoader(
            TensorDataset(logits, targets),
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(26),
        )
        loss, actual, predicted = evaluate(
            nn.Identity(), loader, nn.CrossEntropyLoss()
        )
        self.assertEqual(actual.shape, (135,))
        self.assertEqual(predicted.shape, (135,))
        self.assertEqual(int((actual == predicted).sum()), 101)
        # Use an independent NumPy log-sum-exp reference.
        scores = logits.numpy().astype(np.float64)
        reference = np.mean(
            np.log(np.exp(scores).sum(axis=1))
            - scores[np.arange(135), targets.numpy()]
        )
        self.assertAlmostEqual(loss, reference, places=6)

    def test_training_learns_and_evaluation_preserves_weights(self) -> None:
        """Learn a simple boundary without updates during evaluation."""
        torch.manual_seed(26)
        features = np.array(
            [[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=np.float32
        )
        targets = np.array([0, 0, 1, 1], dtype=np.int64)
        loader = DataLoader(OmicsDataset(features, targets), batch_size=3)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        initial, _, _ = evaluate(model, loader, loss_fn)
        for _ in range(20):
            train_epoch(model, loader, loss_fn, optimizer)
        before = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        final, actual, predicted = evaluate(model, loader, loss_fn)
        self.assertLess(final, initial)
        np.testing.assert_array_equal(actual, predicted)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_checkpoint_round_trip_preserves_logits(self) -> None:
        """Preserve logits when reloading a state dictionary."""
        model = NeuralNetwork(7, 4, 5)
        features = torch.randn(3, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = NeuralNetwork(7, 4, 5)
            restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(model(features), restored(features))

    def test_loader_rejects_malformed_inputs(self) -> None:
        """Reject bad labels, nonfinite values, and duplicate IDs."""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            expression_path = folder / "BRCA_mRNA.csv"
            label_path = folder / "BRCA_label_num.csv"
            frame = pd.DataFrame(
                [range(5)],
                index=["gene"],
                columns=list("abcde"),
                dtype=float,
            )
            frame.to_csv(expression_path)
            pd.DataFrame({"Label": [0, 1, 2, 3, 4.5]}).to_csv(
                label_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "integers"):
                load_data(folder)
            pd.DataFrame({"Label": range(5)}).to_csv(label_path, index=False)
            frame.iloc[0, 0] = np.inf
            frame.to_csv(expression_path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_data(folder)
            expression_path.write_text(
                ",same,same\ngene,1,2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_data(folder)


if __name__ == "__main__":
    unittest.main()
