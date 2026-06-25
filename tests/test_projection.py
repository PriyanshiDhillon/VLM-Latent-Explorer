import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.projection import compute_neighborhood_preservation_scores


def test_neighborhood_scores_are_one_when_neighbours_are_preserved():
    activations = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    coords = np.column_stack([activations[:, 0], np.zeros(len(activations))])

    scores = compute_neighborhood_preservation_scores(
        activations, coords, n_neighbors=1
    )

    np.testing.assert_allclose(scores, np.ones(4))


def test_neighborhood_scores_detect_a_scrambled_projection():
    activations = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    coords = np.array(
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0]],
        dtype=np.float32,
    )

    scores = compute_neighborhood_preservation_scores(
        activations, coords, n_neighbors=1
    )

    np.testing.assert_allclose(scores, np.zeros(4))


def test_neighborhood_scores_reject_mismatched_token_counts():
    try:
        compute_neighborhood_preservation_scores(
            np.zeros((3, 4), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
    except ValueError as exc:
        assert "same number of tokens" in str(exc)
    else:
        raise AssertionError("Expected mismatched token counts to raise ValueError")


if __name__ == "__main__":
    test_neighborhood_scores_are_one_when_neighbours_are_preserved()
    test_neighborhood_scores_detect_a_scrambled_projection()
    test_neighborhood_scores_reject_mismatched_token_counts()
    print("projection uncertainty tests passed")
