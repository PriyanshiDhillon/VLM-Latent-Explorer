import numpy as np

from backend.representation_comparison import cosine_change


def test_cosine_change_identical_is_zero():
    assert abs(cosine_change(np.array([1.0, 2.0]), np.array([1.0, 2.0]))) < 1e-6


def test_cosine_change_orthogonal_is_one():
    assert abs(cosine_change(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 1.0) < 1e-6


def test_cosine_change_rejects_mismatched_shapes():
    assert cosine_change(np.array([1.0]), np.array([1.0, 2.0])) is None


if __name__ == "__main__":
    test_cosine_change_identical_is_zero()
    test_cosine_change_orthogonal_is_one()
    test_cosine_change_rejects_mismatched_shapes()
    print("representation comparison tests passed")
