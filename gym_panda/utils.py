import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    dist = np.linalg.norm(a - b, axis=-1)
    # round at 1e-6 (ensure determinism and avoid numerical noise)
    return np.round(dist, 6)


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist


def normalize_vector(x: np.ndarray) -> np.ndarray:
    """Normalize Vector"""
    norm = np.linalg.norm(x)
    norm = np.where(norm == 0, 1, norm)
    return x / norm


def compute_angle_between(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute angle (radian) between two numpy arrays"""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)

    return np.arccos(dot_prod)
