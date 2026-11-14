"""Least-squares 2-D transforms used by face alignment and the flatness test."""
import cv2
import numpy as np


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity (rotation, uniform scale, translation) mapping src -> dst."""
    n = len(src)
    a = np.zeros((2 * n, 4), np.float64)
    a[0::2] = np.c_[src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)]
    a[1::2] = np.c_[src[:, 1], src[:, 0], np.zeros(n), np.ones(n)]
    b = dst.reshape(-1).astype(np.float64)
    p, a_, tx, ty = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.array([[p, -a_, tx], [a_, p, ty]], np.float32)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def similarity_residual(src: np.ndarray, dst: np.ndarray) -> float:
    m = fit_similarity(src, dst)
    pred = src @ m[:, :2].T + m[:, 2]
    return rms(pred, dst)


def homography_residual(src: np.ndarray, dst: np.ndarray) -> float:
    h, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
    if h is None:
        return float("nan")
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), h).reshape(-1, 2)
    return rms(pred, dst)
