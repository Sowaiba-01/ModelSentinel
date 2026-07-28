"""Framework-agnostic Grad-CAM.

Grad-CAM highlights which spatial regions of a convolutional feature map most
influenced a prediction. This implementation is deliberately framework-neutral:
you extract the target layer's *activations* and their *gradients* from your own
model (PyTorch, TensorFlow, whatever) and pass them in as arrays. That keeps the
core library free of heavy deep-learning dependencies while still working with
real vision models such as DeepGuard's EfficientNet-B4.
"""
from __future__ import annotations

import numpy as np


def grad_cam(activations, gradients) -> np.ndarray:
    """Compute a normalized Grad-CAM heatmap.

    Parameters
    ----------
    activations, gradients:
        Arrays shaped ``[C, H, W]`` or ``[1, C, H, W]`` (a batch of one). These
        are the forward activations of a conv layer and the gradients of the
        target score w.r.t. those activations.

    Returns
    -------
    numpy.ndarray
        An ``H x W`` heatmap scaled to ``[0, 1]``.
    """
    a = np.asarray(activations, dtype=float)
    g = np.asarray(gradients, dtype=float)
    if a.ndim == 4:
        a = a[0]
    if g.ndim == 4:
        g = g[0]
    if a.ndim != 3 or g.ndim != 3:
        raise ValueError("activations and gradients must be [C,H,W] or [1,C,H,W]")

    weights = g.mean(axis=(1, 2))                 # global-average-pool gradients -> channel weights
    cam = np.tensordot(weights, a, axes=([0], [0]))  # weighted sum over channels -> H x W
    cam = np.maximum(cam, 0.0)                    # ReLU
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo + 1e-8)
