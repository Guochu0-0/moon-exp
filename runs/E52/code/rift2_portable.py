"""RIFT2 detector with a floating-point FAST-9 implementation.

The phase congruency below follows Peter Kovesi's phasecong3, as pinned by RIFT2.
The FAST stages follow the published threshold and score definitions. They use
floating-point comparisons and a 3x3 maximum filter, with no claim of MATLAB
bitwise equivalence.
"""

# Copyright (c) 1996-2010 Peter Kovesi
# Centre for Exploration Targeting, The University of Western Australia
# peter.kovesi at uwa.edu.au
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# The Software is provided "as is", without warranty of any kind.

from functools import lru_cache

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import maximum_filter


SCALES = 4
ORIENTATIONS = 6
WAVELENGTH_RATIO = 1.6
EPSILON = 0.0001
FAST_RING = ((0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
             (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3))


@lru_cache(maxsize=4)
def filter_bank(shape):
    def frequencies(length):
        values = np.arange(-(length // 2), (length + 1) // 2) / (length - length % 2)
        return np.fft.ifftshift(values)

    x, y = np.meshgrid(frequencies(shape[1]), frequencies(shape[0]))
    angle = np.arctan2(-y, x)
    radius = np.hypot(x, y)
    lowpass = 1 / (1 + (radius / 0.45) ** 30)
    radius[0, 0] = 1
    scale = 3 * WAVELENGTH_RATIO ** np.arange(SCALES)
    radial = lowpass * np.exp(-np.log(radius[None] * scale[:, None, None]) ** 2 / (2 * np.log(0.75) ** 2))
    radial[:, 0, 0] = 0
    bank = []
    for orientation in np.arange(ORIENTATIONS) * np.pi / ORIENTATIONS:
        difference = np.abs(np.arctan2(np.sin(angle - orientation), np.cos(angle - orientation)))
        spread = (np.cos(np.minimum(difference * ORIENTATIONS / 2, np.pi)) + 1) / 2
        bank.append((orientation, radial * spread))
    return bank


def phase_congruency_max(image):
    spectrum = fft2(image.astype(np.float64))
    cov_xx, cov_yy, cov_xy = np.zeros((3, *image.shape))
    noise_factor = sum(WAVELENGTH_RATIO ** -i for i in range(SCALES))
    rayleigh_energy = (np.sqrt(np.pi / 2) + np.sqrt((4 - np.pi) / 2)) / np.sqrt(np.log(4))
    for orientation, filters in filter_bank(image.shape):
        response = ifft2(filters * spectrum)
        amplitude = np.abs(response)
        mean = response.sum(axis=0)
        mean /= np.abs(mean) + EPSILON
        aligned = response * mean.conjugate()
        noise = np.median(amplitude[0]) * noise_factor * rayleigh_energy
        energy = np.maximum((aligned.real - np.abs(aligned.imag)).sum(axis=0) - noise, 0)
        amplitude_sum = amplitude.sum(axis=0)
        width = (amplitude_sum / (amplitude.max(axis=0) + EPSILON) - 1) / (SCALES - 1)
        weighted = energy / (1 + np.exp(3 * (0.5 - width)))
        pc = np.divide(weighted, amplitude_sum, out=np.zeros_like(weighted), where=amplitude_sum > 0)
        px, py = pc * np.cos(orientation), pc * np.sin(orientation)
        cov_xx += px * px
        cov_yy += py * py
        cov_xy += px * py
    cov_xx *= 2 / ORIENTATIONS
    cov_yy *= 2 / ORIENTATIONS
    cov_xy *= 4 / ORIENTATIONS
    return (cov_xx + cov_yy + np.hypot(cov_xy, cov_xx - cov_yy) + EPSILON) / 2


def fast9(image):
    height, width = image.shape
    center = image[3:-3, 3:-3]
    ring = np.stack([image[3 + y:height - 3 + y, 3 + x:width - 3 + x] - center for x, y in FAST_RING])
    circular = np.concatenate((ring, ring[:8]))
    scores = np.zeros_like(center)
    for start in range(16):
        arc = circular[start:start + 9]
        np.maximum(scores, arc.min(axis=0), out=scores)
        np.maximum(scores, -arc.max(axis=0), out=scores)
    threshold = max(0.0001, 0.0001 * scores.max())
    keep = (scores > threshold) & (scores == maximum_filter(scores, size=3, mode="constant"))
    y, x = np.nonzero(keep)
    selected = scores[y, x]
    strongest = np.argsort(-selected, kind="stable")[:5000]
    return np.column_stack((x[strongest] + 3, y[strongest] + 3, selected[strongest]))


def detect(image):
    response = phase_congruency_max(image)
    low, high = response.min(), response.max()
    if high <= low:
        return np.zeros((0, 3))
    return fast9((response - low) / (high - low))
