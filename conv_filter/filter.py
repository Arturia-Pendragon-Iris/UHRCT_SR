import numpy as np
from scipy import ndimage


def normalize_ct(ct_array, hu_min=-1000.0, hu_max=600.0):
    ct_array = np.asarray(ct_array, dtype=np.float32)
    return np.clip((ct_array - hu_min) / (hu_max - hu_min), 0.0, 1.0)


def _hessian_eigenvalues_2d(image, sigma):
    dxx = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0), mode="nearest")
    dxy = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1), mode="nearest")
    dyy = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2), mode="nearest")

    dxx = (sigma ** 2) * dxx
    dxy = (sigma ** 2) * dxy
    dyy = (sigma ** 2) * dyy

    tmp = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
    lambda_a = 0.5 * (dxx + dyy + tmp)
    lambda_b = 0.5 * (dxx + dyy - tmp)

    swap = np.abs(lambda_a) > np.abs(lambda_b)
    lambda1 = lambda_a.copy()
    lambda2 = lambda_b.copy()
    lambda1[swap] = lambda_b[swap]
    lambda2[swap] = lambda_a[swap]
    return lambda1, lambda2


def frangi_filter_2d(
    image,
    sigmas=(0.5, 1.0, 1.5, 2.0),
    beta=0.5,
    gamma=None,
    black_ridges=False,
    normalize=True,
):
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    responses = []
    eps = np.finfo(np.float32).eps
    for sigma in sigmas:
        lambda1, lambda2 = _hessian_eigenvalues_2d(image, float(sigma))
        rb = np.abs(lambda1) / (np.abs(lambda2) + eps)
        s = np.sqrt(lambda1 ** 2 + lambda2 ** 2)
        c = gamma if gamma is not None else max(float(np.max(s)) * 0.5, eps)

        response = np.exp(-(rb ** 2) / (2.0 * beta ** 2))
        response *= 1.0 - np.exp(-(s ** 2) / (2.0 * c ** 2))
        if black_ridges:
            response[lambda2 < 0] = 0.0
        else:
            response[lambda2 > 0] = 0.0
        responses.append(response)

    vesselness = np.max(np.stack(responses, axis=0), axis=0)
    vesselness = np.nan_to_num(vesselness, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        max_value = float(vesselness.max())
        if max_value > 0:
            vesselness = vesselness / max_value
    return vesselness.astype(np.float32)


def perform_filter(np_array, hu_min=-1000.0, hu_max=600.0, **kwargs):
    image = normalize_ct(np_array, hu_min=hu_min, hu_max=hu_max)
    return frangi_filter_2d(image, **kwargs)


def frangi_filter_stack(volume, axis=0, **kwargs):
    volume = np.asarray(volume)
    moved = np.moveaxis(volume, axis, 0)
    filtered = [perform_filter(slice_array, **kwargs) for slice_array in moved]
    return np.moveaxis(np.stack(filtered, axis=0), 0, axis)
