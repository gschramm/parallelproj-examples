from __future__ import annotations

from types import ModuleType

from shapes import random_2d_shape_image


def generate_random_image(n: int, xp: ModuleType, ndi: ModuleType):
    """generate a random 3D test image with internal structure

    Parameters
    ----------
    n: int
        image dimensions
    xp : ModuleType
        numpy or cupy module
    ndi : ModuleType
        scipy.ndimage or cupyx.scipy.ndimage module

    Returns
    -------
    numpy or cupy array
    """

    bg_img = random_2d_shape_image(n)

    if xp.__name__ == 'cupy':
        bg_img = xp.asarray(bg_img)

    tmp = bg_img.copy()
    tmp *= xp.random.rand(n, n).astype(xp.float32)
    tmp = (ndi.gaussian_filter(tmp, 1.5) > 0.52)

    label_img, num_labels = ndi.label(tmp)

    img = xp.zeros((n, n)).astype(xp.float32)
    for i in range(1, num_labels + 1):
        inds = xp.where(label_img == i)
        img[inds] = 2 * xp.random.rand() + 1

    return bg_img + img
