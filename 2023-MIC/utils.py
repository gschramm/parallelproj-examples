from __future__ import annotations

from types import ModuleType


def generate_random_image(n0: int, n1 : int, n2 : int, xp: ModuleType, ndi: ModuleType):
    """generate a random 3D test image with internal structure

    Parameters
    ----------
    n0, n1, n2 : int
        image dimensions
    xp : ModuleType
        numpy or cupy module
    ndi : ModuleType
        scipy.ndimage or cupyx.scipy.ndimage module

    Returns
    -------
    numpy or cupy array
    """    
    bg_img = xp.zeros((n0, n1, n2)).astype(xp.float32)
    bg_img[:, (n1 // 4):((3 * n1) // 4), (n2 // 4):((3 * n2) // 4)] = 1

    tmp = bg_img.copy()
    tmp *= xp.random.rand(n0, n1, n2).astype(xp.float32)
    tmp = (ndi.gaussian_filter(tmp, 1.5) > 0.52)

    label_img, num_labels = ndi.label(tmp)

    img = xp.zeros((n0, n1, n2)).astype(xp.float32)
    for i in range(1, num_labels + 1):
        inds = xp.where(label_img == i)
        img[inds] = 2 * xp.random.rand() + 1

    return bg_img + img

