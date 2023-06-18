"""utils to generate images containing a pseudo-random shape
   based on 
   https://stackoverflow.com/a/50751932
"""
import numpy as np
from scipy.special import binom
import scipy.spatial
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from types import ModuleType

bernstein = lambda n, k, t: binom(n, k) * t**k * (1. - t)**(n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():

    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1)**2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array(
            [self.r * np.cos(self.angle1), self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([
            self.r * np.cos(self.angle2 + np.pi),
            self.r * np.sin(self.angle2 + np.pi)
        ])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2],
                      points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


def random_2d_shape_image(n,
                          rad=0.25,
                          edgy=0.05,
                          num_points=11,
                          dtype=np.float32):
    img = np.zeros((n, n), dtype=dtype)

    x, y, _ = get_bezier_curve(get_random_points(n=num_points, scale=1),
                               rad=rad,
                               edgy=edgy)

    for point in np.vstack((x, y)).T:
        i0 = int((0.1 + 0.8 * point[0]) * n)
        i1 = int((0.1 + 0.8 * point[1]) * n)

        if i0 < 0: i0 = 0
        if i1 < 0: i1 = 0

        if i0 >= n: i0 = n - 1
        if i1 >= n: i1 = n - 1

        img[i0, i1] = 1

    return ndi.binary_fill_holes(
        ndi.gaussian_filter(img, 0.5) > 0).astype(dtype)


def flood_fill_hull(image):
    """ generate the convex hull of a 3D binary image"""
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img


def generate_random_3d_image(n_trans: int, n_ax: int, xp: ModuleType,
                             ndi: ModuleType):
    """generate a random 3D test image with internal structure

    Parameters
    ----------
    n_trans, n_ax: int
        number of voxels in trans-axial and axial direction
    xp : ModuleType
        numpy or cupy module
    ndi : ModuleType
        scipy.ndimage or cupyx.scipy.ndimage module

    Returns
    -------
    numpy or cupy array
    """

    n_t = 4

    tmp = np.zeros((n_t, n_trans, n_trans, n_ax), dtype=np.float32)
    tmp2 = np.zeros((n_t, n_trans, n_trans, n_ax), dtype=np.float32)

    for i in range(n_t):
        tmp[i, :, :, 0] = random_2d_shape_image(n_trans)
        tmp[i, :, :, -1] = random_2d_shape_image(n_trans)

    for i in range(n_t):
        tmp2[i, ...] = flood_fill_hull(tmp[i, ...])

    bg_img = (tmp2.sum(0) > 0)

    if xp.__name__ == 'cupy':
        bg_img = xp.array(bg_img)

    tmp = bg_img.copy().astype(xp.float32)
    tmp *= xp.random.rand(*bg_img.shape).astype(xp.float32)
    tmp = (ndi.gaussian_filter(tmp, 3) > 0.51)

    label_img, num_labels = ndi.label(tmp)

    img = xp.zeros(bg_img.shape).astype(xp.float32)
    for i in range(1, num_labels + 1):
        inds = xp.where(label_img == i)
        img[inds] = 4 * xp.random.rand() - 1

    scale = float(0.4 * xp.random.rand() + 0.8)
    exp = float(0.2 * xp.random.rand() + 0.9)

    complete_img = bg_img + img
    complete_img = (scale * complete_img)**exp

    return complete_img
