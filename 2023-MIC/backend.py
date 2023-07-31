from __future__ import annotations

import abc
import math
import numpy as np
import numpy.typing as npt
import cupy as cp
import array_api_compat

import requests
import zipfile
import io
from pathlib import Path

from types import ModuleType

cuda_kernel_file = 'projector_kernels.cu'

# load a kernel defined in a external file
with open(cuda_kernel_file, 'r', encoding='utf8') as f:
    lines = f.read()

_joseph3d_fwd_cuda_kernel = cp.RawKernel(
    lines, 'joseph3d_fwd_cuda_kernel')
_joseph3d_back_cuda_kernel = cp.RawKernel(
    lines, 'joseph3d_back_cuda_kernel')
 

def download_data(
        zip_file_url:
    str = 'https://zenodo.org/record/8067595/files/brainweb_petmr_v2.zip',
        force: bool = False,
        out_path: Path | None = None):
    """download simulated brainweb PET/MR images

    Parameters
    ----------
    zip_file_url : str, optional
        by default 'https://zenodo.org/record/8067595/files/brainweb_petmr_v2.zip'
    force : bool, optional
        force download even if data is already present, by default False
    out_path : Path | None, optional
        _output path for the data, by default None
    """

    if out_path is None:
        out_path = Path('.') / 'data'
    out_path.mkdir(parents=True, exist_ok=True)

    if not (out_path / 'subject54').exists() or force:
        print('downloading data')
        r = requests.get(zip_file_url)
        print('download finished')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_path)
        print(f'extracted data into {out_path}')
    else:
        print('data already present')


def joseph3d_fwd(xstart: npt.ArrayLike,
                 xend: npt.ArrayLike,
                 img: npt.ArrayLike,
                 img_origin: npt.ArrayLike,
                 voxsize: npt.ArrayLike,
                 threadsperblock: int = 32) -> npt.ArrayLike:
    """Non-TOF Joseph 3D forward projector

    Parameters
    ----------
    xstart : npt.ArrayLike (numpy/cupy array or torch tensor)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : npt.ArrayLike (numpy/cupy array or torch tensor)
        end world coordinates of the LORs, shape (nLORs, 3)
    img : npt.ArrayLike (numpy/cupy array or torch tensor)
        containing the 3D image to be projected
    img_origin : npt.ArrayLike (numpy/cupy array or torch tensor)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : npt.ArrayLike (numpy/cupy array or torch tensor)
        array containing the voxel size
    threadsperblock : int, optional
        by default 32
    """
    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img)

    # projection of GPU array (cupy to torch GPU array) using the cupy raw kernel
    img_fwd = cp.zeros(xstart.shape[:-1], dtype=cp.float32)

    _joseph3d_fwd_cuda_kernel(
        (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
        (cp.asarray(xstart, dtype=cp.float32).ravel(),
         cp.asarray(xend, dtype=cp.float32).ravel(),
         cp.asarray(img, dtype=cp.float32).ravel(),
         cp.asarray(img_origin, dtype=cp.float32),
         cp.asarray(voxsize, dtype=cp.float32), img_fwd.ravel(), nLORs,
         cp.asarray(img.shape, dtype=cp.int32)))
    cp.cuda.Device().synchronize()

    return xp.asarray(img_fwd, device=array_api_compat.device(img))

def joseph3d_back(xstart: npt.ArrayLike,
                  xend: npt.ArrayLike,
                  img_shape: tuple[int, int, int],
                  img_origin: npt.ArrayLike,
                  voxsize: npt.ArrayLike,
                  img_fwd: npt.ArrayLike,
                  threadsperblock: int = 32) -> npt.ArrayLike:
    """Non-TOF Joseph 3D back projector

    Parameters
    ----------
    xstart : npt.ArrayLike (numpy/cupy array or torch tensor)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : npt.ArrayLike (numpy/cupy array or torch tensor)
        end world coordinates of the LORs, shape (nLORs, 3)
    img_shape : tuple[int, int, int]
        the shape of the back projected image
    img_origin : npt.ArrayLike (numpy/cupy array or torch tensor)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : npt.ArrayLike (numpy/cupy array or torch tensor)
        array containing the voxel size
    img_fwd : npt.ArrayLike (numpy/cupy array or torch tensor)
        array of length nLORs containing the values to be back projected
    threadsperblock : int, optional
        by default 32
    """
    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img_fwd)

    # back projection of cupy or torch GPU array using the cupy raw kernel
    back_img = cp.zeros(img_shape, dtype=cp.float32)

    _joseph3d_back_cuda_kernel(
        (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
        (cp.asarray(xstart, dtype=cp.float32).ravel(),
         cp.asarray(xend, dtype=cp.float32).ravel(), back_img.ravel(),
         cp.asarray(img_origin, dtype=cp.float32),
         cp.asarray(voxsize, dtype=cp.float32),
         cp.asarray(img_fwd, dtype=cp.float32).ravel(), nLORs,
         cp.asarray(back_img.shape, dtype=cp.int32)))
    cp.cuda.Device().synchronize()

    return xp.asarray(back_img, device=array_api_compat.device(img_fwd))

class LinearOperator(abc.ABC):
    """abstract base class for linear operators"""

    def __init__(self, xp: ModuleType):
        """init method

        Parameters
        ----------
        xp : ModuleType
            numpy or cupy module
        """
        self._scale = 1
        self._xp = xp

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        raise NotImplementedError

    @property
    def scale(self) -> int | float | complex:
        """scalar factor applied to the linear operator"""
        return self._scale

    @scale.setter
    def scale(self, value: int | float | complex):
        if not np.isscalar(value):
            raise ValueError
        self._scale = value

    @property
    def xp(self) -> ModuleType:
        """module type (numpy or cupy) of the operator"""
        return self._xp

    @abc.abstractmethod
    def _apply(self, x):
        """forward step y = Ax"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        raise NotImplementedError

    def apply(self, x):
        """forward step y = scale * Ax

        Parameters
        ----------
        x : numpy or cupy array

        Returns
        -------
        numpy or cupy array
        """
        if self._scale == 1:
            return self._apply(x)
        else:
            return self._scale * self._apply(x)

    def __call__(self, x):
        return self.apply(x)

    def adjoint(self, y):
        """adjoint step x = conj(scale) * A^H y

        Parameters
        ----------
        y : numpy or cupy array

        Returns
        -------
        numpy or cupy array
        """
        if self._scale == 1:
            return self._adjoint(y)
        else:
            return np.conj(self._scale) * self._adjoint(y)

    def adjointness_test(self, verbose=False, iscomplex=False, **kwargs):
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        """

        x = self.xp.asarray(np.random.rand(*self.in_shape))
        y = self.xp.asarray(np.random.rand(*self.out_shape))

        if iscomplex:
            x = x + 1j * self.xp.random.rand(*self.in_shape)

        if iscomplex:
            y = y + 1j * self.xp.random.rand(*self.out_shape)

        x_fwd = self.apply(x)
        y_adj = self.adjoint(y)

        if iscomplex:
            ip1 = complex(self.xp.sum(self.xp.conj(x_fwd) * y))
            ip2 = complex(self.xp.sum(self.xp.conj(x) * y_adj))
        else:
            ip1 = float(self.xp.sum(x_fwd * y))
            ip2 = float(self.xp.sum(x * y_adj))

        if verbose:
            print(ip1, ip2)

        assert (np.isclose(ip1, ip2, **kwargs))

    def norm(self, num_iter=30, iscomplex=False, verbose=False) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of power iterations
        iscomplex : bool, optional
            use complex arrays
        verbose : bool, optional
            verbose output

        Returns
        -------
        float
            the norm of the linear operator
        """
        x = self.xp.random.rand(*self.in_shape)

        if iscomplex:
            x = x + 1j * self.xp.random.rand(*self.in_shape)

        for i in range(num_iter):
            x = self.adjoint(self.apply(x))
            norm_squared = self.xp.linalg.norm(x)
            x /= norm_squared

            if verbose:
                print(f'{(i+1):03} {self.xp.sqrt(norm_squared):.2E}')

        return float(self.xp.sqrt(norm_squared))


class ParallelViewProjector3D(LinearOperator):
    """3D non-TOF parallel view projector"""

    def __init__(self,
                 image_shape: tuple[int, int, int],
                 radial_positions: npt.ArrayLike,
                 view_angles: npt.ArrayLike,
                 radius: float,
                 image_origin: tuple[float, float, float],
                 voxel_size: tuple[float, float],
                 ring_positions: npt.ArrayLike,
                 span: int = 1,
                 max_ring_diff: int | None = None):
        """init method

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            shape of the input image (n0, n1, n2) (last direction is axial)
        radial_positions : npt.ArrayLike (numpy, cupy or torch array)
            radial positions of the projection views in world coordinates
        view angles : np.ArrayLike (numpy, cupy or torch array)
            angles of the projection views in radians
        radius : float
            radius of the scanner
        image_origin : tuple[float, float, float]
            world coordinates of the [0,0,0] voxel
        voxel_size : tuple[float, float, float]
            the voxel size in all directions (last direction is axial)
        ring_positions : numpy or cupy array
            position of the rings in world coordinates
        span : int
            span of the sinogram - default is 1
        max_ring_diff : int | None
            maximum ring difference - default is None (no limit)
        """

        super().__init__(array_api_compat.get_namespace(radial_positions))

        self._radial_positions = radial_positions
        self._device = array_api_compat.device(radial_positions)

        self._image_shape = image_shape
        self._image_origin = array_api_compat.to_device(
            self.xp.asarray(image_origin, dtype=self.xp.float32), self._device)
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray(voxel_size, dtype=self.xp.float32), self._device)

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        xstart2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2),
                          dtype=self.xp.float32), self._device)
        xend2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2),
                          dtype=self.xp.float32), self._device)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            xstart2d[:, i, 0] = self._xp.cos(
                phi) * self._radial_positions + self._xp.sin(
                    phi) * self._radius
            xstart2d[:, i, 1] = -self._xp.sin(
                phi) * self._radial_positions + self._xp.cos(
                    phi) * self._radius
            # world coordinates of LOR endpoints
            xend2d[:, i, 0] = self._xp.cos(
                phi) * self._radial_positions - self._xp.sin(
                    phi) * self._radius
            xend2d[:, i, 1] = -self._xp.sin(
                phi) * self._radial_positions - self._xp.cos(
                    phi) * self._radius

        self._ring_positions = ring_positions
        self._num_rings = ring_positions.shape[0]
        self._span = span

        if max_ring_diff is None:
            self._max_ring_diff = self._num_rings - 1
        else:
            self._max_ring_diff = max_ring_diff

        if self._span == 1:
            self._num_segments = 2 * self._max_ring_diff + 1
            self._segment_numbers = np.zeros(self._num_segments,
                                             dtype=np.int32)
            self._segment_numbers[0::2] = np.arange(self._max_ring_diff + 1)
            self._segment_numbers[1::2] = -np.arange(1,
                                                     self._max_ring_diff + 1)

            self._num_planes_per_segment = self._num_rings - np.abs(
                self._segment_numbers)

            self._start_plane_number = []
            self._end_plane_number = []

            for i, seg_number in enumerate(self._segment_numbers):
                tmp = np.arange(self._num_planes_per_segment[i])

                if seg_number < 0:
                    tmp -= seg_number

                self._start_plane_number.append(tmp)
                self._end_plane_number.append(tmp + seg_number)

            self._start_plane_number = np.concatenate(self._start_plane_number)
            self._end_plane_number = np.concatenate(self._end_plane_number)
            self._num_planes = self._start_plane_number.shape[0]
        else:
            raise ValueError('span > 1 not implemented yet')

        self._xstart = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32), self._device)
        self._xend = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32), self._device)

        for i in range(self._num_planes):
            self._xstart[:, :, i, :2] = xstart2d
            self._xend[:, :, i, :2] = xend2d

            self._xstart[:, :, i,
                         2] = self._ring_positions[self._start_plane_number[i]]
            self._xend[:, :, i,
                       2] = self._ring_positions[self._end_plane_number[i]]

    @property
    def in_shape(self):
        return self._image_shape

    @property
    def out_shape(self):
        return (self._num_rad, self._num_views, self._num_planes)

    @property
    def voxel_size(self) -> npt.ArrayLike:
        return self._voxel_size

    @property
    def image_origin(self) -> npt.ArrayLike:
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def xstart(self) -> npt.ArrayLike:
        return self._xstart

    @property
    def xend(self) -> npt.ArrayLike:
        return self._xend

    def _apply(self, x: npt.ArrayLike) -> npt.ArrayLike:
        y = joseph3d_fwd(self._xstart, self._xend, x,
                                      self.image_origin, self.voxel_size)
        return y

    def _adjoint(self, y: npt.ArrayLike) -> npt.ArrayLike:
        x = joseph3d_back(self._xstart, self._xend,
                                       self.image_shape, self.image_origin,
                                       self.voxel_size, y)
        return x
