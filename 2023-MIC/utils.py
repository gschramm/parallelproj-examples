from __future__ import annotations

import numpy as np
import parallelproj

import requests
import zipfile
import io
from pathlib import Path


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


class ParallelViewProjector3D(parallelproj.LinearOperator):
    """2D non-TOF parallel view projector"""

    def __init__(self,
                 image_shape,
                 radial_positions,
                 view_angles,
                 ring_positions,
                 radius,
                 image_origin,
                 voxel_size,
                 xp,
                 span=1,
                 max_ring_diff=None):
        """init method

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            shape of the input image (n0, n1, n2)
        radial_positions : numpy or cupy array
            radial positions of the projection views in world coordinates
        view angles : numpy or cupy array
            angles of the projection views in radians
        radial_positions : numpy or cupy array
            position of the rings in world coordinates
        radius : float
            radius of the scanner
        image_origin : 3 element numpy or cupy array
            world coordinates of the [0,0,0] voxel
        voxel_size : 3 element numpy or cupy array
            the voxel size
        xp : ModuleType
            numpy or cupy module
        span : int
            span of the sinogram - default is 1
        max_ring_diff : int | None
            maximum ring difference - default is None (no limit)
        """
        super().__init__(xp)

        self._image_shape = image_shape
        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]
        self._radial_positions = radial_positions
        self._ring_positions = ring_positions
        self._num_rings = ring_positions.shape[0]
        self._num_rad = radial_positions.shape[0]
        self._span = span

        if max_ring_diff is None:
            self._max_ring_diff = self._num_rings - 1
        else:
            self._max_ring_diff = max_ring_diff

        if self._span == 1:
            #self._num_planes = self._num_rings**2
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

        self._radius = radius
        self._image_origin = image_origin
        self._voxel_size = voxel_size

        self._xp = xp

        self._xstart2d = self._xp.zeros((self._num_views, self._num_rad, 2),
                                        dtype=xp.float32)
        self._xend2d = self._xp.zeros((self._num_views, self._num_rad, 2),
                                      dtype=xp.float32)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart2d[
                i, :,
                0] = self._xp.cos(phi) * self._radial_positions + self._xp.sin(
                    phi) * self._radius
            self._xstart2d[i, :, 1] = -self._xp.sin(
                phi) * self._radial_positions + self._xp.cos(
                    phi) * self._radius
            # world coordinates of LOR endpoints
            self._xend2d[
                i, :,
                0] = self._xp.cos(phi) * self._radial_positions - self._xp.sin(
                    phi) * self._radius
            self._xend2d[i, :, 1] = -self._xp.sin(
                phi) * self._radial_positions - self._xp.cos(
                    phi) * self._radius

        self._xstart = self._xp.zeros(
            (self._num_views, self._num_rad, self._num_planes, 3),
            dtype=self._xp.float32)
        self._xend = self._xp.zeros(
            (self._num_views, self._num_rad, self._num_planes, 3),
            dtype=self._xp.float32)

        for i in range(self._num_planes):
            self._xstart[:, :, i, :2] = self._xstart2d
            self._xend[:, :, i, :2] = self._xend2d

            self._xstart[:, :, i,
                         2] = self._ring_positions[self._start_plane_number[i]]
            self._xend[:, :, i,
                       2] = self._ring_positions[self._end_plane_number[i]]

    @property
    def in_shape(self):
        return self._image_shape

    @property
    def out_shape(self):
        return (self._num_views, self._num_rad, self._num_planes)

    def _apply(self, x):
        y = self._xp.zeros(self.out_shape, dtype=self._xp.float32)
        parallelproj.joseph3d_fwd(self._xstart.reshape(-1, 3),
                                  self._xend.reshape(-1, 3),
                                  x.astype(self._xp.float32),
                                  self._image_origin, self._voxel_size, y)
        return y

    def _adjoint(self, y):
        x = self._xp.zeros(self.in_shape, dtype=self._xp.float32)
        parallelproj.joseph3d_back(self._xstart.reshape(-1, 3),
                                   self._xend.reshape(-1, 3), x,
                                   self._image_origin, self._voxel_size,
                                   y.astype(self._xp.float32))
        return x