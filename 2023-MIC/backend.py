from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt
import cupy as cp
import array_api_compat

import requests
import zipfile
import io
from pathlib import Path

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

