# # Forward and back projection tutorial

# +
from __future__ import annotations
import array_api_compat.torch as xp
dev = 'cuda'

print(f'running on {dev} device using {xp.__name__}')
# -

from array_api_compat import to_device
import matplotlib.pyplot as plt
from backend import ParallelViewProjector3D

# ### Image parameters

# shape of our 3D image
image_shape = (128, 128, 8)
# voxel size of our 3D image
voxel_size = (2., 2., 4.)
# world coordinates of the [0, 0] pixel
image_origin = (-127., -127., -14.)

# ### Projector parameters

# +
# radial positions of the projection lines
radial_positions = to_device(xp.linspace(-128, 128, 200), dev)
# projection angles
view_angles = to_device(xp.linspace(0, xp.pi, 180, endpoint=False), dev)
# distance between the image center and the start / end of the center line
radius = 200.
# axial coordinates of the projection "rings"
ring_positions = to_device(xp.linspace(-14, 14, 8), dev)

# create a 3D parallel view projector
proj3d = ParallelViewProjector3D(image_shape,
                                 radial_positions,
                                 view_angles,
                                 radius,
                                 image_origin,
                                 voxel_size,
                                 ring_positions,
                                 max_ring_diff=5)
# -

# ### Create a simple 3D test image

img = to_device(xp.zeros(proj3d.in_shape, dtype=xp.float32), dev)
img[32:96, 32:64, 1:-1] = 1.
img[48:64, 48:54, 3:-3] = 2.

# ### Forward and back project the test image

img_fwd = proj3d(img)
img_fwd_back = proj3d.adjoint(img_fwd)

# ### Show the image, the forward projection and the backprojection

# +
### note: we only show the "direct" planes of the forward projection

fig, ax = plt.subplots(3, 8, figsize=(16, 6))
for i in range(8):
    ax[0, i].imshow(to_device(img[..., i], 'cpu'), vmin=0, vmax=xp.max(img))
    ax[1, i].imshow(to_device(img_fwd[..., i], 'cpu'),
                    vmin=0,
                    vmax=xp.max(img_fwd))
    ax[2, i].imshow(to_device(img_fwd_back[..., i], 'cpu'),
                    vmin=0,
                    vmax=xp.max(img_fwd_back))
    ax[0, i].set_title(f'x slice {i}')
    ax[1, i].set_title(f'(A x) plane {i}')
    ax[2, i].set_title(f'(A^H A x) slice {i}')
for axx in ax.ravel():
    axx.set_axis_off()
fig.tight_layout()
plt.show()
