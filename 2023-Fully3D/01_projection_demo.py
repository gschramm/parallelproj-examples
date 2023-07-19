# # Minimal parallelproj demo using non-TOF projections

# +
import parallelproj
# import the numpy or cupy as array backend
if parallelproj.cupy_enabled:
    import cupy as xp
else:
    import numpy as xp

import matplotlib.pyplot as plt


# +
# setup an dummy image + image geometry
n0 = 128
n1 = 128
n2 = 25
img = xp.zeros((n0,n1,n2), dtype = xp.float32)
img[(n0//4):((3*n0//4)), (n1//4):((3*n1//4)), :] = 1
img[70:75,70:75, :] = 4

voxel_size = xp.array([3., 3., 2.]).astype(xp.float32)
img_origin = ((-xp.array(img.shape) / 2 + 0.5) * voxel_size).astype(xp.float32)
# -

# load pre-defined LOR start / end points from file
lor_coords = xp.load('lor_coordinates.npz')
xstart = lor_coords['xstart']
xend = lor_coords['xend']

# +
# non-TOF forward projection
img_fwd = xp.zeros(xstart.shape[:-1], dtype = xp.float32)

parallelproj.joseph3d_fwd(xstart.reshape(-1, 3),
                xend.reshape(-1, 3),
                img,
                img_origin, voxel_size, img_fwd)
print(type(img_fwd))
print(img_fwd.shape)
#-


# +
# non-TOF backprojection
img_back = xp.zeros(img.shape, dtype = xp.float32)

parallelproj.joseph3d_back(xstart.reshape(-1, 3),
                           xend.reshape(-1, 3), img_back,
                           img_origin, voxel_size,
                           img_fwd)
print(type(img_back))
print(img_back.shape)
#-
# -

# visualize the results
fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(parallelproj.tonumpy(img[...,0], xp), cmap = 'Greys')
ax[1].imshow(parallelproj.tonumpy(img_fwd[...,0], xp), cmap = 'Greys')
ax[2].imshow(parallelproj.tonumpy(img_back[...,0], xp), cmap = 'Greys')
fig.tight_layout()
plt.show()
