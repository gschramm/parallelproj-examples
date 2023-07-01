# +
# check if we are running in google colab
# if we are running in colab, we install conda/mamba and all dependencies we need
import sys
import os
import subprocess
from distutils.spawn import find_executable

# test if we run in google colab
in_colab = 'google.colab' in sys.modules
print(f'running in colab: {in_colab}')

# test if cuda is present
cuda_present = find_executable('nvidia-smi') is not None
print(f'cuda present: {cuda_present}')

if in_colab:
    # install condacolab to get the conda/mamba command which we need to install parallelproj
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'condacolab'])
    import condacolab
    condacolab.install()

    #-----------------------------------------------------------------------------
    #--- !!! the kernel gets restarted here !!! ----------------------------------
    #-----------------------------------------------------------------------------

    # if we we are running in colab, we install parallelproj from conda-forge
    # and set the environment variable for the parallelproj c/cuda library
    # we need to redo the check for COLAB, because the install
    # of conda on colab, restarts the kernel and deletes all variables 
   

    # install all dependencies
    subprocess.run(['mamba', 'install', 'parallelproj', 'scipy', 'matplotlib', 'nibabel', 'ipykernel', 'pytorch', 'torchmetrics'])
    
    # install cupy if cuda is present
    if cuda_present:
        subprocess.run(['mamba', 'install', 'cupy'])

    subprocess.run(['mamba', 'list'])
    subprocess.run(['git', 'clone', 'https://github.com/gschramm/parallelproj-examples.git'])

# +
import sys
import os
from distutils.spawn import find_executable

# test if we run in google colab
in_colab = 'google.colab' in sys.modules
print(f'running in colab: {in_colab}')

# test if cuda is present
cuda_present = find_executable('nvidia-smi') is not None
print(f'cuda present: {cuda_present}')

if in colab:
    os.environ['PARALLELPROJ_C_LIB']='/usr/local/lib/libparallelproj_c.so'
    if cuda_present:
        os.environ['PARALLELPROJ_CUDA_LIB']='/usr/local/lib/libparallelproj_cuda.so'
# -

# %cd parallelproj-examples/2023-MIC
