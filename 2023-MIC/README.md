## Local installation / setup

0. **To run these tutorials, you need a recent CUDA-capable GPU.**

1. Install the [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) python distribution. Mamba is a more efficient drop-in replacement for `conda` with default channel `conda-forge`.
   **In case you have already installed `conda` on your system, you can replace all `mamba` commands with `conda`**.

2. Clone the parallelproj examples repository: `git clone git@github.com:gschramm/parallelproj-examples.git`

3. `cd 2023-MIC`

4. Create a virtual environment containing all dependencies we need for this tutorial: `mamba env create -f environment.yaml`

## Running this tutorial

1. Activate the virtual mamba environment: `mamba activate 2023-MIC-DL-recon`
2. Run the tutorials (all `.py` files starting with two digits) either as normal python script or as jupyter notebook. The latter should be possible since `jupytext` should get installed as dependency. To run the tutorials as jupyter notebook, run `jupyter notebook` and select the tutorial file.

## Running on google colab

- TODO (explain how to install mamba + dependencies on colab)
