## Local installation / setup

1. Install the [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) python distribution. Mamba is a more efficient drop-in replacement for `conda` with default channel `conda-forge`.

2. Clone the parallelproj examples repository: `git clone git@github.com:gschramm/parallelproj-examples.git`

3. `cd 2023-MIC`

4. Create a virtual environment containing all dependencies we need for this tutorial: `mamba create env -f environment.yaml`

## Running this tutorial

1. Activate the virtual mamba environment: `mamba activate 2023-MIC-DL-recon`
2. Run the tutorials either as normal python script or as jupyer notebook. The latter should be possible since `jupytext` should get installed as dependency.

## Running on google colab

- TODO
