from pathlib import Path
import jupytext

repo_path = 'https://github.com/gschramm/parallelproj-examples/blob/main/2023-MIC/'

notebook_path = Path('01_poisson_projection_varnet.py')

with open('00_colab_setup.py', 'r') as f:
    header_str = f.read()

with open(notebook_path, 'r') as f:
    notebook_str = f.read()

# replace local image path with raw github link, such that
# images are loaded in colab
notebook_str = notebook_str.replace(
    '![](figs/',
    '![](https://github.com/gschramm/parallelproj-examples/blob/main/2023-MIC/figs/'
)
notebook_str = notebook_str.replace('.png)', '.png?raw=True)')

# join colab setup cells and notebook cells
notebook = jupytext.reads(header_str + '\n' + notebook_str)

# write colab notebook to file
jupytext.write(notebook,
               notebook_path.parent / f'{notebook_path.stem}_colab.ipynb')
