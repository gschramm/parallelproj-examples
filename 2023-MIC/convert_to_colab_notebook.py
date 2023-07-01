import jupytext
from pathlib import Path

notebook_path = Path('01_poisson_projection_varnet.py')

with open('00_colab_setup.py', 'r') as f:
    header_str = f.read()

with open(notebook_path, 'r') as f:
    notebook_str = f.read()

notebook = jupytext.reads(header_str + '\n' + notebook_str)

jupytext.write(notebook,
               notebook_path.parent / f'{notebook_path.stem}_colab.ipynb')
