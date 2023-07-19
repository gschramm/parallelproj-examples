from pathlib import Path
import jupytext

notebook_path = Path('01_projection_demo.py')

with open('00_colab_setup.py', 'r') as f:
    header_str = f.read()

with open(notebook_path, 'r') as f:
    notebook_str = f.read()

# join colab setup cells and notebook cells
notebook = jupytext.reads(header_str + '\n' + notebook_str)

# write colab notebook to file
jupytext.write(notebook,
               notebook_path.parent / f'{notebook_path.stem}_colab.ipynb')
