# VAE Playground

This is a playground for experimenting with Variational Autoencoders (VAEs) in PyTorch.

## Installation

- Required: Python 3.10. (You can use `brew install python@3.10` / pyenv / etc.)
- Preferred: A machine with a GPU:
    - Best to use CUDA machine. (You can use Google Colab or similar if you don't have one. I personally like Lambda Labs)
    - M1 Macs also work

```bash
git clone htts://github.com/soof-golan/vae-playground.git
cd vae-playground
python3.10 -m venv venv
source venv/bin/activate
pip install -r jupyterlab
python -m jupyterlab

# In the JupyterLab window, open the notebook `vae-playground.ipynb`.

# When you're done, deactivate the virtual environment with `deactivate`.
```


License MIT (c) 2023 Soof Golan.
