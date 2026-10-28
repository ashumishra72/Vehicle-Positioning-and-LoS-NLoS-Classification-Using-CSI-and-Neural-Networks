# Overview

This tutorial provides an efficient guide to configuring your Python environment for this project (`Project 1`).

## Step 0, Install Conda

### For Windows

1. Open <https://www.anaconda.com/download> and download the installer for your operating system.
2. Run it and keep the default settings.
3. Open **Anaconda Prompt** from the Start menu, and confirm with `conda --version`.

### For Mac and Linux

Run the following in a terminal.

```bash
# macOS, Apple Silicon
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash ./Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
```

```bash
# macOS, Intel
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ./Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
```

```bash
# Linux, 64-bit Intel/AMD
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```
```bash
# Enable conda in this shell, then in every future shell
source ~/miniconda3/bin/activate
conda init --all
```


## Step 1, Install Python Requirements:

### Create New "positioning" Environment:
macOS/Linux/Windows (using Anaconda Prompt)
- `conda create -y -n positioning python=3.10`

### Activate the "positioning" Environment:
- `conda activate positioning`

### Install Dependencies:
- `cd PATH/TO/Project 1`
- `pip install -r requirements.txt`

The notebooks run on the GPU when one is available and fall back to the CPU otherwise (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`), so a CPU-only machine is enough, just slower. The versions in `requirements.txt` are the ones this project was verified with; on a machine with an NVIDIA GPU, install the PyTorch wheel that matches your CUDA version.

## Step 2, Run the Notebooks in Jupyter Notebook:

### Launch Jupyter Notebook in Your Browser
- After activating the `positioning` environment, run:
    - `jupyter notebook`
- Navigate to the `Project 1` folder and open the notebook you want to run.

### Complete and Run the Code
- To execute a single cell, press `Shift + Enter`.
- To run a whole notebook, click `Cell → Run All`.
- **Run each notebook from the first cell to the last.** Later cells reuse variables created by earlier cells (the loaded dataset, the normalized tensors, the trained `model`), so running a cell out of order raises `NameError`.

The three notebooks are independent of each other and can be run in any order; they only share the `Dataset` folder.

## Data Layout

```
Project_ThreeInOne/
|-- LoS_NLoS_classification_DL.ipynb
|-- Coordinate_Regression_DL.ipynb
|-- Coordinate_Regression_LR.ipynb
`-- Dataset/
    |-- train_dl.pt          # 15,000 samples
    `-- valid_dl.pt          # 5,000 samples
```

The campaign covers **476 UE positions** with roughly 42 measured CFR samples each, that is **20,000 samples**, split into **15,000 training** and **5,000 validation** samples. Both splits contain all 476 positions (18 to 45 samples per position in training, 4 to 19 in validation).

Each sample is a **4 x 1632 complex** CFR matrix, one row per antenna and one column per subcarrier. Every notebook takes the modulus with `torch.abs` and normalizes to [0, 1] using the training-set minimum and maximum, which turns the sample into a single-channel 4 x 1632 image for the convolution network.

The two `.pt` files store PyTorch `DataLoader` objects holding one `TensorDataset`. Its second element is a label tensor with **three columns per sample**:

- columns 0 and 1 — the UE coordinate `(x, y)` in metres, relative to the RRU;
- column 2 — the LoS indicator, `0` = LoS and `1` = NLoS.

Each task picks the columns it needs: the classification notebook uses column 2, the coordinate-regression notebook uses columns 0 and 1, and the linear-regression notebook uses columns 0 and 1 for the distance plus the mean power over the whole 4 x 1632 matrix, converted to dBm with `10*log10(P/1e-3)`.

## Notes on the Code as Shipped

Two library API changes break the notebooks as originally written. Both are already applied in the copy deployed on the lab server, listed in `threeinone_compat_fix.patch`:

1. **PyTorch >= 2.6** — `torch.load` now defaults to `weights_only=True` and refuses to unpickle the shipped `Dataset/*.pt` files, because they store `DataLoader` objects. The load cell of each of the three notebooks now passes `weights_only=False`.
2. **scikit-learn >= 1.6** — `accuracy_score` now returns a plain Python `float`, which has no `.item()`, so `train_metric_epoch.append((metric_sum/step).item())` raises `AttributeError`. The last cells of `LoS_NLoS_classification_DL.ipynb` now append `float(metric_sum/step)` and `float(val_metric_sum/val_step)`.

If you run a notebook on a machine without a display (for example with `jupyter nbconvert --to notebook --execute`), leave the `MPLBACKEND` variable alone. The IPython kernel selects matplotlib's inline backend, which embeds the figures in the executed notebook. Forcing `MPLBACKEND=Agg` makes the run finish but silently drops every figure.
