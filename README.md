# Vehicle-Positioning-and-LoS-NLoS-Classification-Using-CSI-and-Neural-Networks
Vehicle positioning and LoS/NLoS classification using wireless Channel State Information (CSI). Compares machine learning and neural networks through preprocessing, accuracy, positioning error, robustness, and generalization to unseen locations.
# Vehicle Positioning and LoS/NLoS Classification Using Machine Learning and Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-f7931e)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

**Author:** Ashutosh Kumar &nbsp;·&nbsp; **Course:** ECE 553/653 Neural Networks — Project 1: Vehicle Positioning

This project uses measured **channel state information (CSI)** from an underground parking lot to solve two tasks:

1. **LoS/NLoS classification**: is the vehicle in line-of-sight (LoS) or non-line-of-sight (NLoS) of the radio unit?
2. **Vehicle position estimation**: where is the vehicle, as (x, y) coordinates in metres?

Each task is solved with **5 machine-learning (ML) models** and **8 neural-network (ANN) architectures**, with
pre-processing, a configuration study, an efficiency evaluation, a comparison of ML and ANN, robustness tests on an
emulated unreliable network and a test on positions never seen in training.

---

## Contents

- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [How to run, step by step](#how-to-run-step-by-step)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Dataset

| Property | Value |
|---|---|
| Positions | 476 vehicle positions (LoS or NLoS) |
| Sample | complex CSI matrix of 4 antennas × 1632 subcarriers |
| Label | `(x, y, LoS)`: coordinates in metres relative to the radio unit, and `0` = LoS / `1` = NLoS |
| Files | `train_dl.pt` (15,000 samples), `valid_dl.pt` (5,000 samples) |

The dataset is provided with the course and is **not included in this repository**. The two files are about 1 GB in
total, which is over GitHub's 100 MB file limit. See [Step 4](#step-4--put-the-dataset-in-place).

> Both files contain **all 476 positions**. The official validation score therefore measures how well a model
> recognizes *known* positions. A separate split by position (380 seen / 96 unseen positions) measures
> generalization.

---

## Methods

**Pre-processing.** All statistics come from the training data only, and a verification cell checks this.

- **Amplitude image** |H|: 4 neighbouring subcarriers averaged (1632 → 408), min-max scaled to [0, 1] → 4 × 408.
- **Channel impulse response (CIR)**: |IFFT(H)|, first 128 taps → 4 × 128.
- **46 physical features**: received power, Rician K-factor, RMS delay spread, kurtosis, first-path delay and power,
  inter-antenna correlations, and more.

| Family | Models |
|---|---|
| ML | Logistic / Ridge regression, SVM / SVR (RBF), KNN, Random Forest, Gradient Boosting |
| ANN | MLP, CNN-1D, CNN-2D, ResNet-1D, CNN-BiGRU, Transformer, CIR-CNN, Fusion CNN (+ tuned and multi-task networks) |

**Training of the networks.** Adam, learning rate 1e-3 with cosine decay, batch 64, and **early stopping** on the
validation set. The best validation checkpoint is restored and saved.

| Notebook | Networks | Budget |
|---|---|---|
| LoS/NLoS | all networks and the configuration study | at most 25 epochs, patience 5 |
| Positioning | architectures, multi-task, augmented, unseen-position networks | at most 100 epochs, patience 15 |
| Positioning | tuned network | at most 200 epochs, patience 20 |
| Positioning | configuration study | at most 50 epochs, patience 15 |

**Configuration study** (one choice changed at a time): hidden activation, output layer, loss, optimizer, learning
rate, batch size, epoch budget and input scaling. The ML models are tuned by grid search.

---

## Results

Numbers from the final runs of the notebooks, on an RTX 5060 laptop GPU.

### LoS/NLoS classification

| Model | Accuracy, known positions | NLoS F1 | Accuracy, unseen positions |
|---|---|---|---|
| KNN | 100.00 % | 1.000 | 76.5 % |
| Random Forest | 99.86 % | 0.996 | **84.9 %** |
| CNN-1D | 100.00 % | 1.000 | 72.1 % |
| ResNet-1D | 100.00 % | 1.000 | 81.8 % |
| CIR-CNN | 99.80 % | 0.995 | 82.9 % |

Always predicting "LoS" gives 81.3 % on the validation set and 78.8 % on the unseen positions.
**Robustness:** a CNN-1D trained with noise and packet-loss augmentation keeps **99.0 %** accuracy at 0 dB SNR and
**98.4 %** with 50 % of the CSI packets lost. Without augmentation it falls to 82.0 % and 57.6 %.

### Position estimation (mean error in metres)

| Model | Known positions | Unseen positions |
|---|---|---|
| KNN (fingerprinting) | **0.004** | 11.04 |
| Random Forest | 0.469 | 9.76 |
| ResNet-1D | 0.330 | **5.46** |
| CNN-BiGRU | 0.324 | – |
| Tuned CNN-BiGRU (200 epochs, best epoch 189) | **0.232** (best network) | – |
| Linear regression | 3.324 | 7.63 |

**Key findings**

- On known positions KNN is almost exact (0.004 m), because every validation position is also in the training set.
- The best regression network is the tuned CNN-BiGRU: 0.232 m.
- On unseen positions all models degrade strongly. The best mean error is 5.46 m, against a 1.18 m spacing between
  measured positions. The CSI fingerprint is highly position-specific.
- The full report, with all tables and figures, is `Final_Report_Vehicle_Positioning.pdf`.

---

## Repository structure

```
.
├── LoS_NLoS_classification.ipynb        # Program 1: LoS/NLoS classification (ML + ANN)
├── Position estimation.ipynb            # Program 2: vehicle position estimation (ML + ANN)
├── Final_Report_Vehicle_Positioning.pdf # Final report (IEEE format, 6 pages)
├── LoS_best_model.pt                    # best LoS/NLoS checkpoint (saved by the notebook)
├── Position_best_model.pt               # best positioning checkpoint (saved by the notebook)
├── requirements.txt                     # Python packages and versions
├── README.md
└── Dataset/                             # not in the repository: add it yourself (Step 4)
    ├── train_dl.pt
    └── valid_dl.pt
```

---

## Requirements

From `requirements.txt` (verified with Python 3.10):

| Package | Version | Package | Version |
|---|---|---|---|
| torch | 2.14.0 | matplotlib | 3.10.9 |
| numpy | 2.2.6 | seaborn | 0.13.2 |
| scipy | 1.15.3 | pandas | 2.3.3 |
| scikit-learn | 1.7.2 | notebook | ≥ 7.0 |
| ipykernel | ≥ 6.0 | nbconvert | ≥ 7.0 |

An NVIDIA GPU is optional. The notebooks run on the CPU too, but much more slowly.

---

## How to run, step by step

### Step 1 — Install conda

**Windows:** download and install Anaconda from <https://www.anaconda.com/download> with the default settings. Open
**Anaconda Prompt** and check the installation:

```bash
conda --version
```

**macOS / Linux:** install Miniconda (choose the installer for your system):

```bash
# Linux 64-bit
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
# macOS Apple Silicon: Miniconda3-latest-MacOSX-arm64.sh  |  macOS Intel: Miniconda3-latest-MacOSX-x86_64.sh

source ~/miniconda3/bin/activate
conda init --all
```

### Step 2 — Create and activate the environment

```bash
conda create -y -n positioning python=3.10
conda activate positioning
```

### Step 3 — Get the code and install the packages

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
pip install -r requirements.txt
```

**NVIDIA GPU (optional):** install the PyTorch build that matches your CUDA version, using the command from
<https://pytorch.org/get-started/locally/>. For example:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

Check the installation. It prints the versions, and `True` if the GPU can be used:

```bash
python -c "import torch, sklearn; print(torch.__version__, torch.cuda.is_available(), sklearn.__version__)"
```

### Step 4 — Put the dataset in place

Create a `Dataset` folder next to the notebooks and copy the two course files into it:

```
Dataset/
├── train_dl.pt     # 15,000 samples
└── valid_dl.pt     #  5,000 samples
```

### Step 5 — Run both programs

**From the terminal.** Each command runs a whole notebook from the first to the last cell and saves all results and
figures into it.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "LoS_NLoS_classification.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "Position estimation.ipynb"
```

**Interactively in the browser:**

```bash
jupyter notebook
```

Then open a notebook and choose **Kernel → Restart & Run All**. Always run from the first cell, because later cells
use variables created by earlier ones.

| Notebook | Run time of the final run (RTX 5060 laptop GPU) |
|---|---|
| `LoS_NLoS_classification.ipynb` | about 30 minutes |
| `Position estimation.ipynb` | about 82 minutes |

Keep the computer **plugged in and awake** during the run. Sleep pauses the training and distorts the measured
training times.

### Step 6 — Check the results

- The verification cell prints `... training samples only: OK`, which confirms there is no data leakage.
- The last cells show the final result table and a JSON summary of all key numbers.
- To view a notebook as a web page, export it to HTML:

  ```bash
  jupyter nbconvert --to html "Position estimation.ipynb"
  ```

**Quick start (environment already set up):**

```bash
conda activate positioning
cd <your-repository>
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "LoS_NLoS_classification.ipynb"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "Position estimation.ipynb"
```

---

## Outputs

| Output | Where |
|---|---|
| All results, tables and figures | inside the executed notebooks |
| Best LoS/NLoS network | `LoS_best_model.pt` |
| Best positioning network | `Position_best_model.pt` |
| Key numbers (JSON) | printed by the "Summary of the key results" cell of each notebook |

Each checkpoint stores the architecture name, the weights, the best epoch and the scaling statistics of the
training set. The positioning checkpoint also stores the target mean and standard deviation.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `conda` or `jupyter` not found | Open Anaconda Prompt and run `conda activate positioning` |
| `FileNotFoundError: Dataset/...` | Run from the project folder and check the Step 4 layout |
| `NameError` | A cell ran out of order; use **Restart & Run All** |
| Figures missing after `nbconvert` | Do not set `MPLBACKEND=Agg`; leave it unset |
| `CUDA out of memory` | Close other GPU programs, or run on the CPU |
| `torch.load` / `weights_only` error | Already handled with `weights_only=False`; keep the PyTorch version from `requirements.txt` |

---

## Limitations

- The official validation set shares all positions with the training set, and it is also used for model selection
  and early stopping, so its scores are optimistic. The unseen-position test is a single split with one random seed.
- The ANN positioning results reflect a limited training budget: most networks reach their best checkpoint at the
  end of the schedule.
- Training times compare a CPU (ML models) with a laptop GPU (networks) and vary between runs.
- Noise and packet loss are simulated (white noise, independent packet losses), not measured on a real network.

---

## Acknowledgements

Dataset and project description: ECE 553/653 Neural Networks, Project 1: Vehicle Positioning.
Built with [PyTorch](https://pytorch.org) and [scikit-learn](https://scikit-learn.org).
