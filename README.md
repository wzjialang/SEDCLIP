# SEDCLIP: Adapting Vision-Language Model for Multi-label Surgical Error Detection

This repository is released to make the Graph-attention Temporal Adaptor (GTA) independently checkable during review. It currently contains the relevant minimal implementation of the CLIP baseline + GTA module, with the central GTA equations (Eqs. 10 and 11) implemented in `temporal_module.py` (Lines 71-77).


## Setup and Quick Start

Follow the steps below to clone the code, restore the example data, and run the minimal training example.

### 1. Clone the repository

```bash
git clone https://github.com/wzjialang/SEDCLIP.git
cd SEDCLIP
```

### 2. Download the example data

Download the following file from the repository's GitHub Releases page and place it in the `SEDCLIP/` root directory:

- `SEDCLIP_example_data.tar.gz`: contains `example_data/`

### 3. Extract the example data

```bash
tar -xzf SEDCLIP_example_data.tar.gz -C .
```

After extraction, the expected repository layout is:

```text
SEDCLIP/
|-- clip/                  # Bundled CLIP implementation
|-- prompt/cls.txt         # Class names used for CLIP text prompts
|-- example_data/          # Small example dataset with one MLE-RARP video
|-- config.py              # Default paths and hyperparameters
|-- dataset.py             # Sliding-window frame dataset
|-- ours_model.py          # CLIP baseline + GTA
|-- temporal_module.py     # Graph-attention temporal adaptor (GTA)
|-- train_example.py       # Minimal training script
`-- requirements.txt       # Python dependencies
```

### 4. Install dependencies

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 5. Run the example

```bash
python train_example.py --device cuda:0 --max_epoch 10
```

On the first run, `clip.load("ViT-B/16")` downloads the official CLIP checkpoint and caches it under `~/.cache/clip` by default. To use a local checkpoint instead, set `self.clip_weights` in `config.py` to that file path.

## Acknowledge
We sincerely appreciate the authors for releasing the following valuable resources: [openai/CLIP](https://github.com/openai/CLIP), [nwpu-zxr/VadCLIP](https://github.com/nwpu-zxr/VadCLIP), [ctX-u/PLOVAD](https://github.com/ctX-u/PLOVAD), [diegoantognini/pyGAT](https://github.com/diegoantognini/pyGAT/tree/master).
