# LeWM Control Analysis — Project Setup Guide

> Windows 11 + WSL2 (Ubuntu 26.04) + RTX 4060 · One-time setup

---

## Prerequisites

- Windows 11 with WSL2 enabled
- Ubuntu 26.04 installed in WSL2
- NVIDIA RTX 4060 (8GB VRAM)
- 16GB+ system RAM

---

## 1. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y

# CUDA — ships natively in Ubuntu 26.04
sudo apt install nvidia-cuda-toolkit -y

# Compression tool for dataset
sudo apt install zstd -y

# Verify GPU
nvidia-smi
```

Expected output from `nvidia-smi`:
```
NVIDIA GeForce RTX 4060 | CUDA Version: 12.x
```

---

## 2. Project Directory Structure

```bash
mkdir -p ~/lewm-project/{checkpoints,datasets,notebooks,results}
export STABLEWM_HOME=~/lewm-project
echo 'export STABLEWM_HOME=~/lewm-project' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. Python Environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create project virtualenv
uv venv ~/lewm-project/.venv --python=3.11
source ~/lewm-project/.venv/bin/activate

# Fix pip to use venv pip, not system pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

> **Every new terminal session:** run `source ~/lewm-project/.venv/bin/activate` before anything else.

---

## 4. Python Packages

```bash
# Core packages
python -m pip install huggingface_hub hdf5plugin
python -m pip install "stable-worldmodel[train,env]"
python -m pip install stable-pretraining
python -m pip install scikit-learn matplotlib numpy jupyterlab

# PyTorch with CUDA 12
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: pin transformers to match checkpoint architecture
python -m pip install transformers==5.0.0
```

> **Why transformers==5.0.0?** The LeWM checkpoint was built with this version. Later versions rename ViT internal layers, causing `load_state_dict` to fail with key mismatches.

Verify GPU is accessible:
```python
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

Expected:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060
VRAM: 8.0 GB
```

---

## 5. Clone Le-WM Repository

```bash
cd ~/lewm-project
git clone https://github.com/lucas-maes/le-wm.git

# le-wm has no setup.py — add to PYTHONPATH instead
echo 'export PYTHONPATH="$PYTHONPATH:/home/$USER/lewm-project/le-wm"' >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Download Checkpoint and Dataset

These are one-time downloads. Once done, never needed again.

```bash
cd ~/lewm-project

# Pretrained LeWM checkpoint
hf download quentinll/lewm-pusht \
    --local-dir ~/lewm-project/hf_pusht

# Push-T expert dataset (13GB compressed)
hf download quentinll/lewm-pusht \
    --repo-type dataset \
    --local-dir ~/lewm-project/pusht-data

# Decompress dataset (~40-65GB decompressed)
zstd -d ~/lewm-project/pusht-data/pusht_expert_train.h5.zst \
    -o ~/lewm-project/datasets/pusht_expert_train.h5
```

---

## 7. Compute Dataset Statistics (One-Time)

Save as `~/lewm-project/stats_calc.py` and run once:

```python
import os
import numpy as np
import json
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

os.environ["STABLEWM_HOME"] = os.path.expanduser("~/lewm-project")

dataset = HDF5Dataset(
    "pusht_expert_train",
    keys_to_cache=["action", "proprio"],   # do NOT include pixels — 328GB
    cache_dir=os.path.expanduser("~/lewm-project"),
)

stats = {}
for col in ["action", "proprio"]:
    print(f"Computing stats for {col}...")
    col_data = dataset.get_col_data(col)
    col_data = col_data[~np.isnan(col_data).any(axis=1)]
    stats[col] = {
        "mean": col_data.mean(axis=0).tolist(),
        "std":  col_data.std(axis=0).tolist()
    }

with open(os.path.expanduser("~/lewm-project/pusht_stats.json"), "w") as f:
    json.dump(stats, f)

print("Done. Stats saved to pusht_stats.json")
print(json.dumps(stats, indent=2))
```

```bash
python3 ~/lewm-project/stats_calc.py
```

> **Why not include pixels?** The pixels column is 328GB uncompressed — loading it crashes any machine. Only `action` and `proprio` are needed for normalisation.

Expected output:
```json
{
  "action": {"mean": [-0.008, 0.007], "std": [0.208, 0.206]},
  "proprio": {"mean": [228.8, 292.2, -2.9, 2.5], "std": [103.3, 98.9, 77.4, 76.8]}
}
```

---

## 8. Convert Checkpoint (One-Time)

Save as `~/lewm-project/checkpoint_converter.py` and run once:

```python
import os, sys, json, torch

os.environ["STABLEWM_HOME"] = os.path.expanduser("~/lewm-project")
sys.path.insert(0, os.path.expanduser("~/lewm-project/le-wm"))

import stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP

src      = Path(os.path.expanduser("~/lewm-project/hf_pusht"))
out_ckpt = Path(os.path.expanduser("~/lewm-project/checkpoints/pusht/lewm_object.ckpt"))
out_pt   = Path(os.path.expanduser("~/lewm-project/checkpoints/pusht/lewm_object.pt"))
out_ckpt.parent.mkdir(parents=True, exist_ok=True)

cfg = json.loads((src / "config.json").read_text())

def clean_cfg(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False,
    use_mask_token=False,
)
mlp = lambda k: MLP(
    input_dim=cfg[k]["input_dim"],
    output_dim=cfg[k]["output_dim"],
    hidden_dim=cfg[k]["hidden_dim"],
    norm_fn=torch.nn.BatchNorm1d,
)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**clean_cfg(cfg["predictor"])),
    action_encoder=Embedder(**clean_cfg(cfg["action_encoder"])),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)

sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)

torch.save(model, out_ckpt)
torch.save(model, out_pt)

print("Checkpoint converted successfully.")
print("get_cost available:", hasattr(model, 'get_cost'))
```

```bash
python3 ~/lewm-project/checkpoint_converter.py
```

Verify the checkpoint is visible to `swm`:
```bash
swm checkpoints
# Expected: pusht | lewm_object
```

---

## 9. Launch JupyterLab

```bash
source ~/lewm-project/.venv/bin/activate
cd ~/lewm-project
jupyter lab --no-browser --port=8888
```

Open `http://localhost:8888` in your Windows browser.

Create notebooks under `~/lewm-project/notebooks/`.

---

## Project Directory After Full Setup

```
~/lewm-project/
├── .venv/                        # Python virtual environment
├── le-wm/                        # LeWM source repo (PYTHONPATH)
├── hf_pusht/                     # Raw HuggingFace checkpoint files
│   ├── weights.pt
│   └── config.json
├── checkpoints/
│   └── pusht/
│       ├── lewm_object.ckpt      # Converted model (AutoCostModel loads this)
│       └── lewm_object.pt        # Same — for swm checkpoints listing
├── datasets/
│   └── pusht_expert_train.h5     # Decompressed dataset
├── pusht-data/
│   └── pusht_expert_train.h5.zst # Compressed dataset (keep as backup)
├── pusht_stats.json              # Precomputed normalisation stats
├── stats_calc.py                 # One-time stats script
├── checkpoint_converter.py       # One-time conversion script
├── notebooks/                    # Research notebooks
└── results/                      # Plots and logged measurements
```

---

## Quick Reference — Every Session

```bash
# 1. Activate environment
source ~/lewm-project/.venv/bin/activate

# 2. Launch JupyterLab
cd ~/lewm-project && jupyter lab --no-browser --port=8888

# 3. Open http://localhost:8888 in Windows browser
```

All notebooks start with:
```python
import os, sys
os.environ["STABLEWM_HOME"] = os.path.expanduser("~/lewm-project")
sys.path.insert(0, os.path.expanduser("~/lewm-project/le-wm"))
```
