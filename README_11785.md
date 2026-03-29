# ESC-50 Quick Start
## 0. Environment Setup

### Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda/Miniconda installed
- CUDA-capable GPU with CUDA 11.7 drivers (for `torch==1.13.0+cu117`)
- Git

### Option A — conda environment file (recommended)

```bash
# Clone the repo first
git clone <repo-url>
cd 11-785_EfficientAT

# Create and activate the environment from environment.yml
conda env create -f environment.yml
conda activate idl_project
```

Then install the vendored PerforatedAI wheel (Python 3.10, Linux x86_64):

```bash
pip install PerforatedAI/releases/e3.1.0/perforatedai-3.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

> **Other platforms:** wheel files for macOS and Windows are also in `PerforatedAI/releases/e3.1.0/`.
> Pick the one matching your OS and Python version, e.g.:
> - macOS:  `perforatedai-3.1.0-cp310-cp310-macosx_10_9_universal2.whl`
> - Windows: `perforatedai-3.1.0-cp310-cp310-win_amd64.whl`

### Option B — manual conda + pip

```bash
# 1. Create a fresh Python 3.10 environment
conda create -n idl_project python=3.10 -y
conda activate idl_project

# 2. Install ffmpeg via conda (required by librosa / torchaudio)
conda install -c conda-forge ffmpeg -y

# 3. Install PyTorch 1.13 with CUDA 11.7
pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install vendored PerforatedAI wheel
pip install PerforatedAI/releases/e3.1.0/perforatedai-3.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```


### Verify the installation

```bash
python - <<'EOF'
import torch, librosa, perforatedai, perforatedbp
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("librosa:", librosa.__version__)
print("perforatedai OK | perforatedbp OK")
EOF
```

If you already have `idl_project` set up with all packages, just activate it:

```bash
conda activate idl_project
```

### Option C — Google Colab

Colab comes with Python 3.10 and a GPU runtime (T4, CUDA 11.x/12.x), so conda is not needed.
Run the following cells at the top of your notebook:

```python
# Cell 1: install system dependency
!apt-get install -y ffmpeg > /dev/null
```

```python
# Cell 2: clone the repo
import os
!git clone <repo-url> /content/11-785_EfficientAT
os.chdir("/content/11-785_EfficientAT")
```

```python
# Cell 3: check CUDA version, then install matching PyTorch 1.13
import subprocess
cuda = subprocess.check_output("nvcc --version", shell=True).decode()
print(cuda)
# Colab typically has CUDA 11.x → use cu117; if CUDA 12.x → use cu118
!pip install torch==1.13.0+cu117 torchaudio==0.13.0+cu117 torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 -q
```

> If `nvcc` reports CUDA 12.x, replace `cu117` with `cu118` in the pip URLs above.

```python
# Cell 4: install remaining requirements
!pip install -r requirements.txt -q
```

```python
# Cell 5: install vendored PerforatedAI wheel (Linux cp310)
!pip install PerforatedAI/releases/e3.1.0/perforatedai-3.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -q
```

```python
# Cell 6: verify
import torch, librosa, perforatedai, perforatedbp
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("librosa:", librosa.__version__)
print("perforatedai OK | perforatedbp OK")
```

> **Note:** Colab resets the runtime between sessions — you need to re-run the install cells each time,
> or save the environment to Google Drive and mount it.

## 1. Download ESC-50 into the repo-local layout
This repo-local layout is the easiest to reproduce:
- metadata can come from either `datasets/esc50.csv` or `datasets/ESC-50/meta/esc50.csv`
- the official ESC-50 clone provides the audio files under `datasets/ESC-50/audio`

With this layout, the ESC-50 scripts do not need dataset path env vars.

```bash
cd ~/11-785_EfficientAT
mkdir -p datasets
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git datasets/ESC-50
```

## 2. Sanity check the dataset layout
The current loader auto-detects:
- `datasets/esc50.csv` or `datasets/ESC-50/meta/esc50.csv`
- `datasets/ESC-50/audio` or `datasets/ESC-50/audio_32k`

```bash
cd ~/11-785_EfficientAT
ls datasets/ESC-50/meta/esc50.csv
ls datasets/ESC-50/audio | head
python ex_esc50_pai_pretrained.py --help
```

## 3. Baseline: AudioSet pretrained MobileNet -> ESC-50 fine-tuning
This is the plain ESC-50 fine-tuning baseline without dendrites.

```bash
cd ~/11-785_EfficientAT
python ex_esc50.py --cuda --pretrained --model_name=mn10_as --fold=1
```

## 4. PAI: AudioSet pretrained MobileNet -> ESC-50 fine-tuning + dendrite search
This starts from AudioSet pretrained `mn10_as`, fine-tunes on ESC-50, and runs PAI dendrite search in the same script.

```bash
cd ~/11-785_EfficientAT
python ex_esc50_pai_pretrained.py \
    --cuda --model_name=mn10_as --fold=1 \
    --experiment_name=ESC50_PAI_pretrained
```

## 5. PAI + Perforated Backpropagation
Use this only if the Perforated Backpropagation package/license is available in your environment.

```bash
cd ~/11-785_EfficientAT
python ex_esc50_pai_pretrained.py \
    --cuda --model_name=mn10_as --fold=1 \
    --perforated_bp \
    --experiment_name=ESC50_PAI_pretrained_pbp
```

## 6. Same commands when the dataset lives outside the repo
If you keep ESC-50 somewhere else, point the scripts at the dataset root.

```bash
cd ~/11-785_EfficientAT
export EFFICIENTAT_ESC50_DIR=~/datasets/ESC-50

python ex_esc50.py --cuda --pretrained --model_name=mn10_as --fold=1

python ex_esc50_pai_pretrained.py \
    --cuda --model_name=mn10_as --fold=1 \
    --experiment_name=ESC50_PAI_pretrained

python ex_esc50_pai_pretrained.py \
    --cuda --model_name=mn10_as --fold=1 \
    --perforated_bp \
    --experiment_name=ESC50_PAI_pretrained_pbp
```

## 7. Optional explicit overrides
If metadata CSV and audio directory are split across different places, you can override them directly.

```bash
export EFFICIENTAT_ESC50_META_CSV=/path/to/esc50.csv
export EFFICIENTAT_ESC50_AUDIO_DIR=/path/to/audio
```

## Evaluation
python ex_esc50.py   --cuda   --eval_only   --model_name=mn10_as   --fold=1   --checkpoint_path=wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt


# Results (ESC-50 fold 1 evaluation)
## 1. baseline: MobileNet + ESC50 finetuning
python ex_esc50.py   --cuda   --eval_only   --model_name=mn10_as   --fold=1   --checkpoint_path=wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt

  checkpoint: wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt
  accuracy: 0.9525
  val_loss: 0.2822

## 2. Adding dendrites
### 2.1 adding dendrites, but use regular backprop
EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 python ex_esc50_perforated.py     --cuda --eval_only --model_name=mn10_as --fold=1     --perforated_bp     --checkpoint_path=ESC50_PAI/backup/best_model.pt

  checkpoint: ESC50_PAI/backup/best_model.pt
  accuracy: 0.9575
  val_loss: 0.2698
