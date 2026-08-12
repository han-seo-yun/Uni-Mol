#!/usr/bin/env bash
# =============================================================================
# Uni-Mol install script
#
# Usage:
#   ./install.sh local   -> CPU-only dev/smoke-test env (macOS/Linux, no GPU)
#   ./install.sh ksc      -> GPU env for KSC (or any CUDA cluster). Run this
#                            AFTER `module load cuda/<version>` (and cudnn if
#                            provided as a separate module).
#
# Verified locally on: macOS arm64, conda (miniconda3), no NVIDIA GPU, 2026-08-12.
# KSC steps below are derived from README.md / Uni-Core README but were NOT
# executed on KSC itself in this pass — verify module names on your cluster
# with `module avail cuda`.
# =============================================================================
set -euo pipefail

MODE="${1:-local}"
ENV_NAME="unimol"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$MODE" = "local" ]; then
  echo "== Creating CPU-only conda env '$ENV_NAME' (python 3.10) =="
  conda create -n "$ENV_NAME" python=3.10 -y
  conda activate "$ENV_NAME"

  echo "== Installing PyTorch (CPU/MPS build) =="
  pip install torch

  echo "== Installing unimol_tools dependencies =="
  pip install "numpy<2.0.0" "pandas<2.0.0" rdkit scikit-learn numba tqdm joblib pyyaml addict huggingface_hub

  echo "== Installing unimol_tools from this repo (Uni-Core NOT required) =="
  (cd unimol_tools && python setup.py install)

  echo "== [OPTIONAL] CPU-only smoke test of the raw unimol/ + Uni-Core pretraining code =="
  echo "   (verified to run 1-2 training steps on CPU; NOT suitable for real training)"
  pip install "rdkit-pypi==2022.9.3" lmdb ml_collections tensorboardX wandb tokenizers
  if [ ! -d /tmp/Uni-Core ]; then
    git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-Core
  fi
  (cd /tmp/Uni-Core && python setup.py install)   # no --enable-cuda-ext => CPU-only, custom kernels disabled

  echo "== Done. Verify with: =="
  echo "   python -c \"from unimol_tools import UniMolRepr; print('OK')\""
  echo "   unicore-train --help"

elif [ "$MODE" = "ksc" ]; then
  cat <<'EOF'
== KSC (GPU cluster) install steps — derived from README.md ==

1. Load CUDA module (check exact version available on KSC first):
     module avail cuda
     module load cuda/11.3     # match to a PyTorch build below, e.g. cu113

2. Create conda env:
     conda create -n unimol python=3.10 -y
     conda activate unimol

3. Install PyTorch matching the loaded CUDA version (README docker image
   uses pytorch1.11.0-cuda11.3; newer CUDA/PyTorch combos likely also work,
   but keep torch's CUDA version == `module load`ed CUDA version):
     pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
     # or a newer matching pair, e.g. torch==2.x with cu118/cu121, if that's
     # what KSC provides.

4. Install rdkit (pinned per unimol/README.md):
     pip install rdkit-pypi==2022.9.3

5. Install Uni-Core WITH CUDA extensions enabled (this is the critical
   difference vs. the local CPU install — omitting --enable-cuda-ext will
   silently give you a CPU-only build with no fused kernels):
     git clone https://github.com/dptech-corp/Uni-Core.git
     cd Uni-Core
     # CUDA_HOME must point at the loaded module's toolkit, and the CUDA
     # version must match the one PyTorch was built against (setup.py checks
     # this and will error if they mismatch).
     python setup.py install --enable-cuda-ext
     cd ..

6. Install remaining unicore/unimol runtime deps (discovered as missing
   at import time; not declared in Uni-Core's own requirements):
     pip install lmdb ml_collections tensorboardX wandb tokenizers

7. (Optional, for the wrapper/property-prediction path) install unimol_tools
   the same way as local -- it does NOT need Uni-Core or GPU-only kernels,
   but obviously trains much faster on GPU:
     cd unimol_tools && python setup.py install && cd ..

8. Download data/pretrained weights per unimol/README.md (large files, GB-scale):
     see the tables in unimol/README.md ("Uni-Mol's 3D conformation data",
     "Uni-Mol's pretrained model weights").

9. Run real training via the scripts in unimol/README.md, e.g. molecular
   pretraining:
     python -m torch.distributed.launch --nproc_per_node=$n_gpu \
       $(which unicore-train) $data_path --user-dir ./unimol ... --fp16 ...
   (n_gpu, batch sizes, --fp16 etc. as documented in unimol/README.md; --fp16
   requires CUDA and will not work on CPU.)

EOF
else
  echo "Unknown mode: $MODE (use 'local' or 'ksc')" >&2
  exit 1
fi
