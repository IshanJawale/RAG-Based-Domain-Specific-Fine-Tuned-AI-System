#!/bin/bash
# One-time environment setup on the HPC login node:   bash setup_env.sh
# Uses the cluster's GPU build of PyTorch (module python) and adds only the small extra packages.
# Do NOT pip-install torch here - the module's build matches the cluster's GPUs.
set -euo pipefail

module load python
python3 -m venv --system-site-packages "$HOME/envs/legal-ft"
source "$HOME/envs/legal-ft/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

python3 - <<'EOF'
import torch, transformers, peft, accelerate
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| accelerate", accelerate.__version__)
print("(torch.cuda.is_available() is False on the login node - that is expected)")
EOF
echo "Environment ready: source ~/envs/legal-ft/bin/activate"
