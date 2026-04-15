# Online-Diffusion-Policy

An online diffusion policy based on Cosmos Policy. 

Things to Test 

-Current and Previous States In Latent Sequence

![Online Diffusion Architecture](pipelineOnlineDiffusion.png)

### Installation

```bash
### 1. Create the Conda Environment
conda env create -f environment.yml
conda activate onlineDiffusionPolicy

### 2. Install Python Dependencies

pip install -r requirements.txt

### 3. Install Cosmos Predict 2.5

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone and install cosmos-predict2.5
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
git lfs install
git lfs pull
uv sync --extra=cu128 --active --inexact
cd ..

cd cosmos-predict2.5
pip install -e . --no-deps

```