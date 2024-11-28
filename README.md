# Speculative Decoding with Mamba

This repository provides a Python implementation for speculative decoding using Mamba models.
Speculative decoding accelerates autoregressive generation by leveraging a small "draft" model to propose tokens and a larger "target" model to validate them, significantly reducing computational overhead.

This code, together with [this](https://github.com/jxiw/MambaInLlama), accompanies the Neurips 2024 paper [*The Mamba in the Llama: Distilling and Accelerating Hybrid Models*](https://arxiv.org/abs/2408.15237).

---

## Features

- **Speculative Decoding**: Use a draft model to accelerate decoding with a target model.
- **Mamba Models**: Includes support for various Mamba models with pre-trained checkpoints.
- **Customizable Generation Parameters**: Adjust decoding hyperparameters like temperature, top-k, and top-p.
- **CUDA Graph Support**: Optional CUDA graph acceleration for faster execution.

---

## Getting Started

### Installation

Follow the steps below to set up the environment and install dependencies:

```bash
# Create a conda environment
conda create --name specmamba python=3.11
conda activate specmamba

# Install PyTorch with CUDA support
conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install required Python packages
pip install causal_conv1d==1.4.0
pip install transformers
pip install flash_attn

# Install the repository
pip install -e .
```

---

### Usage

You can run the decoding script by specifying the prompt and other generation parameters:

```bash
python speculative_mamba/run.py \
    --prompt "Italy is a country" \
    --n_tokens_to_generate 64 \
    --K 3 \
    --model_target state-spaces/mamba-2.8b \
    --model_draft state-spaces/mamba-130m \
    --dtype float16 \
    --top_k 50 \
    --top_p 0.8 \
    --temperature 0.8 \
    --cg
```

#### Parameters

- **`--prompt`**: The initial text for token generation.
- **`--n_tokens_to_generate`**: Number of tokens to generate.
- **`--K`**: Speculative lookahead value (number of tokens drafted ahead).
- **`--model_target`**: Path to the target model (e.g., `state-spaces/mamba-2.8b`).
- **`--model_draft`**: Path to the draft model (e.g., `state-spaces/mamba-130m`).
- **`--dtype`**: Data type for model tensors (`float16`, `float32`, etc.).
- **`--top_k`**: Top-k sampling threshold.
- **`--top_p`**: Top-p sampling threshold.
- **`--temperature`**: Temperature value for controlling randomness in generation.
- **`--cg`**: Enable CUDA graph acceleration. Especially useful for smaller models.

---

### Example

Run the script with the default settings:

```bash
python speculative_mamba/run.py
```

Output:
```
Decoding...
Prompt processing + decoding time: 4364ms
Acceptance rate: 68.25%
Italy is a country that has always had an important role in international affairs, both in the economic and in the political sphere.

But in the last years, the country has been going through a period of great political instability.

In the last year, the country has had three different Prime Ministers: Mario Monti, En
```

*Note*: your output and acceptance rate will vary.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{junxiongdaniele2024mambainllama,
  title   = {The Mamba in the Llama: Distilling and Accelerating Hybrid Models},
  author  = {Junxiong Wang and Daniele Paliotta and Avner May and Alexander M. Rush and Tri Dao},
  journal = {arXiv preprint arXiv:2408.15237},
  year    = {2024}
}
```
