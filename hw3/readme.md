# Federated Learning HW3

This project implements Federated Averaging (FedAvg) with:

- **Sequential FedAvg**
- **Ray-based parallel FedAvg**
- **Differential Privacy experiments**

The code trains a simple MLP on a FEMNIST-style federated dataset.

---

## 1. Environment Setup

Recommended Python:

- **Python 3.10 or 3.11**

Install dependencies:

```bash
pip install numpy torch tqdm matplotlib ray
```
Recommended Ray version:

```bash
pip install "ray[default]==2.9.3"
```

---

## 2. Running the Code

### Sequential FedAvg

Single Experiment

```bash
CUDA_VISIBLE_DEVICES=1 python hw3_federated.py \
  --mode seq \
  --rounds 120 \
  --client_frac 0.1 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01
```

A Series of Experiment. The scales of client per round and local epoch can be modified in the script line 738 and 739.

```bash
CUDA_VISIBLE_DEVICES=1 python hw3_federated.py \
  --mode seq \
  --rounds 120 \
  --batch_size 64 \
  --lr 0.01
  --sweep
```

### Ray-based FedAvg (Parallel Clients)

Current implementation is on CPU, in order to avoid CUDA kernel issues on deepdish server. GPU can be enabled by uncommenting line 385 and 396 (commenting line 386 and 397 at the same time).

```bash
python hw3_federated.py \
  --mode ray \
  --rounds 120 \
  --clients_per_round 4 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01
```

### Differential Privacy (Laplace Noise)

Run a set of experiments different noise scales.

```bash
python hw3_federated.py \
  --mode dp \
  --rounds 120 \
  --client_frac 0.1 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01 \
  --noise_scales 0.0 0.01 0.05 0.1
```

---

## 3. CLI Arguments

| Argument | Type | Default | Description |
|----------|------|----------|-------------|
| `--mode` | str | `seq` | Select training mode: `seq`, `ray`, or `dp`. |
| `--rounds` | int | `50` | Number of FedAvg communication rounds. |
| `--client_frac` | float | `0.1` | Fraction of clients sampled per round (used in `seq` and `dp` modes). |
| `--clients_per_round` | int | `4` | Number of clients sampled per round in Ray mode. |
| `--local_epochs` | int | `1` | Local epochs per client update (E). |
| `--batch_size` | int | `64` | Batch size used for each client’s local training. |
| `--lr` | float | `0.01` | Learning rate for SGD/Adam optimizer. |
| `--noise_scales` | float list | none | List of Laplace noise scales for DP mode (e.g., `0.0 0.01 0.05 0.1`). |
| `--seed` | int | `42` | Random seed for reproducibility. |


















 
