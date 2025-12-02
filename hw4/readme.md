# AutoML Hw4

This project performs hyperparameter optimization using Genetic Algorithms (GA) and Bayesian Optimization (BO) on the EMNIST digit subset.

---

## Environment Setup

Install dependencies:

```bash
pip install numpy torch scikit-learn tqdm matplotlib bayesian-optimization
```

Dataset

The script is using the same data files from HW3. Put test_data.npy and train_data.npy in the same folder as the script. All the preprocessing is handled in the script.

---

## Running the Code

### GA

```bash
python hw4_automl.py --mode ga --gpu 0
```

### BO
```bash
python hw4_automl.py --mode bo --gpu 0
```

### Or Both for direct comparison

```bash
python hw4_automl.py --mode both --gpu 0
```

All the hyperparameters for GA, BO and final training are set in the script. If you would like to change, please modify line 18-40 accordingly.

---

## CLI Arguments

# Command-Line Arguments

| Argument | Type | Default | Description |
|---------|-------|---------|-------------|
| `--mode {ga,bo,both}` | string | `both` | Selects which optimization method(s) to run. `ga` = Genetic Algorithm only, `bo` = Bayesian Optimization only, `both` = run both methods. |
| `--gpu <index>` | integer | `0` | Selects which GPU to use. Use `0`, `1`, `2`, … for GPUs, or `-1` to force CPU. Works with `CUDA_VISIBLE_DEVICES` remapping. |
| `--train_file <path>` | string | `train_data.npy` | Path to the EMNIST training data file. |
| `--test_file <path>` | string | `test_data.npy` | Path to the EMNIST test data file. |
