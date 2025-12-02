import os
import argparse
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# For Bayesian optimization; install with:
#   pip install bayesian-optimization
from bayes_opt import BayesianOptimization

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE_CANDIDATES = [16, 32, 64, 128, 256, 512, 1024]
ACTIVATIONS = ["relu", "sigmoid", "tanh"]

SEARCH_EPOCHS = 10      # epochs used during GA / BO evaluation
FINAL_EPOCHS = 30       # epochs for final training on train+val
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 10
RANDOM_SEED = 42

# GA hyperparameters
GA_POP_SIZE = 10
GA_GENERATIONS = 10
GA_CROSSOVER_PROB = 0.9
GA_MUTATION_PROB = 0.2
GA_MAX_AGE = 5  # for age-based selection

# BO hyperparameters
BO_INIT_POINTS = 5
BO_ITER = 15


# -----------------------------
# Utility: Seed
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset loading
# -----------------------------
class FederatedEMNISTDataset(Dataset):
    """
    Simple Dataset wrapper around numpy arrays.
    Expects images of shape (N, 28, 28) and labels (N,).
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        assert len(images) == len(labels)
        self.X = torch.from_numpy(images).float().unsqueeze(1)
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def filter_digits(images: np.ndarray, labels: np.ndarray):
    images = np.asarray(images)
    labels = np.asarray(labels).astype(int).reshape(-1)

    mask = (labels >= 0) & (labels <= 9)
    return images[mask], labels[mask]


def load_federated_emnist_data(
    train_file: str = "train_data.npy",
    test_file: str = "test_data.npy",
    val_ratio: float = 0.2
) -> Tuple[Dataset, Dataset, Dataset]:

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    all_train_images = []
    all_train_labels = []

    # Filter per-client data
    for client in train_data:
        imgs, labs = filter_digits(client["images"], client["labels"])
        all_train_images.append(imgs)
        all_train_labels.append(labs)

    all_train_images = np.concatenate(all_train_images, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)

    # Filter test set digits only
    test_images, test_labels = filter_digits(
        test_data[0]["images"],
        test_data[0]["labels"]
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        all_train_images, all_train_labels,
        test_size=val_ratio,
        stratify=all_train_labels,
        random_state=42
    )

    train_dataset = FederatedEMNISTDataset(X_train, y_train)
    val_dataset = FederatedEMNISTDataset(X_val, y_val)
    test_dataset = FederatedEMNISTDataset(test_images, test_labels)

    return train_dataset, val_dataset, test_dataset



# -----------------------------
# Model definition
# -----------------------------
def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


class SimpleMLP(nn.Module):
    def __init__(self, activation_name: str = "relu"):
        super().__init__()
        act = get_activation(activation_name)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training / evaluation helpers
# -----------------------------
def evaluate_f1(model: nn.Module, data_loader: DataLoader, device) -> float:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return f1_score(all_targets, all_preds, average="macro")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    epochs: int,
    lr: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
    desc_prefix: str = ""
) -> Tuple[float, List[float], List[float]]:
    """
    Train model and return (best_val_f1, train_f1_history, val_f1_history).
    Uses tqdm for epoch & batch progress.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.to(device)
    best_val_f1 = 0.0
    train_f1_history = []
    val_f1_history = []

    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_targets = []

        epoch_desc = f"{desc_prefix}Epoch {epoch + 1}/{epochs}"
        for inputs, labels in tqdm(train_loader, desc=epoch_desc, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        train_f1 = f1_score(all_targets, all_preds, average="macro")
        val_f1 = evaluate_f1(model, val_loader, device)

        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return best_val_f1, train_f1_history, val_f1_history


def make_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    return train_loader, val_loader


# -----------------------------
# Genetic Algorithm
# -----------------------------
def init_population(pop_size: int) -> List[Dict]:
    """
    Each individual is represented by:
      - batch_idx (index in BATCH_SIZE_CANDIDATES)
      - act_idx (index in ACTIVATIONS)
      - age
      - fitness
    """
    population = []
    for _ in range(pop_size):
        ind = {
            "batch_idx": np.random.randint(0, len(BATCH_SIZE_CANDIDATES)),
            "act_idx": np.random.randint(0, len(ACTIVATIONS)),
            "age": 0,
            "fitness": None,
        }
        population.append(ind)
    return population


def roulette_wheel_selection(population: List[Dict]) -> Dict:
    """
    Roulette selection based on fitness.
    If all fitness are equal or None, selects uniformly.
    """
    fitnesses = np.array([ind["fitness"] for ind in population], dtype=float)

    if np.any(np.isnan(fitnesses)) or np.all(fitnesses <= 0):
        # Default to uniform selection
        idx = np.random.randint(len(population))
        return population[idx]

    # Make sure all are positive
    fitnesses = fitnesses - fitnesses.min() + 1e-8
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def one_point_crossover(parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
    """
    One-point crossover on a 2-gene chromosome [batch_idx, act_idx].
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    # With 2 genes, the only crossover point is between them.
    if np.random.rand() < 0.5:
        # Swap activation index
        child1["act_idx"], child2["act_idx"] = child2["act_idx"], child1["act_idx"]
    else:
        # Swap batch index
        child1["batch_idx"], child2["batch_idx"] = child2["batch_idx"], child1["batch_idx"]

    child1["age"] = 0
    child2["age"] = 0
    child1["fitness"] = None
    child2["fitness"] = None

    return child1, child2


def mutate(ind: Dict, mutation_prob: float):
    if np.random.rand() < mutation_prob:
        ind["batch_idx"] = np.random.randint(0, len(BATCH_SIZE_CANDIDATES))
    if np.random.rand() < mutation_prob:
        ind["act_idx"] = np.random.randint(0, len(ACTIVATIONS))
    ind["fitness"] = None  # needs reevaluation


def age_based_replacement(population: List[Dict], max_age: int, pop_size: int):
    """
    Remove individuals older than max_age and replace with random newcomers.
    """
    # Increment ages
    for ind in population:
        ind["age"] += 1

    # Keep those within age limit
    survivors = [ind for ind in population if ind["age"] <= max_age]

    # Add random new individuals if needed
    while len(survivors) < pop_size:
        survivors.append({
            "batch_idx": np.random.randint(0, len(BATCH_SIZE_CANDIDATES)),
            "act_idx": np.random.randint(0, len(ACTIVATIONS)),
            "age": 0,
            "fitness": None,
        })

    # If too many, truncate (e.g., by youngest first)
    survivors.sort(key=lambda ind: ind["age"])
    return survivors[:pop_size]


def evaluate_individual(
    ind: Dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device
) -> float:
    """
    Train a model with the hyperparameters of 'ind' and return validation F1.
    """
    batch_size = BATCH_SIZE_CANDIDATES[ind["batch_idx"]]
    act_name = ACTIVATIONS[ind["act_idx"]]

    train_loader, val_loader = make_data_loaders(train_dataset, val_dataset, batch_size)
    model = SimpleMLP(act_name)

    best_val_f1, _, _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=SEARCH_EPOCHS,
        desc_prefix=f"[GA {act_name}, B={batch_size}] "
    )
    return best_val_f1


def run_genetic_algorithm(
    train_dataset: Dataset,
    val_dataset: Dataset,
    device
) -> Tuple[Dict, List[float], List[float]]:
    """
    Run GA to optimize batch size and activation.
    Returns best_individual, avg_fitness_per_gen, max_fitness_per_gen.
    """
    population = init_population(GA_POP_SIZE)
    avg_fitness_history = []
    max_fitness_history = []

    for gen in tqdm(range(GA_GENERATIONS), desc="GA Generations"):
        # Evaluate fitness
        for ind in population:
            if ind["fitness"] is None:
                ind["fitness"] = evaluate_individual(ind, train_dataset, val_dataset, device)

        fitnesses = [ind["fitness"] for ind in population]
        avg_fitness = float(np.mean(fitnesses))
        max_fitness = float(np.max(fitnesses))
        avg_fitness_history.append(avg_fitness)
        max_fitness_history.append(max_fitness)

        # Create next generation via roulette selection + crossover + mutation
        new_population = []
        while len(new_population) < GA_POP_SIZE:
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)
            if np.random.rand() < GA_CROSSOVER_PROB:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                child1["age"] = 0
                child2["age"] = 0
                child1["fitness"] = None
                child2["fitness"] = None

            mutate(child1, GA_MUTATION_PROB)
            mutate(child2, GA_MUTATION_PROB)

            new_population.append(child1)
            if len(new_population) < GA_POP_SIZE:
                new_population.append(child2)

        # Age-based selection / replacement
        population = age_based_replacement(new_population, GA_MAX_AGE, GA_POP_SIZE)

    # Re-evaluate final population to get final best
    for ind in population:
        if ind["fitness"] is None:
            ind["fitness"] = evaluate_individual(ind, train_dataset, val_dataset, device)

    best_ind = max(population, key=lambda ind: ind["fitness"])

    # Plot GA fitness curves
    plt.figure()
    plt.plot(avg_fitness_history, label="Average fitness")
    plt.plot(max_fitness_history, label="Best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Validation F1 (fitness)")
    plt.title("GA: Fitness vs Generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_fitness.png")

    return best_ind, avg_fitness_history, max_fitness_history


# -----------------------------
# Bayesian Optimization
# -----------------------------
def black_box_eval_bo(
    batch_idx_cont: float,
    act_idx_cont: float,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device
) -> float:
    """
    Black-box function for Bayesian Optimization.
    Arguments are continuous; we snap to nearest valid indices.
    Returns validation F1.
    """
    batch_idx = int(round(batch_idx_cont))
    act_idx = int(round(act_idx_cont))
    batch_idx = np.clip(batch_idx, 0, len(BATCH_SIZE_CANDIDATES) - 1)
    act_idx = np.clip(act_idx, 0, len(ACTIVATIONS) - 1)

    batch_size = BATCH_SIZE_CANDIDATES[batch_idx]
    act_name = ACTIVATIONS[act_idx]

    train_loader, val_loader = make_data_loaders(train_dataset, val_dataset, batch_size)
    model = SimpleMLP(act_name)

    best_val_f1, _, _ = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=SEARCH_EPOCHS,
        desc_prefix=f"[BO {act_name}, B={batch_size}] "
    )

    return best_val_f1


def run_bayesian_optimization(
    train_dataset: Dataset,
    val_dataset: Dataset,
    device
) -> Dict:
    """
    Run Bayesian Optimization to find best batch size / activation.
    Returns a dict with 'batch_idx', 'act_idx', 'fitness'.
    """
    def wrapper(batch_idx_cont, act_idx_cont):
        return black_box_eval_bo(
            batch_idx_cont,
            act_idx_cont,
            train_dataset,
            val_dataset,
            device
        )

    pbounds = {
        "batch_idx_cont": (0, len(BATCH_SIZE_CANDIDATES) - 1),
        "act_idx_cont": (0, len(ACTIVATIONS) - 1),
    }

    optimizer = BayesianOptimization(
        f=wrapper,
        pbounds=pbounds,
        random_state=RANDOM_SEED,
        verbose=2
    )

    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_ITER)

    best = optimizer.max 
    best_batch_idx = int(round(best["params"]["batch_idx_cont"]))
    best_act_idx = int(round(best["params"]["act_idx_cont"]))

    best_batch_idx = np.clip(best_batch_idx, 0, len(BATCH_SIZE_CANDIDATES) - 1)
    best_act_idx = np.clip(best_act_idx, 0, len(ACTIVATIONS) - 1)

    best_ind = {
        "batch_idx": best_batch_idx,
        "act_idx": best_act_idx,
        "fitness": best["target"],
        "age": 0
    }

    return best_ind


# -----------------------------
# Final training & plotting
# -----------------------------
def train_and_test_final(
    best_ind: Dict,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    device,
    label_prefix: str
) -> float:
    """
    Retrain a model on (train + val) with the selected hyperparameters
    and evaluate on test. Also plots training F1 vs epochs.
    """
    batch_size = BATCH_SIZE_CANDIDATES[best_ind["batch_idx"]]
    act_name = ACTIVATIONS[best_ind["act_idx"]]

    print(f"{label_prefix} best hyperparameters: B={batch_size}, activation={act_name}")

    # Combine train & val
    full_train_images = torch.cat([train_dataset.X, val_dataset.X], dim=0).numpy()
    full_train_labels = torch.cat([train_dataset.y, val_dataset.y], dim=0).numpy()

    full_train_dataset = FederatedEMNISTDataset(full_train_images, full_train_labels)

    # Data loaders
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = SimpleMLP(act_name)
    best_val_f1, train_f1_history, _ = train_model(
        model,
        full_train_loader,
        test_loader,
        device,
        epochs=FINAL_EPOCHS,
        desc_prefix=f"[Final {label_prefix}] "
    )

    # Final test evaluation
    test_f1 = evaluate_f1(model, test_loader, device)
    print(f"{label_prefix} final test F1 = {test_f1:.4f}")

    # Plot training F1 vs epochs
    plt.figure()
    plt.plot(train_f1_history, label="Training F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score (macro)")
    plt.title(f"{label_prefix} - Training F1 vs Epochs")
    plt.legend()
    plt.tight_layout()
    fname = f"{label_prefix.lower().replace(' ', '_')}_training_f1.png"
    plt.savefig(fname)

    return test_f1


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="HW4 AutoML - GA and Bayesian Optimization")

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g. 0). Use -1 to force CPU."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["ga", "bo", "both"],
        help="Which tuning method to run."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train_data.npy",
        help="Path to federated train_data.npy"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test_data.npy",
        help="Path to federated test_data.npy"
    )

    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # Device selection (single-GPU)
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # Load data
    train_dataset, val_dataset, test_dataset = load_federated_emnist_data(
        train_file=args.train_file,
        test_file=args.test_file,
        val_ratio=0.2
    )

    if args.mode in ["ga", "both"]:
        print("\n=== Running Genetic Algorithm ===")
        best_ga_ind, avg_fit, max_fit = run_genetic_algorithm(
            train_dataset,
            val_dataset,
            device
        )
        ga_test_f1 = train_and_test_final(
            best_ga_ind,
            train_dataset,
            val_dataset,
            test_dataset,
            device,
            label_prefix="GA"
        )

    if args.mode in ["bo", "both"]:
        print("\n=== Running Bayesian Optimization ===")
        best_bo_ind = run_bayesian_optimization(
            train_dataset,
            val_dataset,
            device
        )
        bo_test_f1 = train_and_test_final(
            best_bo_ind,
            train_dataset,
            val_dataset,
            test_dataset,
            device,
            label_prefix="BO"
        )

    if args.mode == "both":
        print("\n=== Comparison ===")
        print("Compare chosen hyperparameters (B, activation) and test F1 from GA vs BO.")


if __name__ == "__main__":
    main()
