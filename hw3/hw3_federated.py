"""
HW3 - Federated Learning

Run examples:
- Sequential FedAvg (Part 1, basic):
    python hw3_federated.py --mode seq --rounds 50 --client_frac 0.1 --local_epochs 2

- Ray parallel clients (Part 1, parallel):
    python hw3_federated.py --mode ray --rounds 50 --local_epochs 2

- Differential privacy experiments (Part 2):
    python hw3_federated.py --mode dp --rounds 50 --noise_scales 0.0 0.01 0.05 0.1
"""

import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Dataset & Utilities
# -------------------------


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels, noise_scale: float = 0.0):
        """
        images: (N, 28, 28) array or list of 2D arrays
        labels: (N,) array or list of ints
        noise_scale: Laplace scale b; if > 0, add Laplace noise
        """
        # Convert list inputs to numpy arrays
        if isinstance(images, list):
            images = np.stack(images, axis=0)
        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.int64)

        assert images.shape[0] == labels.shape[0]
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.noise_scale = noise_scale


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.noise_scale > 0.0:
            noise = np.random.laplace(loc=0.0, scale=self.noise_scale, size=img.shape).astype(
                np.float32
            )
            img = img + noise
        img = np.clip(img, 0.0, 1.0)
        img = torch.from_numpy(img).view(-1)  # flatten 28x28 -> 784
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def split_client_data(
    train_data: np.ndarray, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Split each client's data into train/val.
    Returns:
        client_train_splits: list of (images, labels) for each client (both np.ndarray)
        client_val_splits:   list of (images, labels) for each client (both np.ndarray)
    """
    rng = np.random.RandomState(seed)
    num_clients = len(train_data)
    client_train_splits = []
    client_val_splits = []

    for cid in range(num_clients):
        images_list = train_data[cid]["images"]
        labels_list = train_data[cid]["labels"]

        images = np.stack(images_list, axis=0).astype(np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        n = images.shape[0]
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(n * (1.0 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

        client_train_splits.append((images[train_idx], labels[train_idx]))
        client_val_splits.append((images[val_idx], labels[val_idx]))

    return client_train_splits, client_val_splits


def make_dataloader_from_arrays(
    images,
    labels,
    batch_size: int = 64,
    shuffle: bool = True,
    noise_scale: float = 0.0,
) -> DataLoader:
    dataset = NumpyImageDataset(images, labels, noise_scale=noise_scale)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Model Definition
# -------------------------


class SimpleFedEMNISTModel(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 62):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -------------------------
# Training / Evaluation
# -------------------------


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    """
    Train a local model for given epochs; returns:
    - updated state_dict
    - average training loss
    - training accuracy
    - num_samples
    """
    model = copy.deepcopy(model)  # local copy
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return copy.deepcopy(model.state_dict()), avg_loss, acc, total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns:
    - average loss
    - accuracy
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def aggregate_fedavg(
    client_states: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg: weighted average of client parameters by number of samples.
    """
    assert len(client_states) == len(client_sizes)
    total_samples = float(sum(client_sizes))

    global_state = copy.deepcopy(client_states[0])
    for key in global_state.keys():
        global_state[key] = global_state[key] * (client_sizes[0] / total_samples)

    for state, size in zip(client_states[1:], client_sizes[1:]):
        weight = size / total_samples
        for key in global_state.keys():
            global_state[key] += state[key] * weight

    return global_state


# -------------------------
# FedAvg Sequential (Part 1)
# -------------------------


def run_fedavg_sequential(
    train_clients: List[Tuple[np.ndarray, np.ndarray]],
    val_clients: List[Tuple[np.ndarray, np.ndarray]],
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_rounds: int = 50,
    client_frac: float = 0.1,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.01,
    noise_scale: float = 0.0,
):
    """
    Basic FedAvg loop (no Ray). Use this for:
    - Part 1, Q1 (different C, E)
    - Part 2, differential privacy when noise_scale > 0

    Returns logs that you can use to plot later.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = len(train_clients)
    m = max(1, int(client_frac * num_clients))

    # Build global model
    model = SimpleFedEMNISTModel()
    model.to(device)

    # Prebuild test loader
    test_loader = make_dataloader_from_arrays(
        test_images, test_labels, batch_size=batch_size, shuffle=False
    )

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    print(f"Running FedAvg (sequential) on device: {device}")
    print(f"Rounds: {num_rounds}, clients per round: {m}, local epochs: {local_epochs}")

    for r in tqdm(range(num_rounds), desc="Communication rounds"):
        # Sample clients
        selected_clients = random.sample(range(num_clients), m)

        client_states = []
        client_sizes = []
        round_train_losses = []
        round_train_accs = []

        for cid in selected_clients:
            images, labels = train_clients[cid]
            train_loader = make_dataloader_from_arrays(
                images,
                labels,
                batch_size=batch_size,
                shuffle=True,
                noise_scale=noise_scale,
            )

            local_state, train_loss, train_acc, num_samples = train_local(
                model, train_loader, epochs=local_epochs, lr=lr, device=device
            )
            client_states.append(local_state)
            client_sizes.append(num_samples)
            round_train_losses.append(train_loss)
            round_train_accs.append(train_acc)

        # Aggregate
        global_state = aggregate_fedavg(client_states, client_sizes)
        model.load_state_dict(global_state)

        # Compute aggregated train stats (simple average across clients)
        avg_round_train_loss = float(np.mean(round_train_losses))
        avg_round_train_acc = float(np.mean(round_train_accs))

        # Validation: aggregate all client val data
        val_images_all = np.concatenate([val_clients[cid][0] for cid in selected_clients], axis=0)
        val_labels_all = np.concatenate([val_clients[cid][1] for cid in selected_clients], axis=0)
        val_loader = make_dataloader_from_arrays(
            val_images_all,
            val_labels_all,
            batch_size=batch_size,
            shuffle=False,
            noise_scale=0.0,  # no noise in validation
        )
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        train_loss_log.append(avg_round_train_loss)
        train_acc_log.append(avg_round_train_acc)
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

    # Final evaluation on test data (held-out)
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    logs = {
        "train_loss": train_loss_log,
        "train_acc": train_acc_log,
        "val_loss": val_loss_log,
        "val_acc": val_acc_log,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    return model, logs


# -------------------------
# Ray-based Parallel Clients (Part 1 – Parallel)
# -------------------------

def run_fedavg_ray(
    train_clients: List[Tuple[np.ndarray, np.ndarray]],
    val_clients: List[Tuple[np.ndarray, np.ndarray]],
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_rounds: int = 50,
    clients_per_round: int = 4,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.01,
):
    """
    FedAvg with parallel client updates using Ray Actors.
    - Each client uses <= 1 CPU and <= 1 GPU.
    """

    import ray  # local import to avoid dependency if unused

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Ray FedAvg will run on CPU (to avoid CUDA kernel-image issues).")

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    num_clients = len(train_clients)

    # @ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
    @ray.remote(num_cpus=2)   # <-- no num_gpus
    class ClientActor:
        def __init__(self, images, labels, batch_size, lr, local_epochs):
            # Force CPU in Ray workers
            self.device = torch.device("cpu")
            self.images = images
            self.labels = labels
            self.batch_size = batch_size
            self.lr = lr
            self.local_epochs = local_epochs

        def local_update(self, global_state_dict):
            model = SimpleFedEMNISTModel()
            model.load_state_dict(global_state_dict)
            train_loader = make_dataloader_from_arrays(
                self.images, self.labels, batch_size=self.batch_size, shuffle=True
            )
            local_state, train_loss, train_acc, num_samples = train_local(
                model,
                train_loader,
                epochs=self.local_epochs,
                lr=self.lr,
                device=self.device,
            )
            return local_state, train_loss, train_acc, num_samples

    test_loader = make_dataloader_from_arrays(
        test_images,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    model = SimpleFedEMNISTModel()
    model.to(device)

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    print(f"Running FedAvg with Ray on device: {device}")
    print(f"Rounds: {num_rounds}, clients per round: {clients_per_round}, local epochs: {local_epochs}")

    for r in tqdm(num_range := range(num_rounds), desc="Communication rounds (Ray)"):
        selected_clients = random.sample(range(num_clients), clients_per_round)

        # Create actors for selected clients
        actors = []
        for cid in selected_clients:
            images, labels = train_clients[cid]
            actor = ClientActor.remote(images, labels, batch_size, lr, local_epochs)
            actors.append(actor)

        # Broadcast global weights
        global_state = copy.deepcopy(model.state_dict())
        # Trigger local updates in parallel
        futures = [actor.local_update.remote(global_state) for actor in actors]
        results = ray.get(futures)

        client_states = []
        client_sizes = []
        round_train_losses = []
        round_train_accs = []

        for local_state, train_loss, train_acc, num_samples in results:
            client_states.append(local_state)
            client_sizes.append(num_samples)
            round_train_losses.append(train_loss)
            round_train_accs.append(train_acc)

        # Aggregate
        new_global_state = aggregate_fedavg(client_states, client_sizes)
        model.load_state_dict(new_global_state)

        avg_round_train_loss = float(np.mean(round_train_losses))
        avg_round_train_acc = float(np.mean(round_train_accs))

        # Validation: aggregate data of selected clients
        val_images_all = np.concatenate([val_clients[cid][0] for cid in selected_clients], axis=0)
        val_labels_all = np.concatenate([val_clients[cid][1] for cid in selected_clients], axis=0)
        val_loader = make_dataloader_from_arrays(
            val_images_all, val_labels_all, batch_size=batch_size, shuffle=False
        )
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        train_loss_log.append(avg_round_train_loss)
        train_acc_log.append(avg_round_train_acc)
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Loss (Ray): {test_loss:.4f}, Test Accuracy (Ray): {test_acc:.4f}")

    logs = {
        "train_loss": train_loss_log,
        "train_acc": train_acc_log,
        "val_loss": val_loss_log,
        "val_acc": val_acc_log,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    ray.shutdown()
    return model, logs


# -------------------------
# DP Experiments (Part 2)
# -------------------------


def run_dp_experiments(
    train_clients: List[Tuple[np.ndarray, np.ndarray]],
    val_clients: List[Tuple[np.ndarray, np.ndarray]],
    test_images: np.ndarray,
    test_labels: np.ndarray,
    noise_scales: List[float],
    num_rounds: int = 50,
    client_frac: float = 0.1,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.01,
):
    """
    For each noise scale, run FedAvg sequential with Laplace noise added to local training data.
    Returns a dictionary of results for plotting later.
    """
    results = {}
    for b in noise_scales:
        print(f"\n=== DP FedAvg with noise scale b={b} ===")
        _, logs = run_fedavg_sequential(
            train_clients,
            val_clients,
            test_images,
            test_labels,
            num_rounds=num_rounds,
            client_frac=client_frac,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            noise_scale=b,
        )
        results[b] = logs
    return results


# -------------------------
# Main
# -------------------------


def describe_data_distribution(train_data: np.ndarray, num_clients_to_show: int = 5):
    """
    Helper to print basic class distribution overall and for some clients.
    You can extend this to generate plots in your report.
    """
    num_clients = len(train_data)
    print(f"Total clients: {num_clients}")

    # Overall distribution
    all_labels = np.concatenate([train_data[cid]["labels"] for cid in range(num_clients)], axis=0)
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Overall class distribution (label: count):")
    for u, c in zip(unique, counts):
        print(f"{u}: {c}")
    print()

    # Per-client distribution (5 clients of your choice)
    clients_to_show = random.sample(range(num_clients), num_clients_to_show)
    for cid in clients_to_show:
        labels = train_data[cid]["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {cid} class distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        print()

# -------------------------
# Plotting Helpers
# -------------------------


def plot_training_curves(logs, title_prefix="FedAvg", out_prefix="fedavg"):
    """
    logs: dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
          each is a list over communication rounds
    Produces:
      - {out_prefix}_loss.png
      - {out_prefix}_accuracy.png
    """
    rounds = range(1, len(logs["train_loss"]) + 1)

    # Loss curves
    plt.figure()
    plt.plot(rounds, logs["train_loss"], label="Train loss")
    plt.plot(rounds, logs["val_loss"], label="Validation loss")
    plt.xlabel("Communication round")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}: Loss vs Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", dpi=200)
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(rounds, logs["train_acc"], label="Train accuracy")
    plt.plot(rounds, logs["val_acc"], label="Validation accuracy")
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}: Accuracy vs Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_accuracy.png", dpi=200)
    plt.close()


def summarize_and_print_final_metrics(logs, label="Run"):
    print(f"\n=== {label} summary ===")
    print(f"Final train loss: {logs['train_loss'][-1]:.4f}")
    print(f"Final train acc : {logs['train_acc'][-1]:.4f}")
    print(f"Final val loss  : {logs['val_loss'][-1]:.4f}")
    print(f"Final val acc   : {logs['val_acc'][-1]:.4f}")
    if "test_loss" in logs and "test_acc" in logs:
        print(f"Test loss       : {logs['test_loss']:.4f}")
        print(f"Test acc        : {logs['test_acc']:.4f}")


def plot_dp_results(dp_results, out_prefix="dp"):
    """
    dp_results: dict[noise_scale] -> logs (same structure as run_fedavg_sequential logs)
    Produces:
      - {out_prefix}_noise_vs_accuracy.png
      - {out_prefix}_noise_vs_loss.png
    """

    noise_scales = sorted(dp_results.keys())

    final_train_acc = []
    final_val_acc = []
    final_test_acc = []

    final_train_loss = []
    final_val_loss = []
    final_test_loss = []

    for b in noise_scales:
        logs = dp_results[b]
        final_train_acc.append(logs["train_acc"][-1])
        final_val_acc.append(logs["val_acc"][-1])
        final_test_acc.append(logs["test_acc"])

        final_train_loss.append(logs["train_loss"][-1])
        final_val_loss.append(logs["val_loss"][-1])
        final_test_loss.append(logs["test_loss"])

    # Accuracy vs noise scale
    plt.figure()
    plt.plot(noise_scales, final_train_acc, marker="o", label="Final train acc")
    plt.plot(noise_scales, final_val_acc, marker="o", label="Final val acc")
    plt.plot(noise_scales, final_test_acc, marker="o", label="Test acc")
    plt.xlabel("Noise scale b")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise Scale (Laplace noise)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_noise_vs_accuracy.png", dpi=200)
    plt.close()

    # Loss vs noise scale (optional but nice for the report)
    plt.figure()
    plt.plot(noise_scales, final_train_loss, marker="o", label="Final train loss")
    plt.plot(noise_scales, final_val_loss, marker="o", label="Final val loss")
    plt.plot(noise_scales, final_test_loss, marker="o", label="Test loss")
    plt.xlabel("Noise scale b")
    plt.ylabel("Loss")
    plt.title("Loss vs Noise Scale (Laplace noise)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_noise_vs_loss.png", dpi=200)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train_data.npy")
    parser.add_argument("--test_path", type=str, default="test_data.npy")

    parser.add_argument("--mode", type=str, choices=["seq", "ray", "dp"], default="seq")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--client_frac", type=float, default=0.1)
    parser.add_argument("--clients_per_round", type=int, default=4)  # for Ray
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)

    # For seq mode
    parser.add_argument("--sweep", action="store_true",
                        help="If set, run a small sweep over C and E (sequential FedAvg only).",)

    # For DP mode
    parser.add_argument(
        "--noise_scales",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.05, 0.1],
        help="List of Laplace scales b for DP experiments.",
    )

    args = parser.parse_args()
    set_seed(42)

    # Load data
    train_data = np.load(args.train_path, allow_pickle=True)
    test_data = np.load(args.test_path, allow_pickle=True)

    # Each train_data[i] is a client dict with 'images' and 'labels'
    client_train_splits, client_val_splits = split_client_data(train_data, val_ratio=0.2, seed=42)

    # Test data is aggregated in test_data[0]
    test_images = test_data[0]["images"]
    test_labels = test_data[0]["labels"]

    # Convert lists from test_data to numpy arrays if needed
    if isinstance(test_images, list):
        test_images = np.stack(test_images, axis=0).astype(np.float32)
    if isinstance(test_labels, list):
        test_labels = np.array(test_labels, dtype=np.int64)


    # Optional: describe data distribution (for report)
    # print("=== Data Distribution Summary ===")
    # describe_data_distribution(train_data, num_clients_to_show=5)

    
    C_values = [0.02, 0.05, 0.10]
    E_values = [1, 2, 3, 5]

    if args.mode == "seq":
        if args.sweep:
            for C in C_values:
                for E in E_values:
                    print(f"\n=== Running Sequential FedAvg (C={C}, E={E}) ===")
                    _, logs = run_fedavg_sequential(
                        client_train_splits,
                        client_val_splits,
                        test_images,
                        test_labels,
                        num_rounds=args.rounds,
                        client_frac=C,
                        local_epochs=E,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        noise_scale=0.0,
                    )

                    summarize_and_print_final_metrics(
                        logs,
                        label=f"Seq FedAvg (C={C}, E={E})",
                    )

                    exp_prefix = f"seq_C{C}_E{E}"
                    plot_training_curves(
                        logs,
                        title_prefix=f"Seq FedAvg (C={C}, E={E})",
                        out_prefix=exp_prefix,
                    )
        else:
            # original single-run behavior
            print("\n=== Running Sequential FedAvg ===")
            _, logs = run_fedavg_sequential(
                client_train_splits,
                client_val_splits,
                test_images,
                test_labels,
                num_rounds=args.rounds,
                client_frac=args.client_frac,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                noise_scale=0.0,
            )

            summarize_and_print_final_metrics(
                logs,
                label=f"Seq FedAvg (C={args.client_frac}, E={args.local_epochs})",
            )

            exp_prefix = f"seq_C{args.client_frac}_E{args.local_epochs}"
            plot_training_curves(
                logs,
                title_prefix=f"Seq FedAvg (C={args.client_frac}, E={args.local_epochs})",
                out_prefix=exp_prefix,
            )


    elif args.mode == "ray":
        print("\n=== Running Ray-based FedAvg (Parallel Clients) ===")
        _, logs = run_fedavg_ray(
            client_train_splits,
            client_val_splits,
            test_images,
            test_labels,
            num_rounds=args.rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        summarize_and_print_final_metrics(logs, label="Ray FedAvg")
        plot_training_curves(
            logs,
            title_prefix="Ray FedAvg (Parallel Clients)",
            out_prefix="ray_fedavg",
        )
        # np.save("ray_logs.npy", logs, allow_pickle=True)

    elif args.mode == "dp":
        print("\n=== Running Differential Privacy Experiments ===")
        results = run_dp_experiments(
            client_train_splits,
            client_val_splits,
            test_images,
            test_labels,
            noise_scales=args.noise_scales,
            num_rounds=args.rounds,
            client_frac=args.client_frac,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        # For convenience you can also save all results
        # np.save("dp_results.npy", results, allow_pickle=True)

        plot_dp_results(results, out_prefix="dp")



if __name__ == "__main__":
    main()
