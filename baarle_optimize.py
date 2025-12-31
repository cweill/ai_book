#!/usr/bin/env python3
"""
Baarle Hertog Neuron Optimization Challenge

Goal: Achieve 99.5%+ accuracy with <100 hidden neurons.
Uses wandb for hyperparameter sweeps.
"""

import argparse
import io

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")  # Non-interactive backend for saving figures


def load_baarle_data(
    data_path: str = "data/Baarle-Nassau_-_Baarle-Hertog-en no legend.png",
):
    """Load and preprocess Baarle Hertog map data."""
    m = cv2.imread(data_path)[:, :, (2, 1, 0)]

    belgium_color = np.array([251, 234, 81])
    netherlands_color = np.array([255, 255, 228])

    netherlands_region = ((m - netherlands_color) ** 2).sum(-1) < 50
    belgium_region = ((m - belgium_color) ** 2).sum(-1) < 10000

    b_coords = np.array(np.where(belgium_region)).T.astype("float")
    n_coords = np.array(np.where(netherlands_region)).T.astype("float")

    # Flip and normalize coordinates to [-1, 1]
    belgium_coords = np.zeros_like(b_coords)
    netherlands_coords = np.zeros_like(n_coords)

    belgium_coords[:, 0] = b_coords[:, 1] / (960 / 2) - 1
    belgium_coords[:, 1] = (960 - b_coords[:, 0]) / (960 / 2) - 1
    netherlands_coords[:, 0] = n_coords[:, 1] / (960 / 2) - 1
    netherlands_coords[:, 1] = (960 - n_coords[:, 0]) / (960 / 2) - 1

    # Combine into dataset
    X = np.vstack((netherlands_coords, belgium_coords))
    y = np.concatenate(
        (np.zeros(len(netherlands_coords)), np.ones(len(belgium_coords)))
    ).astype("int")

    # Shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    return torch.FloatTensor(X), torch.tensor(y)


def viz_decision_boundary(model, device, res=256):
    """Generate decision boundary visualization."""
    # Load map image
    m = cv2.imread("data/Baarle-Nassau_-_Baarle-Hertog-en no legend.png")[
        :, :, (2, 1, 0)
    ]

    # Create probe grid
    probe = np.zeros((res, res, 2))
    for j, xx in enumerate(np.linspace(-1, 1, res)):
        for k, yy in enumerate(np.linspace(-1, 1, res)):
            probe[j, k] = [yy, xx]
    probe = probe.reshape(res**2, -1)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        probe_logits = model(torch.tensor(probe).float().to(device))
        probe_logits = probe_logits.cpu().numpy().reshape(res, res, 2)

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(m.mean(2), cmap="gray")
    ax.imshow(
        np.flipud(np.argmax(probe_logits, 2)),
        extent=[0, 960, 960, 0],
        alpha=0.7,
        cmap="viridis",
    )
    ax.set_title("Decision Boundary")
    ax.axis("off")

    # Convert to PIL Image for wandb
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def build_hidden_layers(
    num_layers: int, layer_width: int, taper_ratio: float
) -> list[int]:
    """Generate layer sizes with optional tapering."""
    layers = []
    current_width = layer_width
    for _ in range(num_layers):
        layers.append(max(1, int(current_width)))
        current_width *= taper_ratio
    return layers


def count_neurons(hidden_layers: list[int]) -> int:
    """Count total hidden neurons."""
    return sum(hidden_layers)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaarleNet(nn.Module):
    def __init__(
        self,
        hidden_layers: list[int],
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        self.hidden_layers_config = hidden_layers
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections

        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input layer
        self.input_layer = nn.Linear(2, hidden_layers[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.can_skip = []  # Track which layers can use skip connections

        for idx in range(len(hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers[idx], hidden_layers[idx + 1])
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_layers[idx + 1]))
            # Only use skip when dimensions match (identity add, no projection matrix)
            self.can_skip.append(hidden_layers[idx] == hidden_layers[idx + 1])

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 2)

    def forward(self, x):
        """Forward pass through the network."""
        # Input to first hidden layer
        x = self.input_layer(x)

        # Hidden layers with optional skip connections
        for i, layer in enumerate(self.hidden_layers):
            identity = x
            out = self.activation(layer(x))

            # Identity skip connection (no projection matrix)
            if self.use_skip_connections and self.can_skip[i]:
                out = out + identity

            if self.use_layer_norm and len(self.layer_norms) > i:
                out = self.layer_norms[i](out)
            x = out

        # Output layer
        return self.output_layer(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, X, y, device):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(X.to(device))
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y.to(device)).float().mean().item()
    return accuracy


def train(config=None, use_wandb=True):
    """Main training function with optional wandb integration."""
    if use_wandb:
        wandb.init(config=config, project="baarle-hertog-optimization")
        config = wandb.config
    else:
        # Convert dict to namespace for consistent attribute access
        class Config:
            pass

        cfg = Config()
        for k, v in config.items():
            setattr(cfg, k, v)
        config = cfg

    # Build architecture
    hidden_layers = build_hidden_layers(
        config.num_layers, config.layer_width, config.taper_ratio
    )
    total_neurons = count_neurons(hidden_layers)

    # Check neuron constraint (90-99 neurons for best results)
    if total_neurons < 90 or total_neurons > 99:
        if use_wandb:
            wandb.log({"filtered": True, "total_neurons": total_neurons})
            wandb.finish()
        print(f"Filtered: {total_neurons} neurons (must be 90-99)")
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data
    X, y = load_baarle_data()

    # Create dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Create model
    model = BaarleNet(
        hidden_layers=hidden_layers,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        use_skip_connections=config.use_skip_connections,
    ).to(device)

    total_params = count_parameters(model)

    # Log architecture info
    if use_wandb:
        wandb.log(
            {
                "hidden_layers": str(hidden_layers),
                "total_neurons": total_neurons,
                "total_parameters": total_params,
                "filtered": False,
            }
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop with plateau detection
    best_accuracy = 0.0
    epochs_without_improvement = 0
    patience = 5  # Stop if no improvement for 5 epochs
    min_delta = 0.001  # Minimum improvement threshold

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        accuracy = evaluate(model, X, y, device)

        # Check for improvement
        if accuracy > best_accuracy + min_delta:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "accuracy": accuracy,
                    "best_accuracy": best_accuracy,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )

        # Early stopping if we hit target
        if accuracy >= 0.995:
            print(f"Target accuracy reached at epoch {epoch}!")
            break

        # Early stopping if plateaued
        if epochs_without_improvement >= patience:
            print(f"Plateaued at {best_accuracy:.4f} after {epoch} epochs. Stopping.")
            break

        # Aggressive checkpoints based on top performer pace
        if epoch == 5 and best_accuracy < 0.90:
            print(f"Only {best_accuracy:.4f} at epoch 5. No chance. Stopping.")
            break
        if epoch == 10 and best_accuracy < 0.95:
            print(f"Only {best_accuracy:.4f} at epoch 10. No chance. Stopping.")
            break
        if epoch == 20 and best_accuracy < 0.97:
            print(f"Only {best_accuracy:.4f} at epoch 20. No chance. Stopping.")
            break
        if epoch == 50 and best_accuracy < 0.98:
            print(f"Only {best_accuracy:.4f} at epoch 50. Won't hit 99.5%. Stopping.")
            break

    # Generate and log decision boundary visualization
    boundary_img = viz_decision_boundary(model, device)
    if use_wandb:
        wandb.log(
            {
                "final_accuracy": accuracy,
                "best_accuracy": best_accuracy,
                "decision_boundary": wandb.Image(
                    boundary_img,
                    caption=f"{hidden_layers} - {best_accuracy:.4f} acc",
                ),
            }
        )
        wandb.finish()
    else:
        # Save locally if not using wandb
        boundary_img.save(f"decision_boundary_{total_neurons}neurons.png")

    print(
        f"Architecture: {hidden_layers} ({total_neurons} neurons, {total_params} params)"
    )
    print(f"Best accuracy: {best_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Baarle Hertog optimization")
    parser.add_argument("--sweep", action="store_true", help="Run as wandb sweep agent")
    parser.add_argument("--sweep-id", type=str, help="Existing sweep ID to join")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of runs for sweep agent"
    )

    # Single run parameters
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--layer-width", type=int, default=24)
    parser.add_argument("--taper-ratio", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument(
        "--activation", type=str, default="gelu", choices=["gelu", "relu", "silu"]
    )
    parser.add_argument("--use-layer-norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="use_layer_norm", action="store_false")
    parser.add_argument("--use-skip-connections", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-connections", dest="use_skip_connections", action="store_false"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.sweep:
        if args.sweep_id:
            # Join existing sweep
            wandb.agent(args.sweep_id, function=train, count=args.count)
        else:
            # Create new sweep
            sweep_config = {
                "method": "bayes",
                "metric": {"name": "best_accuracy", "goal": "maximize"},
                "parameters": {
                    "num_layers": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 10]},
                    "layer_width": {"distribution": "int_uniform", "min": 8, "max": 64},
                    "taper_ratio": {"distribution": "uniform", "min": 0.5, "max": 1.0},
                    "learning_rate": {
                        "distribution": "log_uniform_values",
                        "min": 0.001,
                        "max": 0.1,
                    },
                    "batch_size": {"values": [5000, 10000, 20000]},
                    "num_epochs": {"values": [100, 200, 300]},
                    "use_layer_norm": {"values": [True, False]},
                    "use_skip_connections": {"values": [True, False]},
                    "activation": {"values": ["gelu", "relu", "silu"]},
                    "seed": {"distribution": "int_uniform", "min": 0, "max": 1000},
                },
            }
            sweep_id = wandb.sweep(
                sweep_config, project="baarle-hertog-optimization"
            )
            print(f"Created sweep: {sweep_id}")
            wandb.agent(sweep_id, function=train, count=args.count)
    else:
        # Single run with command-line args
        config = {
            "num_layers": args.num_layers,
            "layer_width": args.layer_width,
            "taper_ratio": args.taper_ratio,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "use_layer_norm": args.use_layer_norm,
            "use_skip_connections": args.use_skip_connections,
            "activation": args.activation,
            "seed": args.seed,
        }
        train(config)


if __name__ == "__main__":
    main()
