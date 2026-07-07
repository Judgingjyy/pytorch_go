import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, depth):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        layers = []

        input_dim = 2
        hidden_dim = 32

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_xor_data(n_samples=400, seed=0):
    generator = torch.Generator().manual_seed(seed)
    X = torch.rand((n_samples, 2), generator=generator) * 2 - 1
    y = ((X[:, 0] * X[:, 1]) > 0).long()
    return X, y


def train(model, X, y, epochs=500):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    losses = []

    for epoch in range(epochs):
        out = model(X)
        loss = criterion(out, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print("loss:", loss.item())

    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        accuracy = (pred == y).float().mean().item()
    print("accuracy:", accuracy)
    return losses, accuracy


def plot_results(results, output_path="depth_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for depth, metrics in results.items():
        axes[0].plot(metrics["losses"], label=f"depth={depth}")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    depths = list(results.keys())
    accuracies = [results[depth]["accuracy"] for depth in depths]
    axes[1].bar([str(depth) for depth in depths], accuracies, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("Final Accuracy")
    axes[1].set_xlabel("Network Depth")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print("saved plot:", output_path)


if __name__ == "__main__":
    torch.manual_seed(0)
    X, y = make_xor_data()
    results = {}

    for depth in [2, 4, 8]:
        print("==== depth:", depth)
        model = MLP(depth)
        losses, accuracy = train(model, X, y)
        results[depth] = {"losses": losses, "accuracy": accuracy}

    plot_results(results)
