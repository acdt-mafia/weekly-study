import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

"""
# Usage
python3 ./lab_mlp.py --n_nodes 3 --learning_rate 0.001 --epoch 100


# To-Do

- Visualize the computations performed from x to y_pred(\hat{y}), under the default model architecture.
- Tweak hyperparameters such as
    - # of nodes
    - learning rate
    - # of hidden layers
    - type of activation function
"""

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epoch', default=100, type=int)

    args = parser.parse_args()

    # Main code
    set_seed(42)

    df = pd.read_csv("data_week1.csv")
    x = torch.tensor(df["x"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).view(-1, 1)

    # Model
    model = nn.Sequential(
        nn.Linear(1, args.n_nodes),
        nn.Sigmoid(),
        nn.Linear(args.n_nodes, 1)
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()


    # Training
    model.train()

    for epoch in range(args.epoch):
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    # Evaluation
    model.eval()

    with torch.no_grad():
        y_pred = model(x)

    y_np = y_pred.squeeze().numpy()


    # Manual computation
    W1 = model[0].weight.detach().numpy()
    b1 = model[0].bias.detach().numpy()

    W2 = model[2].weight.detach().numpy()
    b2 = model[2].bias.detach().numpy()

    print("W1:\n",W1)
    print("b1:",b1)
    print("W2:",W2)
    print("b2:",b2)

    x_np = df["x"].values.reshape(-1, 1)
    z1 = x_np @ W1.T + b1
    h = 1 / (1 + np.exp(-z1))
    y_hat_manual = h @ W2.T + b2


    # Example computatoin for a single datapoint
    print("\nExample: model(0.25) =", model(torch.tensor([[0.25]], dtype=torch.float32)).item())


    # Plot
    plt.scatter(x, y, label="Observations", alpha=0.6)
    plt.plot(x, y_np, label="Predictions", color="red")
    plt.plot(x, y_hat_manual, label="Predictions_manual", color="green", ls='--', alpha=0.75)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()