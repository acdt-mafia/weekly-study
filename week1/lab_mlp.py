import argparse
from html import parser
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

"""
# Usage
python3 ./lab_mlp.py --n_nodes 3 --learning_rate 0.001 --activation sigmoid--epoch 100
(sigmoid, relu, tanh 중 선택 가능)

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
    parser.add_argument('--n_nodes', default=3, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--activation', default="sigmoid", type=str, choices=["sigmoid", "relu", "tanh"])

    args = parser.parse_args()

    act_map = {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh()
    }
    activation = act_map[args.activation]

    # Main code
    set_seed(42)

    df = pd.read_csv("data_week1.csv")
    x = torch.tensor(df["x"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).view(-1, 1)


    ## 은닉층 레이어 만들기
    layers = []
    in_dim = 1

    ## hidden layers
    for _ in range(args.n_layers):
        layers.append(nn.Linear(in_dim, args.n_nodes))
        layers.append(activation)
        in_dim = args.n_nodes

    ## output layer
    layers.append(nn.Linear(in_dim, 1))
    
    ## model
    model = nn.Sequential(*layers)

    # # Model
    # model = nn.Sequential(
    #     nn.Linear(1, args.n_nodes),
    #     activation,
    #     nn.Linear(args.n_nodes, 1)

    # )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    loss_history = []


    # Training
    model.train()

    for epoch in range(args.epoch):
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    

    # Evaluation
    model.eval()

    with torch.no_grad():
        y_pred = model(x)

    y_np = y_pred.squeeze().numpy()

    if args.n_layers == 1 and args.activation == "sigmoid":
        # manual computation 수행
        W1 = model[0].weight.detach().numpy()
        b1 = model[0].bias.detach().numpy()

        W2 = model[2].weight.detach().numpy()
        b2 = model[2].bias.detach().numpy()

        print("W1:\n",W1)
        print("b1:",b1)
        print("W2:",W2)
        print("b2:",b2)

        x_np = df["x"].values.reshape(-1, 1)
        z1 = x_np @ W1.T + b1    ## 은닉층으로 들어가기 직전의 계산 결과 = 가중치가 적용된 선형 조합 값.
        h = 1 / (1 + np.exp(-z1))  ## z1에 activation(비선형 함수) 적용한 값. 신경망이 곡선을 만들 수 있게 해주는 값. z1은 아무 값이나 될 수 있는데 시그모이드로 그걸 0~1사이로 눌러서 꺾어주는 역햘!이라고 함. 
        y_hat_manual = h @ W2.T + b2

    else:
        print("Manual computation is shown only for n_layers=1 and sigmoid (default).")
        # 아래 plot 블록에서 에러가 안 나도록 기본값 세팅 (plot 관련 처리)
        x_np = df["x"].values.reshape(-1, 1)
        z1 = np.zeros((len(x_np), 1), dtype=float)
        h = np.zeros((len(x_np), 1), dtype=float)
        y_hat_manual = np.full((len(x_np), 1), np.nan, dtype=float)


    # Example computatoin for a single datapoint
    print("\nExample: model(0.25) =", model(torch.tensor([[0.25]], dtype=torch.float32)).item())


    # =========================
    # Plot (ALL-IN-ONE Figure)
    # =========================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Loss curve
    axes[0, 0].plot(loss_history)
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MSE loss")
    axes[0, 0].set_title("Training Loss Curve")

    # (2) z1 histogram
    axes[0, 1].hist(z1.flatten(), bins=30, alpha=0.8)
    axes[0, 1].set_title("Distribution of z1 (pre-activation values)")
    axes[0, 1].set_xlabel("z1")
    axes[0, 1].set_ylabel("count")

    # (3) h histogram
    axes[1, 0].hist(h.flatten(), bins=30, alpha=0.8)
    axes[1, 0].set_title("Distribution of h = sigmoid(z1) (post-activation)")
    axes[1, 0].set_xlabel("h")
    axes[1, 0].set_ylabel("count")

    # (4) Predictions vs Observations
    axes[1, 1].scatter(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="Observations", alpha=0.6)
    axes[1, 1].plot(x.detach().cpu().numpy(), y_np, label="Predictions", color="red")

    # manual predictions only meaningful for default (n_layers=1, sigmoid)
    if args.n_layers == 1 and args.activation == "sigmoid":
        axes[1, 1].plot(x.detach().cpu().numpy(), y_hat_manual, label="Predictions_manual", color="green", ls='--', alpha=0.75)

    axes[1, 1].legend()
    axes[1, 1].set_title("Observations vs Predictions")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
