import argparse #실험 조건 더 쉽게 바꾸게 해줌
import random
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot  



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):  #multi layer perceptron, nn.module 상속
    def __init__(self, n_nodes=3, n_hidden=1, activation='relu'):  #init으로 객체에 들어갈 내용 넣는듯
        super().__init__() #init생성자, nn.module 초기화
        layers = [] # 빈 레이어 리스트
        
        # Input layer
        layers.append(nn.Linear(1, n_nodes)) # 가중치 행렬, n_nodes개 숫자가 1개의 입력에 대해 출력
        layers.append(self.get_activation(activation)) 
        
        # Hidden layers
        for _ in range(n_hidden-1):
            layers.append(nn.Linear(n_nodes, n_nodes)) # hidden layer에서 같은 크기로 변환
            layers.append(self.get_activation(activation)) #활성화 함수 적용
        
        # Output layer
        layers.append(nn.Linear(n_nodes, 1))
        
        self.model = nn.Sequential(*layers)  # 자동으로 linear1, activation, linear2 실행 (가중치 bias 적용된 1차함수와 활성화함수)
    
    def get_activation(self, name):
        if name == 'sigmoid': return nn.Sigmoid()
        elif name == 'relu': return nn.ReLU()
        elif name == 'tanh': return nn.Tanh()
        else: return nn.Sigmoid()
    
    def forward(self, x):
        return self.model(x)

def visualize_forward(model, x_sample):  #hyperparameter 쉽게 바꿀 수 ㅇ있도록. hyper parameter는 사람이 정하는거? learning rate같은거
    """Visualize forward pass x -> y_pred"""
    model.eval()  #평가모드
    with torch.no_grad(): # 기울기 ㄴㄴㄴ
        activations = {}
        def hook_fn(name): #hook 함수 생성 -> 모든 레이어 출력값 확인 위해서?
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        handles = []  # 모든 레이어에 hook 적용
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Sigmoid)):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        y_pred = model(x_sample)   # forward 실행
        for h in handles:
            h.remove()  #hook 정리
        
        print("\n=== Forward Pass (x=0.25) ===")
        print(f"Input x: {x_sample.item():.3f}")
        for name, act in activations.items():
            if 'linear' in name or 'sigmoid' in name:
                print(f"{name}: {act.item():.4f}")
        print(f"Final prediction: {y_pred.item():.4f}")
    
    return activations   #activation 딕셔너리 완성

def main():
    parser = argparse.ArgumentParser()  # 조건 쉽게 바꿀 수 있도록, 학습 횟수 줄임
    parser.add_argument('--n_nodes', type=int, default=3)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--n_hidden', type=int, default=1)
    parser.add_argument('--activation', default='sigmoid', choices=['sigmoid', 'relu', 'tanh'])
    args = parser.parse_args()

    set_seed(42)
    
    # Load data
    try:
        df = pd.read_csv("data_week1.csv")
    except FileNotFoundError:
        print("data_week1.csv not found! Generating sample data...")
        np.random.seed(42)
        x = np.sort(np.random.uniform(-3, 3, 100))
        y = 0.5 * x**2 + np.random.normal(0, 0.5, 100)
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv("data_week1.csv", index=False)
    
    x = torch.tensor(df["x"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df["y"].values, dtype=torch.float32).view(-1, 1)
    
    # Model
    model = MLP(
        n_nodes=args.n_nodes,
        n_hidden=args.n_hidden,
        activation=args.activation
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #optimizer는 편향이랑 가중치 업데이트 알고리즘
    loss_fn = nn.MSELoss()
    
    # Training
    losses = []
    model.train()
    for epoch in range(args.epoch):
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epoch}, Loss: {loss.item():.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
    
    # Forward pass visualization
    x_sample = torch.tensor([[0.25]], dtype=torch.float32)
    visualize_forward(model, x_sample)
    
    # Manual computation verification
    W1 = model.model[0].weight.detach().numpy()
    b1 = model.model[0].bias.detach().numpy()
    W2 = model.model[-1].weight.detach().numpy()
    b2 = model.model[-1].bias.detach().numpy()
    
    print("\n=== Learned Weights ===")
    print("W1:", W1.flatten())
    print("b1:", b1)
    print("W2:", W2.flatten())
    print("b2:", b2)
    
    # Manual forward pass
    x_np = x_sample.numpy()
    z1 = x_np @ W1.T + b1
    if args.activation == 'sigmoid':
        h = 1 / (1 + np.exp(-z1))
    elif args.activation == 'relu':
        h = np.maximum(0, z1)
    else:  # tanh
        h = np.tanh(z1)
    y_manual = h @ W2.T + b2
    
    print(f"Manual calc: {y_manual[0,0]:.4f} (Model: {model(x_sample).item():.4f})")
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, label="Data", alpha=0.6)
    plt.plot(x, y_pred, "r-", label="Prediction", linewidth=2)
    plt.legend()
    plt.title(f"Prediction (Nodes={args.n_nodes}, Hidden={args.n_hidden})")
    
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    
    plt.subplot(1, 3, 3)
    try:
        dot = make_dot(model(x_sample), params=dict(model.named_parameters()))
        plt.title("Compute Graph")
        plt.axis('off')
        dot.format = 'png'
        dot.render('model_graph', cleanup=True)
        print("Graph saved: model_graph.png")
    except:
        plt.text(0.5, 0.5, "Need torchviz\ngraphviz", ha='center')
        plt.title("Compute Graph")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal MSE: {nn.MSELoss()(y_pred, y):.6f}")

if __name__ == "__main__":
    main()
