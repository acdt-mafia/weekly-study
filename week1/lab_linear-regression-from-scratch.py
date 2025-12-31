import argparse
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch', default=100, type=int)

    args = parser.parse_args()

    # Main code
    set_seed(42)

    x = np.linspace(-2, 2, 200)
    u = np.random.normal(0, 0.1, 200)
    y = 0.8 + 0.7*x + u

    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epoch

    a = 0 # Y INTERCEPT
    b = 0 # COEFFICIENT

    for epoch in range(epochs):
        pass
        """
        # To-Do
        
        - Implement gradient descent by applying following equations:
            1. a = a - lr*(∂MSE/∂a)
            2. b = b - lr*(∂MSE/∂b)
        
        - Implement BGD, SGD, mini-batch GD
        """

if __name__ == "__main__":
    main()