import argparse
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

class Model:
    def __init__(self, learning_rate, batch_size, epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        """
        # To-Do
        
        - To regress y = a + b x, implement gradient descent by applying following equations:
            1. a = a - lr*(∂MSE/∂a)
            2. b = b - lr*(∂MSE/∂b)
        
        - Implement BGD, SGD, mini-batch GD
        - Print out computed \hat{a}, \hat{b}
        """
        a = 0 # Y INTERCEPT
        b = 0 # COEFFICIENT
        #BDG code
        for epoch in range(self.epochs):
            y_pred = a+b*X
            MSE = ((y_pred-y)**2).mean()
            if MSE < 0.001:
                break
            a = a - self.learning_rate*(y_pred-y).mean()
            b = b - self.learning_rate*((y_pred-y)*X).mean()
            if self.epochs % 5 == 0:
                print('epoch =', epoch, 'a =', a, 'b =', b)
        
        #SGD
        for epoch in range(self.epochs):
            X_sgd = np.random.choice(X)
            y_sgd = a + b * X_sgd
            a = a - self.learning_rate*(y_sgd-y).mean()
            b = b - self.learning_rate*((y_sgd-y)*X).mean()
            error = ((y - y_sgd)).mean()
            if self.epochs % 5 == 0:
                print('epoch =', epoch, 'a =', a, 'b =', b)
            if error < 0.001:
                break
        
        #mini-batch GD
        for epoch in range(self.epochs):
            self.batch_size = 20 
            batch_number = 200 / 20
            
            for iteration in range(int(batch_number)):
                idx = np.random.choice(len(X), self.batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
                
                y_batch = a + b * X_batch

                a = a - self.learning_rate*(y_batch-y).mean()
                b = b - self.learning_rate*((y_batch-y)*X).mean()

                error = ((y-y_batch)**2).mean()

                print('iteration', iteration, 'a=', a, 'b=', b)

                if error < 0.001:
                    break

            
    
    
    def predict(self, X, y):
        y_pred = self.a + self.b * X
        return y_pred


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

    ## Data generation
    x = np.linspace(-2, 2, 200)
    u = np.random.normal(0, 0.1, 200)
    y = 0.8 + 0.7*x + u

    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epoch

    ## Model
    model = Model(learning_rate=lr,
                  batch_size=batch_size, 
                  epochs=epochs)

    model.fit(x,y)


if __name__ == "__main__":
    main()