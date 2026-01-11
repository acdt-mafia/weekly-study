## 명령어 모음
## 기본값은 learning_rate = 0.001, batch_size = None, epochs = 100
## 기본 실행 (BGD)
# python lab_linear-regression-from-scratch.py

## SGD
# python lab_linear-regression-from-scratch.py --batch_size 1

## mini-batch
# python lab_linear-regression-from-scratch.py --batch_size 32

## learning rate 변경
# python lab_linear-regression-from-scratch.py --learning_rate 0.01

## epoch 늘리기
# python lab_linear-regression-from-scratch.py --epoch 300


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
        self.loss_history = []  #추가

    def fit(self, X, y):
        """
        # To-Do
        
        - To regress y = a + b x, implement gradient descent by applying following equations:
            1. a = a - lr*(∂MSE/∂a)
            2. b = b - lr*(∂MSE/∂b)
        
        - Implement BGD: 전체 데이터 n개를 한 번에 써서 1회 업데이트, 
                    SGD: 데이터 1개로 업데이트를 n번(혹은 epoch동안 n회)수행, 
                    mini-batch GD: 예를 들어 32개씩 끊어서 업데이트
        - Print out computed \hat{a}, \hat{b}

        할 일: 
        현재 a,b로 예측값 y_hat계산 = a + bx
        MSE 계산 + 그래디언트 계산
        a,b 업데이트
        """
        X = np.array(X).reshape(-1)  #1차원 배열로 만들기
        y = np.array(y).reshape(-1)
        n = len(X)

        # 1. a,b를 시작값으로 초기화
        a = 0 # Y INTERCEPT
        b = 0 # COEFFICIENT

        # 1-2. batch_size 가 none 이면 BGD로 처리하기 
        if self.batch_size is None: 
          batch_size = n  ##BGD로 처리

        #2. epoch 반복문 안에서 데이터 섞기. 중요한 이유: 크기 순으로 되어있으면 편향되어서 업데이트 됨. 
        for epoch in range(self.epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            batch_losses = []
            #iterate
            for start in range(0, n, batch_size):
              end = start + batch_size
              Xb = X_shuffled[start:end]
              yb = y_shuffled[start:end]

              m=len(Xb)  #batch size

              #prediction
              y_hat = a + b*Xb  
              error = yb - y_hat

              #MSE
              mse = np.mean(error **2)
              batch_losses.append(mse)

              # gradients of MSE
              d_a = -(2 / m) * np.sum(error)   #m = 배치크기 
              d_b = -(2 / m) * np.sum(Xb * error)

              lr = self.learning_rate
              #업데이트 -> 경사하강
              a = a - lr * d_a
              b = b - lr * d_b
          
            epoch_loss = float(np.mean(batch_losses))
            self.loss_history.append(epoch_loss)

            # optional: print occasionally
            if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                mode = "BGD" if batch_size == n else ("SGD" if batch_size == 1 else f"mini-batch({batch_size})")
                print(f"[{mode}] epoch {epoch+1:4d}/{self.epochs}  loss={epoch_loss:.6f}  a={a:.6f}  b={b:.6f}")

        # save learned params
        self.a = a
        self.b = b

        print("\nFinal parameters:")
        print(f"a_hat = {self.a:.6f}")
        print(f"b_hat = {self.b:.6f}")

        return self

            
    
    def predict(self, X, y=None):
        '''
        X가 주어지면 y_hat을 반환
        y = None : y가 있을수도 있고 없을수도 있고~
        fit으로 학습된 a,b 사용해서 정답 y랑 비교해서 MSE 계산해준다. 
        '''
        X = np.asarray(X).reshape(-1)
        ##선형회귀 예측 식
        y_hat = self.a + self.b * X

        if y is None:
            return y_hat   ## y 안주면 예측만 하고 끝

        y = np.asarray(y).reshape(-1)
        mse = float(np.mean((y - y_hat) ** 2))
        return y_hat, mse  ## 정답 y 주면 평가까지. 


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