import argparse
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

class Model:
    # 직접 변경 가능한 클래스 속성!
    learning_rate = 0.0005
    batch_size = 32
    epochs = 500
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        n = len(X)  #데이터 개수
        bs = Model.batch_size if Model.batch_size else n
        
        a = 0.0
        b = 0.0
        
        for epoch in range(Model.epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n)
            
            num_batches = 0
            for i in range(0, n, bs):
                batch_indices = indices[i:min(i+bs, n)]  # 안전한 인덱싱
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                y_pred = a + b * X_batch
                errors = y_pred - y_batch
                da = np.mean(errors) * (-1)
                db = np.mean(errors * (-X_batch))
                
                a -= Model.learning_rate * da
                b -= Model.learning_rate * db
                
                epoch_loss += np.mean(errors**2)
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{Model.epochs}, Loss: {epoch_loss/num_batches:.6f}, a: {a:.4f}, b: {b:.4f}")
        
        print(f"\n최종 학습 결과:")
        print(f"a: {a:.4f} (실제: 0.8)")
        print(f"b: {b:.4f} (실제: 0.7)")
        print(f"MSE: {np.mean((a + b*X - y)**2):.6f}")
        
        self.a = a
        self.b = b
    
    def predict(self, X):
        if not hasattr(self, 'a'):
            raise ValueError("먼저 fit()하세요.")
        return self.a + self.b * X

# 데이터 생성 (전역)
def generate_data():
    set_seed(42)
    x = np.linspace(-2, 2, 200)
    u = np.random.normal(0, 0.1, 200)
    y = 0.8 + 0.7*x + u
    return x, y

if __name__ == "__main__":
    Model.learning_rate = 0.0005
    Model.batch_size = 64
    Model.epochs = 500  # 원하는 값 자유 변경!
    
    # 데이터
    x, y = generate_data()
    
    # 학습
    model = Model()
    model.fit(x, y)
    
    # 예측 테스트
    test_x = np.array([-1, 0, 1])
    pred_y = model.predict(test_x)
    print(f"\nx: {test_x}")
    print(f"예측: {pred_y}")
    print(f"실제: {0.8 + 0.7*test_x}")
