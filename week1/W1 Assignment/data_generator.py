import numpy as np
import matplotlib.pyplot as plt

N=200
MEAN = 0
STD = 0.2
SEED=42

# Generate data
np.random.seed(SEED)
x = np.linspace(-1, 3, N)
u = np.random.normal(MEAN, STD, N)
y_true = -x**2 + 2*x
y = -x**2 + 2*x + u

# Export
data = np.column_stack((x, y))
np.savetxt(
    "data.csv",
    data,
    delimiter=",",
    header="x,y",
    comments=""
)

# Plot
plt.plot(x,y_true, label="true y")
plt.scatter(x,y, label="observations", alpha=0.5)
plt.title(f"Data with error of mean={MEAN}, std={STD}")
plt.savefig("data_plot.png")
plt.show()