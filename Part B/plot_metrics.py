import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")
plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
plt.plot(df["epoch"], df["test_rmse"], label="Test RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("RMSE vs Epochs")
plt.legend()
plt.grid()
plt.show()
