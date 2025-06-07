import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv("rmse_log.csv")

# Plot with thinner lines
plt.figure(figsize=(8, 6))
plt.plot(df['Epoch'], df['TrainRMSE'], 'o-', color='blue', label='Train RMSE', linewidth=0.5, markersize=2)
plt.plot(df['Epoch'], df['TestRMSE'], 's-', color='red', label='Test RMSE', linewidth=0.5, markersize=2)

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training and Testing RMSE over Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
