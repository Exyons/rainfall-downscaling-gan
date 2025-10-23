import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_bias_using_QQ_plot(pred_images, true_images):
    # Flatten all pixels across all days (excluding NaNs or negatives if any)
    downscaled_flat = pred_images.flatten()
    actual_flat = true_images.flatten()

    # Optionally, remove missing or invalid rainfall values (e.g., negative or NaN)
    mask = np.isfinite(actual_flat) & np.isfinite(downscaled_flat)
    downscaled_flat = downscaled_flat[mask]
    actual_flat = actual_flat[mask]

    # (Optional) Remove zero rainfall values if you only want to focus on rainy days
    # nonzero_mask = (actual_flat > 0) & (downscaled_flat > 0)
    # downscaled_flat = downscaled_flat[nonzero_mask]
    # actual_flat = actual_flat[nonzero_mask]

    # --- Q-Q Plot ---
    plt.figure(figsize=(6, 6))
    stats.probplot(downscaled_flat, dist="norm", plot=plt)  # sanity check (normal)
    plt.close()

    # Use Q-Q plot comparing two datasets directly
    quantiles = np.linspace(0, 100, 1000)
    q_actual = np.percentile(actual_flat, quantiles)
    q_downscaled = np.percentile(downscaled_flat, quantiles)

    plt.figure(figsize=(7, 7))
    plt.plot(q_actual, q_downscaled, 'o', markersize=2, alpha=0.6, label='QQ points')
    plt.plot([0, max(q_actual.max(), q_downscaled.max())],
            [0, max(q_actual.max(), q_downscaled.max())],
            'r--', label='1:1 line')

    plt.xlabel('Actual Rainfall Quantiles (mm/day)')
    plt.ylabel('Downscaled Rainfall Quantiles (mm/day)')
    plt.title('Q–Q Plot: Downscaled vs Actual Rainfall')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Bias interpretation ---
    # You can also compute mean bias, RMSE, etc. for reference:
    mean_bias = np.mean(downscaled_flat - actual_flat)
    rmse = np.sqrt(np.mean((downscaled_flat - actual_flat)**2))
    print(f"Mean Bias: {mean_bias:.3f}")
    print(f"RMSE: {rmse:.3f}")