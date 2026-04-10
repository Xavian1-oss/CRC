import numpy as np
import matplotlib.pyplot as plt


def main():
    # Validation ratios (as used in CRC val subset)
    val_ratios = np.array([0.05, 0.10, 0.20])
    val_labels = ["5%", "10%", "20%"]

    # ========================
    # Weather + TimesNet + pl=96
    # ========================
    # From logs/val_sensitivity_20260407_210341/master.log
    weather_baseline_mse = np.array([0.1685, 0.1685, 0.1685])
    weather_crc_mse = np.array([0.1617, 0.1611, 0.1611])
    weather_ndr = np.array([74.2, 77.7, 78.6])

    # ========================
    # Electricity + TimesNet + pl=96
    # ========================
    # From logs/val_sensitivity_20260407_210341/master.log
    elec_baseline_mse = np.array([0.2512, 0.2512, 0.2512])
    elec_crc_mse = np.array([0.1791, 0.1791, 0.1784])
    elec_ndr = np.array([97.3, 97.6, 97.5])

    plt.rcParams.update({
        "font.size": 12,
        # 横向排布两个子图，适当加宽
        "figure.figsize": (10.0, 4.0),
    })

    # 横向两个子图
    fig, axes = plt.subplots(1, 2, sharey=False, sharex=True)

    def plot_dataset(ax, title, baseline_mse, crc_mse, ndr):
        # Left axis: MSE
        color_baseline = "tab:blue"
        color_crc = "tab:orange"
        ax.plot(val_ratios, baseline_mse, marker="o", color=color_baseline,
                label="Baseline MSE")
        ax.plot(val_ratios, crc_mse, marker="s", color=color_crc,
                label="CRC MSE")
        ax.set_ylabel("Test MSE")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Right axis: NDR
        ax2 = ax.twinx()
        color_ndr = "tab:green"
        ax2.plot(val_ratios, ndr, marker="D", linestyle="--", color=color_ndr,
                 label="NDR")
        ax2.set_ylabel("NDR (%)")

        # Merge legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2,
                  loc="best", frameon=False)

    # Top subplot: Weather
    plot_dataset(
        axes[0],
        "Weather (TimesNet, pred_len=96)",
        weather_baseline_mse,
        weather_crc_mse,
        weather_ndr,
    )

    # Right subplot: Electricity
    plot_dataset(
        axes[1],
        "Electricity (TimesNet, pred_len=96)",
        elec_baseline_mse,
        elec_crc_mse,
        elec_ndr,
    )

    for ax in axes:
        ax.set_xlabel("Validation ratio used by CRC")
        ax.set_xticks(val_ratios)
        ax.set_xticklabels(val_labels)

    plt.tight_layout()
    plt.savefig("crc_validation_sensitivity.png", dpi=300, bbox_inches="tight")
    # Uncomment to display interactively
    # plt.show()


if __name__ == "__main__":
    main()
