import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

log_path = Path("./logs/coverage_sweep_20260408_225314/master.log")

# 手动同步一下这次代码里的映射（q_val -> quantile_q）
def map_q_to_quantile(q):
    if q <= 1:
        return 0.10
    elif q == 2:
        return 0.30
    elif q == 3:
        return 0.50
    elif q == 4:
        return 0.70
    else:
        return 0.90

q_vals, quantiles, mses_crc, ndrs = [], [], [], []

current_q = None
with log_path.open("r", encoding="utf-8") as f:
    for line in f:
        # 1) 找到 q_val
        m_q = re.search(r"CRC Coverage Sweep: ds=Electricity,\s*q_val=(\d+)", line)
        if m_q:
            current_q = int(m_q.group(1))
            continue

        # 2) 在对应 block 里抓 CRC 纠偏后的 MSE 和 NDR
        if current_q is not None:
            m_mse = re.search(r"CRC 纠偏后 -> MSE:\s*([0-9.]+), MAE:", line)
            if m_mse:
                mse_crc = float(m_mse.group(1))
                continue

            m_ndr = re.search(r"NDR \(非恶化率\):\s*([0-9.]+)%", line)
            if m_ndr:
                ndr = float(m_ndr.group(1))
                # 记录一次完整样本
                q_vals.append(current_q)
                quantiles.append(map_q_to_quantile(current_q))
                mses_crc.append(mse_crc)
                ndrs.append(ndr)
                current_q = None  # 本次 block 结束

q_vals = np.array(q_vals)
quantiles = np.array(quantiles)
mses_crc = np.array(mses_crc)
ndrs = np.array(ndrs)

print("q_vals:     ", q_vals)
print("quantiles:  ", quantiles)
print("CRC MSE:    ", mses_crc)
print("NDR:        ", ndrs)

# 图 1：以 quantile_q 为横轴的两条曲线
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(quantiles, mses_crc, "-o")
plt.xlabel("safety quantile q")
plt.ylabel("CRC MSE")
plt.title("CRC MSE vs safety quantile")

plt.subplot(1,2,2)
plt.plot(quantiles, ndrs, "-o", color="tab:orange")
plt.xlabel("safety quantile q")
plt.ylabel("NDR (%)")
plt.title("NDR vs safety quantile")
plt.tight_layout()
# 图片直接保存在仓库根目录
plt.savefig("./electricity_crc_tau_sweep_curves_q.png", dpi=200)

# 图 2：直接画 safety–MSE 曲线（每个点一个 τ）
plt.figure(figsize=(4,4))
plt.plot(mses_crc, ndrs, "-o")
for i, q in enumerate(q_vals):
    plt.text(mses_crc[i]+0.0003, ndrs[i], f"q={q}", fontsize=8)
plt.gca().invert_xaxis()  # 左好右坏，更直观一点（可选）
plt.xlabel("CRC MSE (lower=better)")
plt.ylabel("NDR (%)")
plt.title("Safety–accuracy tradeoff across τ")
plt.tight_layout()
# 图片直接保存在仓库根目录
plt.savefig("./electricity_crc_tau_sweep_safety_mse.png", dpi=200)

plt.show()