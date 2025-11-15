import os
import torch
import numpy as np
import warnings
from copy import deepcopy

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.CRC import (
    PretrainableEncoderMTL, HybridTrainer, mtl_pretrain_trainer,
    build_corr_graph, build_physics_priors_ms
)
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.dtw_metric import accelerated_dtw
from utils.tools import visual
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr
import seaborn as sns


warnings.filterwarnings('ignore')

# === 时间因果：TCL 第一层权重可视化（热力图 + 平均重要性曲线） ===
def visualize_time_causality_matrix(stcl_core, save_dir=None, tag="val"):
    W = stcl_core.b[0].l.weight.detach().cpu().numpy()  # (d, P)
    Wn = W / (np.abs(W).max() + 1e-8)

    plt.figure(figsize=(9, 3))
    plt.imshow(Wn, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Normalized weight')
    plt.xlabel('Historical step (0..P-1)'); plt.ylabel('Hidden unit')
    plt.title('Time Causality Matrix (TCL layer-1)')
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, f"time_causality_{tag}.png"), dpi=200)
    plt.close()

    imp = np.abs(W).mean(axis=0); imp /= (imp.max() + 1e-8)
    plt.figure(figsize=(8, 2.6))
    plt.plot(imp, '-o', ms=3); plt.grid(True, alpha=0.3)
    plt.xlabel('Historical step'); plt.ylabel('Normalized importance')
    plt.title('Temporal Importance Curve')
    plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, f"time_importance_{tag}.png"), dpi=200)
    plt.close()

def visualize_spatial_causal_heatmap(S, node_names=None, save_dir=None, tag="val"):
    """
    使用热力图可视化空间因果矩阵 S。
    基于您的代码，S[i, j] 对应 j -> i 的因果强度。
    """
    N = S.shape[0]
    if node_names is None:
        node_names = [f"N{i}" for i in range(N)]
    
    figsize = (max(8, N // 3), max(6, N // 3))
    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(
        S, 
        xticklabels=node_names, 
        yticklabels=node_names,
        cmap='RdBu_r',       # 推荐使用发散色图
        center=0,            # 将 0 设置为颜色的中心
        annot=False,         # 如果 N 较小 (如 N < 30)，可设为 True 来显示数字
        fmt=".2f"            # (如果 annot=True) 数字的格式
    )
    
    plt.xlabel("Source Node (j)   [j → i]", fontsize=12)
    plt.ylabel("Target Node (i)   [j → i]", fontsize=12)
    
    # 旋转 x 轴标签，防止重叠
    plt.xticks(rotation=90) 
    plt.yticks(rotation=0)
    
    plt.title(f"Spatial Causality Matrix (Heatmap: S[i, j] = strength of j → i)")
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"spatial_causal_heatmap_{tag}.png")
        print(f"Saving heatmap to {save_path}")
        plt.savefig(save_path, dpi=200)
    plt.close()


# === 先验一致性与滞后一致性的一些指标（简版） ===
def precision_at_k_vs_prior(S, A_prior, k_ratio=0.1):
    N = S.shape[0]; M = N*(N-1); k = max(1, int(k_ratio*M))
    mask = ~np.eye(N, dtype=bool)
    scores = np.abs(S[mask]).flatten()
    prior  = A_prior[mask].flatten().astype(np.int32)
    top_idx = np.argsort(-scores)[:k]
    return float(prior[top_idx].mean())

def spearman_vs_prior(S, A_prior):
    mask = ~np.eye(S.shape[0], dtype=bool)
    s = np.abs(S[mask]).flatten()
    p = A_prior[mask].flatten()
    rho = spearmanr(s, p).correlation
    return float(rho) if rho==rho else 0.0

def quick_val_series_from_windows(W_val):
    """
    用滑窗近似拼一条每节点的验证序列（简易、稳妥、不依赖 dataset 内部）。
    思路：取每个窗口的“最后一个时间点”拼接成长度 B 的序列。
    返回 arr:(B,N)
    """
    return W_val[:, -1, :]  # (B, N)，每行看作一个连续采样点的“快照”


class Exp_CRC(Exp_Long_Term_Forecast):
    """
    Causal Residual Corrector (CRC) 实验流程类（两阶段：基座→纠偏）。
    继承自 Exp_Long_Term_Forecast，但不在 __init__ 里构建 self.model。

    做法A改动要点：
      1) _get_predictions() 评测时也传入时间标记 (x_mark / y_mark)，与训练保持一致；
      2) Hybrid 纠偏器的 N 代表 “输出/目标通道数”，使用 y_tr_win.shape[-1]（MS=1，M=enc_in）；
      3) test() 阶段可选 inverse，按 LTF 的方式在通道维做 tile→inverse→再裁回；
      4) DTW 的计算采用 flatten(H*C) 的定义，便于与 LTF/基线对齐。
    """
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.baseline_model = None
        self.hybrid_corrector = None

    def _build_model(self):
        # CRC 不直接用 self.model
        return None

    def _get_predictions(self, model, data_loader):
        """
        用基线模型在 loader 上做推理，返回：
          preds  (B, H, N_sel)
          trues  (B, H, N_sel)
          inputs (B, P, N_all)
        说明：
          - MS: N_sel = 1（只目标列）
          - M : N_sel = N_all（全通道）
          - inputs 保留全部输入通道，供 CRC 使用
        """
        model.eval()
        all_preds, all_trues, all_inputs = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)          # (B, P, N_all)
                batch_y = batch_y.float().to(self.device)          # (B, L+H, N_all 或 1)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input（与基线/LTF一致）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                # 关键修复：评测时也传入 time marks，避免与训练分布不一致
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 与 LTF 完全一致的通道选择策略
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]      # (B, H, N_sel)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]      # (B, H, N_sel)

                all_preds.append(outputs.detach().cpu().numpy())
                all_trues.append(batch_y.detach().cpu().numpy())
                all_inputs.append(batch_x.detach().cpu().numpy())       # (B, P, N_all)

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        return preds, trues, inputs

    def train(self, setting):
        print(">>>>>>> 开始 CRC 两阶段训练流程 <<<<<<<")

        # =======================
        # 阶段一：训练基线模型
        # =======================
        print("\n" + "=" * 50)
        print("--- 阶段一：训练基线模型 ---")
        print("=" * 50)

        baseline_args = deepcopy(self.args)
        baseline_args.model = self.args.baseline_model
        print(f"正在训练基线模型: {baseline_args.model}...")
        baseline_exp = Exp_Long_Term_Forecast(baseline_args)
        self.baseline_model = baseline_exp.train(setting + '_baseline')
        print("基线模型训练完成。")

        # =======================
        # 阶段二：训练 CRC 校正器
        # =======================
        print("\n" + "=" * 50)
        print("--- 阶段二：训练 CRC 校正器 ---")
        print("=" * 50)

        train_set, train_loader = self._get_data(flag='train')
        val_set,   val_loader   = self._get_data(flag='val')

        print("使用基线模型生成预测、真实标签和输入窗口...")
        yhat_tr_bl, y_tr_win, W_tr = self._get_predictions(self.baseline_model, train_loader)  # (B,H,N_sel),(B,H,N_sel),(B,P,N_all)
        yhat_vl_bl, y_vl_win, W_vl = self._get_predictions(self.baseline_model, val_loader)

        print(f"DEBUG: Train - W:{W_tr.shape}, y:{y_tr_win.shape}, yhat:{yhat_tr_bl.shape}")
        print(f"DEBUG:  Val  - W:{W_vl.shape}, y:{y_vl_win.shape}, yhat:{yhat_vl_bl.shape}")

        # 1) 仅用训练集构图，避免数据泄露
        raw_train_data = train_set.scaler.inverse_transform(W_tr.reshape(-1, W_tr.shape[-1]))
        A = build_corr_graph(raw_train_data, topk=self.args.top_k, thr=0.2).astype(np.float32)
        print(f"邻接矩阵完成，密度: {A.sum() / (A.shape[0] ** 2):.3f}")

        # 2) Stage A：MTL 预训练（不确定性加权）
        print("\n--- Stage A: MTL Pre-training ---")
        pre_encoder = PretrainableEncoderMTL(
            A, self.args.seq_len, self.args.pred_len, self.args.enc_in,
            self.args.d_model, self.args.q_val
        )
        mtl_pretrain_trainer(
            pre_encoder,
            W_tr, y_tr_win, W_vl, y_vl_win,
            yhat_tr_bl, yhat_vl_bl,
            mask_ratio=0.0,
            # 其他超参：mask_ratio=0.15, lr=1e-3, wd=1e-4, ...
        )

        # >>>>>>>>>>>>> INSERT HERE: 时间因果可视化（基于 encoder.core 第一层） <<<<<<<<<<<<<
        viz_dir = os.path.join('./results', setting, 'viz_train')
        os.makedirs(viz_dir, exist_ok=True)
        print("[Viz] 绘制时间因果热力图/曲线 ...")
        visualize_time_causality_matrix(pre_encoder.encoder.core, save_dir=viz_dir, tag="after_stageA")

        # 3) Stage B：Hybrid 纠偏（多尺度先验）
        print("\n--- Stage B: Hybrid Corrector (multi-scale priors) ---")
        # 仅用训练集生成先验统计（mu/sd）
        F_tr, F_mu, F_sd, feat_names = build_physics_priors_ms(
            W_tr, A,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True,
            top_m=None,                      # 如需特征筛选可设整数，例如 32
            R_train=(y_tr_win - yhat_tr_bl), # 仅在 top_m 非 None 时会用到
            mu=None, sd=None
        )
        print(f"[Priors] 多尺度先验维度 = {F_tr.shape[-1]}")

        # N = 目标通道数（与 y_tr_win / yhat_tr_bl 最后一维一致）
        N_out = y_tr_win.shape[-1]
        self.hybrid_corrector = HybridTrainer(
            pre_encoder.encoder, A, F_mu=F_mu, F_sd=F_sd,
            H=self.args.pred_len, N=N_out, P=self.args.seq_len,
            # 可选：l2_delta=1e-4, lr=1e-4, wd=1e-4, patience=20, max_epochs=60,
            node_emb_dim=16, use_dir_gate=True,
            quantile_q=0.80, improve_eps=0.01,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True,
            top_m=None
        )

        # 保护性断言（避免通道数不一致）
        assert N_out == yhat_tr_bl.shape[-1], "HybridTrainer.N must equal number of target channels"
        self.hybrid_corrector.train(W_tr, y_tr_win, yhat_tr_bl, W_vl, y_vl_win, yhat_vl_bl)

        # >>>>>>>>>>>>> INSERT HERE: 空间因果矩阵 S（基于验证集窗口）& 可视化 <<<<<<<<<<<<<
        print("[Viz] 估计空间因果矩阵 S (j→i) 并可视化 ...")
        S_val = pre_encoder.encoder.estimate_spatial_causality_matrix(W_vl, batch_size=256)  # (N,N)
        np.save(os.path.join(viz_dir, 'S_val.npy'), S_val)
        node_names_list = [f"N{i}" for i in range(S_val.shape[0])]
        visualize_spatial_causal_heatmap(S_val, node_names=node_names_list, save_dir=viz_dir, tag="val")

        # >>>>>>>>>>>>> INSERT HERE: 与先验的一致性（避免“自我检查”，只是报告用） <<<<<<<<<<<<<
        print("[Check] 先验一致性指标（与 A_corr 对比，仅做报告，不作为“真值”）")
        prec10 = precision_at_k_vs_prior(S_val, A_prior=A, k_ratio=0.10)
        rho_sp = spearman_vs_prior(S_val, A_prior=A)
        print(f"Precision@10% (S vs A_corr): {prec10:.3f}")
        print(f"Spearman (|S| vs A_corr): {rho_sp:.3f}")

        # >>>>>>>>>>>>> INSERT HERE: 滞后一致性（O vs |C|），先用快照序列做近似 <<<<<<<<<<<<<
        print("[Check] 滞后一致性（occlusion vs cross-corr）...")
        # 选几条最强边（按 |S| 排序），以免全量太慢
        N = S_val.shape[0]
        cand = [(i,j,abs(S_val[i,j])) for i in range(N) for j in range(N) if i!=j and S_val[i,j]!=0]
        cand.sort(key=lambda x: -x[2])
        top_pairs = cand[: min(len(cand), 8)]  # 例如取前 8 条

        # 快照近似的连续序列 (B,N)
        snap = quick_val_series_from_windows(W_vl)
        # 简易标准化
        snap = (snap - snap.mean(axis=0, keepdims=True)) / (snap.std(axis=0, keepdims=True) + 1e-8)

        for (i,j,_) in top_pairs:
            # 1) Occlusion：对 j→i 扫描滞后
            lags, O = pre_encoder.encoder.occlusion_curve_a21(
                W_vl, i=i, j=j, 
                max_lag=min(12, W_vl.shape[1]//2),
                win=2, mode='mean'
            )
            
            # 2) Cross-corr：用快照近似的连续序列计算 |C|
            xi = snap[:, i]; xj = snap[:, j]
            # 只取正滞后：corr( xi[t], xj[t-L] )
            Lmax = min(12, len(xi)//4)
            C = []
            Ls = list(range(1, Lmax+1))
            
            if not Ls: # 如果 Ls 为空（序列太短），则跳过
                print(f"  pair j→i = {j}→{i} | Skipped (Lmax=0)")
                continue

            for L in Ls:
                xi_cut = xi[L:]; xj_cut = xj[:-L]
                C.append( (xi_cut * xj_cut).sum() / (np.sqrt((xi_cut**2).sum()) * np.sqrt((xj_cut**2).sum()) + 1e-8) )
            C = np.asarray(C, dtype=np.float32)

            # 3) 指标
            O_n = O[:len(Ls)] / (O[:len(Ls)].max() + 1e-8)
            C_n = np.abs(C) / (np.abs(C).max() + 1e-8)
            rho = spearmanr(O_n, C_n).correlation
            rho = 0.0 if rho != rho else rho # 处理 NaN 的情况
            l_occ = int(Ls[int(np.argmax(O_n))]); l_xc = int(Ls[int(np.argmax(C_n))])
            print(f"  pair j→i = {j}→{i} | Δlag={abs(l_occ-l_xc)}, ρ={rho:.2f} (Lmax={Lmax})")

            # ========================================================
            # === 在这里插入您的可视化代码 ===
            # ========================================================
            plt.figure(figsize=(10, 4)) # 创建一个新的图形
            
            # 绘制模型曲线
            plt.plot(Ls, O_n, 'b-o', label='Model Importance (Occlusion)', markersize=4, linewidth=2)
            # 绘制数据曲线
            plt.plot(Ls, C_n, 'r--s', label='Data Correlation (|C|)', markersize=4, linewidth=2)
            
            # 垂直线标记模型峰值
            plt.axvline(x=l_occ, color='blue', linestyle=':', alpha=0.7, label=f'Model Peak (Lag={l_occ})')
            # 垂直线标记数据峰值
            plt.axvline(x=l_xc, color='red', linestyle=':', alpha=0.7, label=f'Data Peak (Lag={l_xc})')

            plt.title(f"Lag Consistency Check: pair j→i = {j}→{i}\nΔlag={abs(l_occ-l_xc)}, ρ={rho:.2f}")
            plt.xlabel("Lag (L steps)")
            plt.ylabel("Normalized Importance / Correlation")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.ylim(-0.05, 1.05) # 设置Y轴范围
            plt.xlim(0, Lmax + 1) # 设置X轴范围
            plt.tight_layout()
            
            # 确保 viz_dir 变量在循环外已经定义
            save_path = os.path.join(viz_dir, f"lag_consistency_pair_{j}_to_{i}.png")
            plt.savefig(save_path, dpi=150)
            plt.close() # <-- 在循环中必须关闭图形，防止内存泄漏和图表重叠
            # ========================================================
            # === 可视化代码结束 ===
            # ========================================================

        print("\n>>>>>>> CRC 训练流程全部完成 <<<<<<<")
        return self.hybrid_corrector

    # ---- LTF 风格的 inverse：必要时在通道维做 tile → inverse → 裁回 ----
    def _inverse_like_ltf(self, arr, ref_cols, scaler):
        """
        输入:
          arr      : (B, H, C_sel)
          ref_cols : C_all（Dataset 中 df_data 的列数）
          scaler   : 与 Dataset 相同的 StandardScaler
        返回:
          arr_inv  : (B, H, C_sel) 逆标准化后的数组
        说明:
          - 若 C_sel == C_all : 直接 inverse；
          - 若 C_sel <  C_all : 在通道维 tile 至 >= C_all，inverse 后再裁回最后的 C_sel 通道。
        """
        B, H, C_sel = arr.shape
        if C_sel == ref_cols:
            return scaler.inverse_transform(arr.reshape(B * H, C_sel)).reshape(B, H, C_sel)

        reps = int(np.ceil(ref_cols / max(C_sel, 1)))
        tiled = np.tile(arr, [1, 1, reps])[:, :, :ref_cols]  # (B, H, C_all)
        inv_full = scaler.inverse_transform(tiled.reshape(B * H, ref_cols)).reshape(B, H, ref_cols)
        return inv_full[:, :, -C_sel:]  # 按“选择最后几列”的习惯裁回（MS=最后1列；M=全列时不会走到这里）

    def test(self, setting):
        print(">>>>>>> 开始 CRC 测试流程 <<<<<<<")
        if self.hybrid_corrector is None or self.baseline_model is None:
            raise ValueError("模型尚未训练，请先调用 train()。")

        test_set, test_loader = self._get_data(flag='test')

        print("在测试集上生成基线预测、真实标签和输入窗口...")
        yhat_te_bl, y_te, W_te = self._get_predictions(self.baseline_model, test_loader)  # (B,H,C_sel),(B,H,C_sel),(B,P,C_all)

        print("使用 Hybrid Corrector 对基线预测进行校正...")
        yfinal_te = self.hybrid_corrector.predict(W_te, yhat_te_bl)  # (B, H, C_sel)

        # ---------- 与 LTF 对齐的可选 inverse ----------
        if getattr(test_set, "scale", False) and getattr(self.args, "inverse", False):
            C_all = W_te.shape[-1]   # Dataset 中 scaler 期望的列数
            train_set, _ = self._get_data(flag='train')
            scaler = train_set.scaler
            y_te       = self._inverse_like_ltf(y_te,       C_all, scaler)
            yhat_te_bl = self._inverse_like_ltf(yhat_te_bl, C_all, scaler)
            yfinal_te  = self._inverse_like_ltf(yfinal_te,  C_all, scaler)
            # 输入窗口也 inverse，便于可视化
            try:
                W_te = scaler.inverse_transform(W_te.reshape(-1, C_all)).reshape(W_te.shape)
            except Exception:
                pass

        # ---------- 评估 ----------
        print("\n" + "=" * 50)
        print("--- 最终性能评估（统一协议） ---")
        print("=" * 50)
        base_mae, base_mse, _, _, _ = metric(yhat_te_bl, y_te)
        print(f"基线模型({self.args.baseline_model}) -> MSE: {base_mse:.4f}, MAE: {base_mae:.4f}")

        final_mae, final_mse, _, _, _ = metric(yfinal_te, y_te)
        print(f"CRC 纠偏后 -> MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")

        # ---------- DTW（flatten(H*C)） ----------
        if self.args.use_dtw:
            print("计算 DTW (flatten(H*C)) ...")
            manhattan = lambda x, y: np.abs(x - y)
            dtw_list = []
            for i in range(yfinal_te.shape[0]):
                x = yfinal_te[i].reshape(-1, 1)  # (H*C_sel, 1)
                y = y_te[i].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan)
                dtw_list.append(d)
            print(f"CRC 纠偏后 -> DTW: {np.mean(dtw_list):.4f}")

        print("-" * 50)
        mae_improve = (base_mae - final_mae) / max(base_mae, 1e-8) * 100
        mse_improve = (base_mse - final_mse) / max(base_mse, 1e-8) * 100
        print(f"MAE 相对提升率: {mae_improve:.2f}%")
        print(f"MSE 相对提升率: {mse_improve:.2f}%")
        print("=" * 50)

        # ---------- 可视化（可选） ----------
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        print("生成可视化图片...")
        for i in range(min(W_te.shape[0], 100)):
            if i % 20 == 0:
                # 仍沿用“最后一列”的可视化习惯；如需严格按目标列索引，按需替换这里的 -1
                gt = np.concatenate((W_te[i, :, -1], y_te[i, :, -1]), axis=0)
                pd = np.concatenate((W_te[i, :, -1], yfinal_te[i, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))
        print(f"图片保存在: {folder_path}")

        # ---------- 保存结果 ----------
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'metrics.npy', np.array([final_mae, final_mse]))
        np.save(folder_path + 'pred.npy', yfinal_te)
        np.save(folder_path + 'true.npy', y_te)
        np.save(folder_path + 'pred_baseline.npy', yhat_te_bl)
        return
