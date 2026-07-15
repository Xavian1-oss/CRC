import os
import torch
import numpy as np
import warnings
from copy import deepcopy

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.CRC import (
    ResidualCorrector, CRCConfig,
    build_corr_graph, load_crc_checkpoint, save_crc_checkpoint,
)
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.dtw_metric import accelerated_dtw


warnings.filterwarnings('ignore')

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
        self.corrector = None

    @property
    def hybrid_corrector(self):
        """Backward-compatible alias for ResidualCorrector.hybrid."""
        return self.corrector.hybrid if self.corrector is not None else None

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

        # 为 baseline 构造 dataset-specific 的 setting，避免多个 DATASETS（例如 Electricity/Weather）
        # 在 data="custom" 时共用同一套 checkpoint，导致维度不匹配
        if getattr(self.args, 'data', None) != 'custom':
            dataset_tag = self.args.data
        else:
            data_path = getattr(self.args, 'data_path', '')
            if data_path:
                dataset_tag = os.path.splitext(os.path.basename(data_path))[0]
            else:
                root_path = getattr(self.args, 'root_path', '')
                if root_path:
                    dataset_tag = os.path.basename(os.path.normpath(root_path))
                else:
                    dataset_tag = 'custom'

        # baseline 的 setting 不依赖 CRC 的验证比例，但会区分不同数据集与输入维度
        baseline_setting = (
            f"baseline_{self.args.baseline_model}_{dataset_tag}"
            f"_pl{self.args.pred_len}_dm{self.args.d_model}_enc{self.args.enc_in}"
        )
        baseline_ckpt_dir = os.path.join(self.args.checkpoints, baseline_setting)
        baseline_ckpt_path = os.path.join(baseline_ckpt_dir, 'checkpoint.pth')

        baseline_exp = Exp_Long_Term_Forecast(baseline_args)

        # ---- Warm start for baseline: directly load its checkpoint if available ----
        if getattr(self.args, 'warm_start', False) and os.path.exists(baseline_ckpt_path):
            print(f"[WarmStart] Found baseline checkpoint at {baseline_ckpt_path}, trying to load and skip baseline training.")
            ckpt = torch.load(baseline_ckpt_path, map_location=self.device)
            try:
                baseline_exp.model.load_state_dict(ckpt)
                self.baseline_model = baseline_exp.model.to(self.device)
                print("[WarmStart] Baseline checkpoint loaded successfully.")
            except RuntimeError as e:
                msg = str(e)
                # 同样特判 TimeFeatureEmbedding 的延迟初始化导致的 embed.* 额外参数
                if 'enc_embedding.temporal_embedding.embed' in msg:
                    print("[WarmStart] Baseline temporal_embedding.embed mismatch only; retrying with strict=False.")
                    baseline_exp.model.load_state_dict(ckpt, strict=False)
                    self.baseline_model = baseline_exp.model.to(self.device)
                    print("[WarmStart] Baseline checkpoint loaded successfully with relaxed strictness.")
                else:
                    # 其他真正不兼容的情况：回退为重新训练并覆盖
                    print(f"[WarmStart] Failed to load baseline checkpoint due to mismatch: {e}\n"
                          f"[WarmStart] Retraining baseline model and overwriting checkpoint.")
                    self.baseline_model = baseline_exp.train(baseline_setting)
                    print("基线模型训练完成。")
        else:
            print(f"正在训练基线模型: {baseline_args.model} (setting={baseline_setting}) ...")
            self.baseline_model = baseline_exp.train(baseline_setting)
            print("基线模型训练完成。")

        val_ratio = float(getattr(self.args, 'crc_val_ratio', 1.0))
        if val_ratio <= 0.0 or val_ratio > 1.0:
            raise ValueError(f"crc_val_ratio must be in (0,1], got {val_ratio}")
        val_suffix = int(round(val_ratio * 100))
        crc_fname = f'crc_corrector_val{val_suffix}.pth'
        crc_ckpt_dir_arg = getattr(self.args, 'crc_ckpt_dir', None)
        if crc_ckpt_dir_arg and str(crc_ckpt_dir_arg).strip():
            crc_ckpt_dir = os.path.normpath(str(crc_ckpt_dir_arg).strip())
            os.makedirs(crc_ckpt_dir, exist_ok=True)
            crc_ckpt_path = os.path.join(crc_ckpt_dir, crc_fname)
        else:
            crc_ckpt_dir = os.path.join(self.args.checkpoints, setting)
            os.makedirs(crc_ckpt_dir, exist_ok=True)
            crc_ckpt_path = os.path.join(crc_ckpt_dir, crc_fname)

        if getattr(self.args, 'crc_skip_retrain', False):
            if not os.path.isfile(crc_ckpt_path):
                raise FileNotFoundError(
                    f"--crc_skip_retrain set but checkpoint not found: {crc_ckpt_path}"
                )
            print("\n" + "=" * 50)
            print("--- 阶段二：CRC 校正器（跳过训练，从 checkpoint 加载） ---")
            print("=" * 50)
            print(f"[CRC] Loading corrector from {crc_ckpt_path}")
            self.corrector = load_crc_checkpoint(crc_ckpt_path, map_location=self.device)
            print("[CRC] crc_skip_retrain: 阶段二完成（未重训 corrector）。")
            return self.corrector

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

        # ---- 根据 crc_val_ratio 子采样验证窗口，仅用于 HybridTrainer（安全机制）的训练 ----
        if val_ratio < 1.0:
            num_v = max(1, int(round(W_vl.shape[0] * val_ratio)))
            rng = np.random.RandomState(getattr(self.args, 'seed', 42))
            idx = rng.permutation(W_vl.shape[0])[:num_v]
            W_vl_sub = W_vl[idx]
            y_vl_sub = y_vl_win[idx]
            yhat_vl_sub = yhat_vl_bl[idx]
            print(f"[CRC] Using a subset of val windows for HybridTrainer: ratio={val_ratio:.2f}, size={num_v}/{W_vl.shape[0]}")
        else:
            W_vl_sub, y_vl_sub, yhat_vl_sub = W_vl, y_vl_win, yhat_vl_bl

        # 1) 仅用训练集构图，避免数据泄露
        raw_train_data = train_set.scaler.inverse_transform(W_tr.reshape(-1, W_tr.shape[-1]))
        crc_cfg = CRCConfig.from_args(self.args)
        A = build_corr_graph(raw_train_data, topk=crc_cfg.graph_topk, thr=0.2).astype(np.float32)
        print(f"邻接矩阵完成，密度: {A.sum() / (A.shape[0] ** 2):.3f}")

        N_out = y_tr_win.shape[-1]
        self.corrector = ResidualCorrector(
            crc_cfg, A, H=self.args.pred_len, N=N_out,
            P=self.args.seq_len, d_model=self.args.d_model, enc_in=self.args.enc_in,
        )
        assert N_out == yhat_tr_bl.shape[-1], "Corrector.N must equal number of target channels"
        self.corrector.train(W_tr, y_tr_win, yhat_tr_bl, W_vl_sub, y_vl_sub, yhat_vl_sub)

        save_crc_checkpoint(crc_ckpt_path, self.corrector)
        print(f"[Checkpoint] Saved CRC corrector (version={ResidualCorrector.VERSION}) to {crc_ckpt_path}")

        # ---------- 可视化 ----------
        # 批量实验时会显著拖慢速度且可能触发绘图边界问题，这里默认不在 train() 中生成空间因果可视化。
        
        print("\n>>>>>>> CRC 训练流程全部完成 <<<<<<<")
        return self.corrector

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
        if self.corrector is None or self.baseline_model is None:
            raise ValueError("模型尚未训练，请先调用 train()。")

        test_set, test_loader = self._get_data(flag='test')

        print("在测试集上生成基线预测、真实标签和输入窗口...")
        yhat_te_bl, y_te, W_te = self._get_predictions(self.baseline_model, test_loader)  # (B,H,C_sel),(B,H,C_sel),(B,P,C_all)

        print("使用 ResidualCorrector 对基线预测进行校正 ...")
        yfinal_te, nl_abs, clip_mask = self.corrector.predict(
            W_te, yhat_te_bl, return_nl_abs=True, return_clip_mask=True
        )  # nl_abs = |w2 * r_delta_c|；clip_mask = (|r_delta| > tau)

        # NDR (非恶化率)：按 (horizon, node) 聚合 MAE，再比较修正前后误差
        err_baseline = np.abs(y_te - yhat_te_bl)  # (B, H, C)
        err_final = np.abs(y_te - yfinal_te)      # (B, H, C)
        mae_baseline = err_baseline.mean(axis=0)  # (H, C)
        mae_final = err_final.mean(axis=0)        # (H, C)
        non_degraded_mask = mae_final <= (mae_baseline + 1e-8)
        paper_ndr = np.mean(non_degraded_mask) * 100.0

        # Correction coverage（tau-sensitive）：非线性支路被 τ 限幅（clip）到边界的比例
        coverage = clip_mask.mean() * 100.0

        # 额外统计：非线性经限幅后的“有效贡献”超过阈值的比例（可能不随 tau_scale 明显变化）
        cov_eps = float(getattr(self.args, "crc_coverage_eps", 1e-4))
        nl_active = (nl_abs > cov_eps).mean() * 100.0

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
        print(f"NDR (非恶化率): {paper_ndr:.1f}%") # 保持一位小数
        print(f"Correction coverage (clip rate |r_delta|>tau): {coverage:.1f}%")
        print(f"Nonlinear active (|w2·r_delta_c| > {cov_eps:g}): {nl_active:.1f}%")
        print("=" * 50)

        # ---------- 可视化 ----------
        # 批量实验时会生成大量文件且拖慢速度，这里默认不在 test() 中生成可视化。

        # ---------- 保存结果 ----------
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'metrics.npy', np.array([final_mae, final_mse]))
        np.save(folder_path + 'pred.npy', yfinal_te)
        np.save(folder_path + 'true.npy', y_te)
        np.save(folder_path + 'pred_baseline.npy', yhat_te_bl)
        return
