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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import spearmanr
import seaborn as sns


warnings.filterwarnings('ignore')

def visualize_dynamic_spatial_causality_composite(
    dyn_series: np.ndarray, 
    edge_index: np.ndarray, 
    N: int, 
    num_snapshots: int = 4,   # 核心参数：要展示的快照数量
    sample_stride: int = 100, # 核心参数：快照之间的时间间隔
    save_dir: str = None
):
    """
    基于动态边强度序列，生成包含多个子图的复合空间因果矩阵图。
    """
    B, M = dyn_series.shape
    
    # 1. 确定要采样的索引
    # 确保索引不超过 B
    sample_indices = np.arange(0, B, sample_stride)
    # 只取前 num_snapshots 个，如果不够则全部取
    sample_indices = sample_indices[:num_snapshots] 
    
    if len(sample_indices) < 2:
        print("警告: 采样的快照数量不足 2 个，无法有效展示动态变化。")
        return

    # 2. 准备映射关系和标签
    target_nodes = edge_index[0, :] # i
    source_nodes = edge_index[1, :] # j
    node_names = [f"N{i}" for i in range(N)]
    
    # 确定全局 VMAX/VMIN 以统一色标
    global_vmax = np.max(dyn_series)
    
    # 3. 创建画布和子图
    # 布局为 1 行 num_snapshots 列
    # 调整figsize以容纳多个子图和一个共享的色标
    figsize = (4 * num_snapshots + 2, max(5, N / 3)) # +2 用于色标
    fig, axes = plt.subplots(
        1, len(sample_indices), 
        figsize=figsize, 
        sharey=True, 
        gridspec_kw={'wspace': 0.1} # 减少子图间距
    )
    
    # 如果只有一个子图，plt.subplots返回的不是数组，需要转换
    if len(sample_indices) == 1:
        axes = [axes]

    print(f"将生成包含 {len(sample_indices)} 个子图的复合动态因果矩阵图...")

    for k, t in enumerate(sample_indices):
        ax = axes[k]
        current_strengths = dyn_series[t, :]
        S_t = np.zeros((N, N), dtype=np.float32)
        
        # 填充 S_t 矩阵 (S_t[i, j] = strength of j -> i)
        for i_idx in range(M):
            i = target_nodes[i_idx]
            j = source_nodes[i_idx]
            S_t[i, j] = current_strengths[i_idx]
        
        # 4. 绘制热力图
        # 注意: cbar=False 用于除了最后一个子图外的所有图
        cbar_bool = (k == len(sample_indices) - 1)
        
        sns.heatmap(
            S_t, 
            ax=ax,
            xticklabels=node_names, 
            yticklabels=node_names if k == 0 else False, # 只有第一张图显示Y轴标签
            cmap='Reds',
            vmin=0, vmax=global_vmax, # 统一色标
            cbar=cbar_bool,           # 只有最后一张图显示色标
            cbar_kws={'label': 'Causal Strength (αj→i)'}
        )
        
        ax.set_title(f"Window Index {t}", fontsize=10)
        ax.set_xlabel("Source Node (j)", fontsize=8)
        
        # 调整刻度标签，防止重叠
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(node_names, rotation=90, fontsize=6)
        
        if k == 0:
             ax.set_ylabel("Target Node (i)", fontsize=8)
        else:
             # 确保后续子图的Y轴标签和刻度被隐藏，但共享坐标
             ax.tick_params(axis='y', length=0)
             
    plt.suptitle(f"Dynamic Spatial Causality Matrix Snapshots ({len(sample_indices)} Time Points)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 为suptitle留出空间
    
    # 5. 保存
    if save_dir:
        save_path = os.path.join(save_dir, "dynamic_causal_composite.png")
        print(f"Saving composite heatmap to {save_path}")
        plt.savefig(save_path, dpi=300)
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
            print(f"[CRC] Loading hybrid corrector from {crc_ckpt_path}")
            try:
                self.hybrid_corrector = torch.load(
                    crc_ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                self.hybrid_corrector = torch.load(crc_ckpt_path, map_location=self.device)
            print("[CRC] crc_skip_retrain: 阶段二完成（未重训 Hybrid）。")
            return self.hybrid_corrector

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
        A = build_corr_graph(raw_train_data, topk=self.args.top_k, thr=0.2).astype(np.float32)
        print(f"邻接矩阵完成，密度: {A.sum() / (A.shape[0] ** 2):.3f}")

        # 2) Stage A：MTL 预训练（不确定性加权）
        print("\n--- Stage A: MTL Pre-training ---")
        # q_val 用作两个目的：
        #  (1) 传给 TorchCausalStateEncoder 的 Q（编码器容量超参）；
        #  (2) 通过分位数映射控制 HybridTrainer 中的安全边界 quantile_q。
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

        # 将整数 q_val 映射到 HybridTrainer 使用的分位数 quantile_q，
        # 以控制安全边界 tau 的松紧（从“非常保守”到“较激进”）。
        # 这里刻意拉大分位数跨度：
        #   q_val = 1 -> quantile_q = 0.10  (极保守，coverage 预期很低)
        #   q_val = 2 -> quantile_q = 0.30
        #   q_val = 3 -> quantile_q = 0.50
        #   q_val = 4 -> quantile_q = 0.70
        #   q_val >= 5 -> quantile_q = 0.90  (较激进，coverage 预期较高)
        q_int = int(getattr(self.args, 'q_val', 3))
        if q_int <= 1:
            quantile_q = 0.10
        elif q_int == 2:
            quantile_q = 0.30
        elif q_int == 3:
            quantile_q = 0.50
        elif q_int == 4:
            quantile_q = 0.70
        else:
            quantile_q = 0.90
        print(f"[Hybrid] Using safety quantile_q={quantile_q:.2f} mapped from q_val={q_int}")

        # N = 目标通道数（与 y_tr_win / yhat_tr_bl 最后一维一致）
        N_out = y_tr_win.shape[-1]
        self.hybrid_corrector = HybridTrainer(
            pre_encoder.encoder, A, F_mu=F_mu, F_sd=F_sd,
            H=self.args.pred_len, N=N_out, P=self.args.seq_len,
            # 可选：l2_delta=1e-4, lr=1e-4, wd=1e-4, patience=20, max_epochs=60,
            node_emb_dim=16, use_dir_gate=True,
            quantile_q=quantile_q, improve_eps=0.01,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True,
            top_m=None
        )

        # 保护性断言（避免通道数不一致）
        assert N_out == yhat_tr_bl.shape[-1], "HybridTrainer.N must equal number of target channels"
        # HybridTrainer 的验证集使用 (可能子采样后的) W_vl_sub / y_vl_sub / yhat_vl_sub
        self.hybrid_corrector.train(W_tr, y_tr_win, yhat_tr_bl, W_vl_sub, y_vl_sub, yhat_vl_sub)

        # 训练完成后保存 CRC 纠偏器 checkpoint（见 --crc_ckpt_dir / --crc_skip_retrain）
        crc_save_path = crc_ckpt_path
        torch.save(self.hybrid_corrector, crc_save_path)
        print(f"[Checkpoint] Saved CRC hybrid corrector to {crc_save_path}")

        # >>>>>>>>>>>>> INSERT HERE: 空间因果矩阵 S（基于验证集窗口）& 可视化 <<<<<<<<<<<<<
        print("[Viz] 估计空间因果矩阵 S (j→i) 并可视化 ...")
        viz_dir = os.path.join('./results', setting, 'viz_train')
        os.makedirs(viz_dir, exist_ok=True)
        # S_val = pre_encoder.encoder.estimate_spatial_causality_matrix(W_vl, batch_size=256)  # (N,N)
        # np.save(os.path.join(viz_dir, 'S_val.npy'), S_val)
        # 展示动态序列
        S_val, dyn_series_val = pre_encoder.encoder.estimate_spatial_causality_matrix(
            W_vl, batch_size=256, 
            return_series=True  
        )  
        np.save(os.path.join(viz_dir, 'S_val_dynamic_series.npy'), dyn_series_val)
        node_names_list = [f"N{i}" for i in range(S_val.shape[0])]
        visualize_spatial_causal_heatmap(S_val, node_names=node_names_list, save_dir=viz_dir, tag="val_avg")

        N_nodes = S_val.shape[0] 

        visualize_dynamic_spatial_causality_composite(
            dyn_series=dyn_series_val, 
            edge_index=pre_encoder.encoder.edge_index.cpu().numpy(), 
            N=N_nodes,
            num_snapshots=4,     # 论文中展示 4 个快照
            sample_stride=200,   # 每隔 200 个时间窗口取一个快照
            save_dir=viz_dir
        )
        
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

        # 仅在推理阶段应用 tau_scale：训练阶段始终使用原始 tau_map
        tau_scale = float(getattr(self.args, "tau_scale", 1.0))
        if hasattr(self.hybrid_corrector, "tau_scale"):
            self.hybrid_corrector.tau_scale = tau_scale

        print(f"使用 Hybrid Corrector 对基线预测进行校正 (tau_scale={tau_scale:.3f}) ...")
        yfinal_te, nl_abs, clip_mask = self.hybrid_corrector.predict(
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
