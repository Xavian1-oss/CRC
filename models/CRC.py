from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# =========================
# Helpers
# =========================
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def make_identity_adjacency(N: int) -> np.ndarray: return np.eye(N, dtype=np.int32)

def build_corr_graph(X_tr_raw: np.ndarray, topk: int = 3, thr: Optional[float] = 0.2) -> np.ndarray:
    """Correlation graph on training data; supports (T,N) or (B,P,N)."""
    X = X_tr_raw.reshape(-1, X_tr_raw.shape[-1]) if X_tr_raw.ndim == 3 else X_tr_raw
    C = np.corrcoef(X.T); N = C.shape[0]; A = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        idx = np.argsort(-np.abs(C[i])); cnt = 0
        for j in idx:
            if i == j: continue
            if (thr is None) or (abs(float(C[i,j])) >= thr):
                A[i, j] = 1; cnt += 1
            if cnt >= topk: break
    np.fill_diagonal(A, 1); return A

@dataclass
class GraphCfg:
    N: int = 1; P: int = 96; H: int = 24
    d_model: int = 64; Q: int = 2; seed: int = 42

def sliding_windows(X: np.ndarray, P: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    T, N = X.shape; B = T - P - H + 1
    if B <= 0: return np.empty((0, P, N), np.float32), np.empty((0, H, N), np.float32)
    W = np.array([X[b:b+P] for b in range(B)], dtype=np.float32)
    y = np.array([X[b+P:b+P+H] for b in range(B)], dtype=np.float32)
    return W, y

# =========================
# Multi-Scale Priors
# =========================
def build_physics_priors_ms(
    W: np.ndarray,
    A: np.ndarray,
    scales: List[int] = [6, 12, 24, 48],
    freq_bands: List[Tuple[float, float]] = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
    use_robust: bool = True,
    top_m: Optional[int] = None,            # e.g. 32; None = no feature selection
    R_train: Optional[np.ndarray] = None,   # (B,H,N), only used when top_m is set
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    构建多尺度物理先验特征，并进行邻居聚合与标准化。
    返回:
      F_norm: (B,N,F)  标准化后的特征
      mu:     (1,1,F)  标准化用中心（robust: median；否则 mean）
      sd:     (1,1,F)  标准化用尺度（robust: MAD；否则 std）
      names:  List[str] 特征名列表（顺序与 F_norm 的最后一维一致）
    变更点（FIX）:
      - 使用行归一化后的 A 进行邻居聚合，缓解度数偏置。
      - 改良 ACF 归一化口径，使分母与分子长度一致，稳定性更好。
    """
    assert W.ndim == 3, "W must be (B,P,N)"
    B, P, N = W.shape
    A = A.astype(np.float32)
    feats: List[np.ndarray] = []
    names: List[str] = []

    # ---------- helpers ----------
    def _ema_last_window(x: np.ndarray, s: int) -> np.ndarray:
        """对最后 s 步做 EMA，返回 (B,N)"""
        alpha = 2.0 / (s + 1.0)
        xk = x[:, -min(s, P):, :]  # (B, s_eff, N)
        ema = np.zeros((B, N), dtype=np.float32)
        for t in range(xk.shape[1]):
            ema = xk[:, t, :] if t == 0 else alpha * xk[:, t, :] + (1 - alpha) * ema
        return ema.astype(np.float32)

    def _acf_lag(xs: np.ndarray, lag: int) -> np.ndarray:
        """
        改良版 ACF：分母按对应长度对齐，避免不同长度带来的偏差。
        xs: (B, L, N)
        返回: (B, N)
        """
        if xs.shape[1] <= lag:
            return np.zeros((xs.shape[0], xs.shape[2]), dtype=np.float32)
        x = xs - xs.mean(axis=1, keepdims=True)        # (B, L, N)
        num = (x[:, lag:, :] * x[:, :-lag, :]).sum(axis=1)            # (B, N)
        d1 = np.sqrt((x[:, lag:, :] ** 2).sum(axis=1))                 # (B, N)
        d2 = np.sqrt((x[:, :-lag, :] ** 2).sum(axis=1))                # (B, N)
        return (num / (d1 * d2 + 1e-8)).astype(np.float32)

    def _fft_bands(xs: np.ndarray, bands: List[Tuple[float, float]]) -> np.ndarray:
        """
        对每个节点在窗口内做 Hann 加窗 + rFFT，按频段累加功率
        xs: (B, L, N)
        返回: (B, N, #bands)
        """
        Bz, Lz, Nz = xs.shape
        if Lz <= 0:
            return np.zeros((Bz, Nz, len(bands)), dtype=np.float32)
        win = np.hanning(Lz).astype(np.float32)[None, :, None]   # (1,L,1)
        xw = xs * win
        spec = np.fft.rfft(xw, axis=1)
        power = (spec.real**2 + spec.imag**2) / max(Lz, 1)
        freqs = np.fft.rfftfreq(Lz, d=1.0)
        out = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                out.append(power[:, mask, :].sum(axis=1).astype(np.float32))
            else:
                out.append(np.zeros((Bz, Nz), dtype=np.float32))
        return np.stack(out, axis=-1)  # (B,N,#bands)

    # ---------- 单点统计 ----------
    last = W[:, -1, :]                                 # (B,N)
    prev = W[:, -2, :] if P >= 2 else W[:, -1, :]
    feats += [last, last - prev, last * prev, last**2]
    names += ["last", "diff1", "prod12", "last_sq"]

    # ---------- 多尺度统计 ----------
    for s in scales:
        s_eff = min(s, P)
        recent = W[:, -s_eff:, :]                      # (B,s_eff,N)
        feats.append(recent.mean(axis=1)); names.append(f"mean_s{s}")
        feats.append(_ema_last_window(W, s_eff)); names.append(f"ema_s{s}")
        feats.append(recent.var(axis=1)); names.append(f"var_s{s}")
        q75 = np.quantile(recent, 0.75, axis=1).astype(np.float32)
        q25 = np.quantile(recent, 0.25, axis=1).astype(np.float32)
        feats.append(q75 - q25); names.append(f"iqr_s{s}")
        feats.append(np.abs(np.diff(recent, axis=1)).sum(axis=1)); names.append(f"absdiff_energy_s{s}")
        feats.append(_acf_lag(recent, 1)); names.append(f"acf1_s{s}")
        feats.append(_acf_lag(recent, max(2, s_eff // 2))); names.append(f"acf_half_s{s}")
        pb = _fft_bands(recent, freq_bands)            # (B,N,#bands)
        for bi in range(pb.shape[-1]):
            feats.append(pb[..., bi]); names.append(f"pband{bi}_s{s}")

    # 聚合前的基础特征
    F0 = np.stack(feats, axis=-1).astype(np.float32)   # (B,N,F0)
    Bz, Nz, Fz = F0.shape

    # ---------- FIX: 图行归一化，缓解度偏置 ----------
    deg = A.sum(axis=1, keepdims=True).astype(np.float32)
    deg = np.clip(deg, 1.0, None)                      # 防止除零
    An = (A / deg).astype(np.float32)                  # 行归一化 A
    At = An.T                                          # 用 An 的转置做邻居聚合

    # 邻居聚合 (按列: 对每个特征维做一次 (B,N) @ (N,N))
    Fg = np.empty_like(F0, dtype=np.float32)
    for k in range(Fz):
        Fg[:, :, k] = F0[:, :, k] @ At                 # (B,N)

    # ---------- 可选: 按残差相关性进行特征选择 ----------
    if (top_m is not None) and (R_train is not None):
        # R_train: (B,H,N) -> 聚合为 (B,N)，采用 |R| 的时间平均
        Rm = np.mean(np.abs(R_train), axis=1)          # (B,N)
        F_sel = np.zeros_like(Fg, dtype=np.float32)
        for i in range(N):
            Fi = Fg[:, i, :]                           # (B,Fz)
            ri = Rm[:, i]                              # (B,)
            # 去中心化
            Fi_c = Fi - Fi.mean(axis=0, keepdims=True)
            ri_c = ri - ri.mean()
            num = (Fi_c * ri_c[:, None]).sum(axis=0)                   # (Fz,)
            den = np.sqrt((Fi_c**2).sum(axis=0)) * (np.sqrt((ri_c**2).sum()) + 1e-8)
            corr = num / (den + 1e-8)                                   # (Fz,)
            top_idx = np.argsort(-np.abs(corr))[:top_m]
            mask = np.zeros(Fz, dtype=bool); mask[top_idx] = True
            F_sel[:, i, :] = Fg[:, i, :] * mask[None, :]
        Fg = F_sel

    # ---------- 标准化 ----------
    # 若给定 mu/sd 则直接使用（推理/验证阶段）
    if (mu is not None) and (sd is not None):
        F_norm = ((Fg - mu) / (sd + 1e-6)).astype(np.float32)
        return F_norm, mu.astype(np.float32), sd.astype(np.float32), names

    # 否则根据当前数据估计（训练阶段）
    if use_robust:
        med = np.median(Fg, axis=(0, 1), keepdims=True)
        mad = np.median(np.abs(Fg - med), axis=(0, 1), keepdims=True) + 1e-6
        F_norm = ((Fg - med) / mad).astype(np.float32)
        return F_norm, med.astype(np.float32), mad.astype(np.float32), names
    else:
        mu_out = Fg.mean(axis=(0, 1), keepdims=True)
        sd_out = Fg.std(axis=(0, 1), keepdims=True) + 1e-6
        F_norm = ((Fg - mu_out) / sd_out).astype(np.float32)
        return F_norm, mu_out.astype(np.float32), sd_out.astype(np.float32), names


def build_physics_priors_minimal(
    W: np.ndarray,
    A: np.ndarray,
    ema_scale: int = 12,
    use_robust: bool = True,
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Minimal physics priors: last, diff1, ema_s12 (+ neighbor aggregation)."""
    assert W.ndim == 3, "W must be (B,P,N)"
    B, P, N = W.shape
    A = A.astype(np.float32)

    last = W[:, -1, :]
    prev = W[:, -2, :] if P >= 2 else W[:, -1, :]
    diff1 = last - prev

    s_eff = min(ema_scale, P)
    xk = W[:, -s_eff:, :]
    alpha = 2.0 / (s_eff + 1.0)
    ema = np.zeros((B, N), dtype=np.float32)
    for t in range(xk.shape[1]):
        ema = xk[:, t, :] if t == 0 else alpha * xk[:, t, :] + (1 - alpha) * ema
    ema = ema.astype(np.float32)

    names = ["last", "diff1", f"ema_s{ema_scale}"]
    F0 = np.stack([last, diff1, ema], axis=-1).astype(np.float32)

    deg = A.sum(axis=1, keepdims=True).astype(np.float32)
    deg = np.clip(deg, 1.0, None)
    At = (A / deg).astype(np.float32).T
    Fg = np.empty_like(F0, dtype=np.float32)
    for k in range(F0.shape[-1]):
        Fg[:, :, k] = F0[:, :, k] @ At

    if (mu is not None) and (sd is not None):
        F_norm = ((Fg - mu) / (sd + 1e-6)).astype(np.float32)
        return F_norm, mu.astype(np.float32), sd.astype(np.float32), names

    if use_robust:
        med = np.median(Fg, axis=(0, 1), keepdims=True)
        mad = np.median(np.abs(Fg - med), axis=(0, 1), keepdims=True) + 1e-6
        F_norm = ((Fg - med) / mad).astype(np.float32)
        return F_norm, med.astype(np.float32), mad.astype(np.float32), names

    mu_out = Fg.mean(axis=(0, 1), keepdims=True)
    sd_out = Fg.std(axis=(0, 1), keepdims=True) + 1e-6
    F_norm = ((Fg - mu_out) / sd_out).astype(np.float32)
    return F_norm, mu_out.astype(np.float32), sd_out.astype(np.float32), names

# =========================
# Encoder / MTL (uncertainty-weighted)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, i_d, o_d):
        super().__init__(); self.l = nn.Linear(i_d, o_d)
        self.s = nn.Linear(i_d, o_d) if i_d != o_d else nn.Identity()
    def forward(self, x): return F.relu(self.l(x) + self.s(x))


class TemporalEncoder(nn.Module):
    """Lightweight per-node temporal encoder (STCL candidate replacement)."""

    def __init__(self, P: int, d: int, N: int):
        super().__init__()
        self.P, self.d, self.N = P, d, N
        self.proj = nn.Linear(P, d)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # W: (B, P, N) -> Z: (B, N, d)
        B, P, N = W.shape
        x = W.permute(0, 2, 1).reshape(B * N, P)
        z = F.gelu(self.proj(x)).reshape(B, N, self.d)
        return z


class STCL_Core(nn.Module):
    def __init__(self, P, d, nl):
        super().__init__()
        self.b = nn.ModuleList([ResidualBlock(P, d)] + [ResidualBlock(d, d) for _ in range(nl - 1)])
        self.g = nn.Parameter(torch.ones(nl))
        self.m = nn.Sequential(nn.Linear(2*d,128), nn.GELU(), nn.Linear(128,4))
    def forward(self, x):
        B = x.shape[0]; A_a = x.new_zeros(B,2,2); r_in = x.permute(0,2,1).contiguous(); R_layers=[]
        for q,blk in enumerate(self.b):
            Rq = blk(r_in); Aq = self.m(Rq.view(B,-1)).view(B,2,2)
            R_layers.append(Aq @ Rq * self.g[q]); A_a += self.g[q]*Aq; r_in = Rq
        return {"R_layers": R_layers, "A_agg": A_a}
    def time_causality_weight(self) -> torch.Tensor:
        """
        返回 TCL 第一层线性层的权重 (d, P)，用于时间因果可视化。
        """
        return self.b[0].l.weight  # shape: (d, P)

class TorchCausalStateEncoder(nn.Module):
    """
    Corrected and optimized version of TorchCausalStateEncoder.
    - Parallelized forward pass and explainability methods.
    - Corrected aggregation logic in the forward pass.
    """
    def __init__(self, A, P, d, Q, temp: float = 3.0, self_gate: float = 0.3):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.d = d
        self.core = STCL_Core(P, d, Q)
        self.temp, self.self_gate = float(temp), float(self_gate)

        edges = np.array([(j, i) for i, row in enumerate(A) for j, val in enumerate(row) if val == 1 and i != j], dtype=np.int64).T
        if edges.size == 0:
            self.register_buffer('edge_index', torch.empty((2, 0), dtype=torch.long))
        else:
            self.register_buffer('edge_index', torch.from_numpy(edges))

        deg = A.sum(axis=1).astype(np.float32)
        self.register_buffer('deg', torch.from_numpy(np.maximum(deg, 1e-6)))

    def forward(self, Wb):
        B, P, N = Wb.shape
        
        # 1. Process self-dynamics (parallel)
        self_pairs = Wb.unsqueeze(-1).repeat(1, 1, 1, 2)
        self_pairs_batched = self_pairs.permute(0, 2, 1, 3).reshape(B * N, P, 2)
        
        out_self = self.core(self_pairs_batched)
        R_last_self, A_agg_self = out_self['R_layers'][-1], out_self['A_agg']
        
        Ri = F.normalize(R_last_self[:, 0, :], dim=1)
        
        a_self_scalar = 0.5 * (A_agg_self[:, 0, 0] + A_agg_self[:, 1, 1])
        a_self = torch.tanh(self.temp * a_self_scalar).unsqueeze(1)

        causal = (self.self_gate * a_self * Ri).view(B, N, self.d)

        if self.edge_index.shape[1] == 0:
            return causal / (self.deg + self.self_gate).view(1, -1, 1)

        # 2. Process cross-node dynamics (parallel)
        source_nodes_idx, target_nodes_idx = self.edge_index[0], self.edge_index[1]
        
        Wb_j = Wb[:, :, source_nodes_idx]
        Wb_i = Wb[:, :, target_nodes_idx]
        
        edge_pairs = torch.stack([Wb_i, Wb_j], dim=-1)
        edge_pairs_batched = edge_pairs.permute(0, 2, 1, 3).reshape(B * self.edge_index.shape[1], P, 2)
        
        out_edge = self.core(edge_pairs_batched)
        R_last_edge, A_agg_edge = out_edge['R_layers'][-1], out_edge['A_agg']

        Rj = F.normalize(R_last_edge[:, 1, :], dim=1)
        a21 = torch.tanh(self.temp * A_agg_edge[:, 1, 0]).unsqueeze(1)

        messages = (a21 * Rj).view(B, self.edge_index.shape[1], self.d)
        
        # --- Scatter / Aggregate (Corrected) ---
        causal_cross = torch.zeros_like(causal)
        for b_idx in range(B):
            causal_cross[b_idx].index_add_(0, target_nodes_idx, messages[b_idx])
        
        causal = causal + causal_cross

        return causal / (self.deg + self.self_gate).view(1, -1, 1)

    @torch.no_grad()
    def estimate_spatial_causality_matrix(self, W: np.ndarray, batch_size: int = 256, 
                                        return_series: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        估计空间因果矩阵。
        
        新增 (MOD):
        :param return_series: 若为 True，则额外返回 (B, num_edges) 的动态 a21 序列。
        :return: 
            S: (N, N) 平均因果矩阵
            all_strengths_tensor (optional): (B, num_edges) 动态 a21 序列
        """
        self.eval()
        device = next(self.parameters()).device
        N = W.shape[2]
        S = np.zeros((N, N), dtype=np.float32)

        if self.edge_index.shape[1] == 0:
            if return_series:
                return S, np.empty((W.shape[0], 0), dtype=np.float32)
            return S
        
        source_nodes_idx, target_nodes_idx = self.edge_index[0], self.edge_index[1]
        all_strengths_list = [] # <-- MOD: 名字从 all_strengths 改为 all_strengths_list

        for k in range(0, W.shape[0], batch_size):
            Wb = torch.from_numpy(W[k:k+batch_size]).to(device)
            B = Wb.shape[0]

            Wb_j = Wb[:, :, source_nodes_idx]
            Wb_i = Wb[:, :, target_nodes_idx]
        
            edge_pairs = torch.stack([Wb_i, Wb_j], dim=-1)
            edge_pairs_batched = edge_pairs.permute(0, 2, 1, 3).reshape(B * self.edge_index.shape[1], -1, 2)
            
            out_edge = self.core(edge_pairs_batched)
            A_agg_edge = out_edge['A_agg']
            
            a21 = torch.tanh(self.temp * A_agg_edge[:, 1, 0])
            all_strengths_list.append(a21.view(B, -1).detach().cpu()) # <-- MOD: 使用 .append()

        # --- MOD: 在 .mean() 之前保存完整的序列 ---
        all_strengths_tensor = torch.cat(all_strengths_list, dim=0) # (B, num_edges)
        
        # 计算平均值 S 矩阵 (这部分不变)
        avg_strengths = all_strengths_tensor.mean(dim=0).numpy() # (num_edges,)
        S[target_nodes_idx.cpu().numpy(), source_nodes_idx.cpu().numpy()] = avg_strengths
        
        if return_series:
            # S 是 (N,N) numpy 矩阵
            # all_strengths_tensor 是 (B, num_edges) numpy 矩阵
            return S, all_strengths_tensor.numpy()
        else:
            return S

class PretrainableEncoderMTL(nn.Module):
    def __init__(self, A: np.ndarray, P: int, H: int, N: int, d_model: int = 64, Q: int = 2, recon_hidden: int = 64):
        super().__init__()
        self.encoder = TorchCausalStateEncoder(A, P, d_model, Q)
        self.head_y  = nn.Linear(d_model, H); self.head_dy = nn.Linear(d_model, H)
        self.head_r  = nn.Linear(d_model, H)
        self.recon = nn.Sequential(nn.Linear(d_model, recon_hidden), nn.GELU(), nn.Linear(recon_hidden, 1))
        # learned log variances
        self.log_sigma_y = nn.Parameter(torch.zeros(1))
        self.log_sigma_dy = nn.Parameter(torch.zeros(1))
        self.log_sigma_r = nn.Parameter(torch.zeros(1))
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        if self.training and mask_ratio > 0:
            K=max(1,int(math.ceil(x.shape[1]*mask_ratio)))
            x_m=x.clone(); x_m[:,-K:,:]=0.0; causal=self.encoder(x_m)
            target_recon=x[:,-K:,:].mean(dim=1); recon_pred=self.recon(causal).squeeze(-1)
        else:
            causal=self.encoder(x); recon_pred, target_recon = None, None
        y_p = self.head_y(causal).permute(0,2,1)
        dy_p= self.head_dy(causal).permute(0,2,1)
        r_p = self.head_r(causal).permute(0,2,1)
        return (y_p, dy_p, r_p, recon_pred, target_recon)


def mtl_pretrain_trainer(model: PretrainableEncoderMTL,
                         W_tr, y_tr, W_vl, y_vl, yhat_tr_bl, yhat_vl_bl,
                         mask_ratio=0.15, lr=1e-3, wd=1e-4, max_epochs=80, patience=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MTL pretrainer using device: {device}")
    model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # 训练集 DataLoader (这部分不变)
    R_tr = y_tr - yhat_tr_bl
    tr_ds = torch.utils.data.TensorDataset(*(torch.from_numpy(d) for d in [W_tr, y_tr, R_tr]))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    
    # ==================== 修改点 1: 创建验证集 DataLoader ====================
    # 注意：验证时 shuffle=False，batch_size 可以适当增大以加快速度
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(W_vl), torch.from_numpy(y_vl))
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    # =========================================================================

    best_mae, bad, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs+1):
        model.train(); total_loss = 0.0
        # 训练循环 (这部分不变)
        for wb, yb, Rb in tr_dl:
            wb, yb, Rb = wb.to(device), yb.to(device), Rb.to(device)
            opt.zero_grad()
            y_p, dy_p, r_p, rec_p, rec_t = model(wb, mask_ratio=mask_ratio)
            last_val_wb = wb[:, -1:, :]
            dy_target = yb - last_val_wb
            Ly  = F.mse_loss(y_p, yb)
            Ldy = F.mse_loss(dy_p, dy_target)
            Lr  = F.l1_loss(r_p, Rb)
            Lrec = F.mse_loss(rec_p, rec_t) if rec_p is not None else torch.tensor(0.0, device=device)
            # uncertainty weighting
            loss = (
                torch.exp(-model.log_sigma_y)  * Ly   * 0.5 + model.log_sigma_y * 0.5 +
                torch.exp(-model.log_sigma_dy) * Ldy  * 0.5 + model.log_sigma_dy * 0.5 +
                torch.exp(-model.log_sigma_r)  * Lr   * 0.5 + model.log_sigma_r * 0.5 +
                torch.exp(-model.log_sigma_recon) * Lrec * 0.5 + model.log_sigma_recon * 0.5
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += float(loss.item())

        # ==================== 修改点 2: 分批进行验证 ====================
        model.eval()
        total_val_error = 0.0
        with torch.no_grad():
            for Wb_v, yb_v in val_dl: # <-- 使用 val_dl 进行循环
                Wb_v, yb_v = Wb_v.to(device), yb_v.to(device)
                yv_p, _, _, _, _ = model(Wb_v, mask_ratio=0.0) # <-- 一次只处理一个批次
                
                # 使用 reduction='sum' 累加每个批次的误差总和，以确保计算准确
                total_val_error += F.l1_loss(yv_p, yb_v, reduction='sum').item()

        # 计算整个验证集的平均 MAE
        mae_v_y = total_val_error / len(val_ds)
        # =============================================================
            
        print(f"MTL Epoch {epoch:03d}: TrainLoss={total_loss/len(tr_dl):.4f} | Val MAE(y)={mae_v_y:.4f}")
        
        # Early stopping 逻辑 (这部分不变)
        if mae_v_y < best_mae:
            best_mae, bad, best_state = mae_v_y, 0, deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: print(f"Early stopping at epoch {epoch}."); break
            
    if best_state: model.load_state_dict(best_state)

# =========================
# Hybrid Corrector (ridge + shared MLP + tau + blending)
# =========================
def encode_with_encoder(encoder: nn.Module, W: np.ndarray, device: torch.device) -> np.ndarray:
    encoder.eval(); Z_list = []
    with torch.no_grad():
        for b in torch.from_numpy(W).split(512):
            Z_list.append(encoder(b.to(device)).cpu().numpy())
    return np.concatenate(Z_list, axis=0)  # (B,N,d)

def ridge_probe_val(Z_tr, R_tr, Z_vl, yhat_vl, y_vl, l2=1e-2):
    """Per-node ridge for all horizons: Zi_tr:(B,d), Ri_tr:(B,H) -> w:(d,H)"""
    B_tr, H, N = R_tr.shape; _, _, d = Z_tr.shape
    W_list = []; rhat_vl = np.zeros_like(y_vl)
    for i in range(N):
        Zi_tr, Zi_vl = Z_tr[:, i, :], Z_vl[:, i, :]
        Ri_tr = R_tr[:, :, i]  # (B,H)
        A = Zi_tr.T @ Zi_tr + l2 * np.eye(d, dtype=Zi_tr.dtype)      # (d,d)
        b = Zi_tr.T @ Ri_tr                                          # (d,H)
        w = np.linalg.solve(A, b)                                    # (d,H)
        W_list.append(w); rhat_vl[:, :, i] = Zi_vl @ w               # (B,H)
    return mae(y_vl, yhat_vl + rhat_vl), W_list

class SmallMLP(nn.Module):
    def __init__(self, d_in, H, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, H))
    def forward(self, Z): return self.net(Z)  # (B,H)

class HybridTrainer:
    def __init__(self, encoder, A, F_mu, F_sd, H: int, N: int, P: int,
                 l2_delta=1e-4, lr=1e-4, wd=1e-4, patience=20, max_epochs=60, batch_size=64,
                 node_emb_dim: int = 16, use_dir_gate: bool = True,
                 use_blend_search: bool = True,
                 quantile_q: float = 0.80, improve_eps: float = 0.01,
                 scales: List[int] = [6,12,24,48],
                 freq_bands: List[Tuple[float,float]] = [(0.0,0.05),(0.05,0.15),(0.15,0.35)],
                 use_robust: bool = True, top_m: Optional[int] = None,
                 prior_mode: str = "full"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = deepcopy(encoder).to(self.device).eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.A, self.F_mu, self.F_sd, self.H, self.N, self.P = A, F_mu, F_sd, H, N, P
        self.l2_delta, self.lr, self.wd = l2_delta, lr, wd
        self.patience, self.max_epochs, self.batch_size = patience, max_epochs, batch_size
        self.use_dir_gate = use_dir_gate
        self.use_blend_search = use_blend_search
        self.quantile_q, self.improve_eps = quantile_q, improve_eps
        self.scales, self.freq_bands, self.use_robust, self.top_m = scales, freq_bands, use_robust, top_m
        self.prior_mode = prior_mode
        self.W_list_t: List[torch.Tensor] = []; self.tau_map = None   # (H,N)
        self.W_blend = None  # (N,H,2)
        self.linear_is_best = None
        self.node_emb = nn.Embedding(num_embeddings=N, embedding_dim=node_emb_dim).to(self.device)
        self.shared_mlp = SmallMLP(d_in=1, H=H)  # placeholder; reset later

    # ---- feature encoding ----
    def _build_priors(self, W, R_train=None, mu=None, sd=None):
        if self.prior_mode == "minimal":
            return build_physics_priors_minimal(
                W, self.A, use_robust=self.use_robust, mu=mu, sd=sd,
            )
        return build_physics_priors_ms(
            W, self.A, scales=self.scales, freq_bands=self.freq_bands,
            use_robust=self.use_robust, top_m=self.top_m,
            R_train=R_train, mu=mu, sd=sd,
        )

    def _encode_aug(self, W):
        Z = encode_with_encoder(self.encoder, W, self.device)  # (B,N,d)
        F_aug, _, _, _ = self._build_priors(W, mu=self.F_mu, sd=self.F_sd)
        return np.concatenate([Z, F_aug], axis=-1)  # (B,N,d+f)

    def _init_shared(self, d_aug: int, node_emb_dim: int):
        self.shared_mlp = SmallMLP(d_in=d_aug + node_emb_dim, H=self.H, h=128).to(self.device)

    # ---- residual pieces ----
    def _r_linear(self, Z: torch.Tensor):
        B = Z.shape[0]; r_lin = torch.zeros(B, self.H, self.N, device=self.device)
        for i in range(self.N): r_lin[:, :, i] = Z[:, i, :] @ self.W_list_t[i]  # (B,H)
        return r_lin

    def _r_delta(self, Z: torch.Tensor):
        B = Z.shape[0]; r_delta = torch.zeros(B, self.H, self.N, device=self.device)
        idx = torch.arange(self.N, device=self.device); emb = self.node_emb(idx)  # (N,d_e)
        for i in range(self.N):
            Zi = torch.cat([Z[:, i, :], emb[i].expand(B, -1)], dim=-1)
            r_delta[:, :, i] = self.shared_mlp(Zi)  # (B,H)
        return r_delta

    # ---- safety gate ----
    def _apply_dir_gate(self, r_delta: torch.Tensor, yb: torch.Tensor, yh_b: torch.Tensor,
                        r_lin: torch.Tensor, tau: torch.Tensor):
        if not self.use_dir_gate:
            return torch.clamp(r_delta, -tau, tau)
        err = yb - (yh_b + r_lin)
        same_sign = torch.sign(r_delta) * torch.sign(err) >= 0
        gated = torch.where(same_sign | (torch.abs(r_delta) <= 0.5*tau),
                            r_delta, torch.zeros_like(r_delta))
        return torch.clamp(gated, -tau, tau)

    def train(self, W_tr, y_tr, yhat_tr, W_vl, y_vl, yhat_vl):
        # 1) 仅使用训练数据计算和设置先验统计量 (mu/sd)
        # 这一步是后续特征标准化的基础，必须首先进行。
        _, F_mu, F_sd, _ = self._build_priors(
            W_tr, R_train=(y_tr - yhat_tr), mu=None, sd=None,
        )
        self.F_mu, self.F_sd = F_mu.astype(np.float32), F_sd.astype(np.float32)

        # ==================== 优化核心：特征预计算与缓存 ====================
        # 在这里一次性计算好训练集和验证集的所有增强特征，避免后续重复计算。
        print("[HybridTrainer] Pre-calculating and caching all augmented features for train and validation sets...")
        Z_tr_aug = self._encode_aug(W_tr)
        Z_vl_aug = self._encode_aug(W_vl)
        print(f"[HybridTrainer] Feature encoding complete. Train features shape: {Z_tr_aug.shape}, Val features shape: {Z_vl_aug.shape}")
        # ==================================================================

        # 2) 初始化共享MLP的维度
        d_aug = Z_tr_aug.shape[2]
        self._init_shared(d_aug=d_aug, node_emb_dim=self.node_emb.embedding_dim)

        # 3) 训练线性基底 (Ridge Regression Floor)
        # 直接使用已经缓存的 Z_tr_aug 和 Z_vl_aug
        R_tr = y_tr - yhat_tr
        val_mae_lin, W_list = ridge_probe_val(Z_tr_aug, R_tr, Z_vl_aug, yhat_vl, y_vl, l2=1e-2)
        self.W_list_t = [torch.from_numpy(w.astype(np.float32)).to(self.device) for w in W_list]
        print(f"[Hybrid] Linear-only Val MAE = {val_mae_lin:.4f}")

        # 4) 计算安全边界 tau
        # 将缓存的 Z_vl_aug 转换为 tensor
        Zvl_t = torch.from_numpy(Z_vl_aug).to(self.device)
        yvl_t = torch.from_numpy(y_vl).to(self.device)
        yhvl_t = torch.from_numpy(yhat_vl).to(self.device)
        with torch.no_grad():
            r_lin_v = self._r_linear(Zvl_t)
            abs_res = torch.abs(yvl_t - (yhvl_t + r_lin_v))
        tau_np = np.quantile(abs_res.cpu().numpy(), self.quantile_q, axis=0)
        self.tau_map = torch.from_numpy(tau_np.astype(np.float32)).to(self.device)

        # 5) 训练共享的非线性修正 MLP
        # 直接使用缓存的 Z_tr_aug 创建数据加载器
        tr_ds = torch.utils.data.TensorDataset(*(torch.from_numpy(d) for d in [Z_tr_aug, y_tr, yhat_tr]))
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        
        opt = torch.optim.Adam(self.shared_mlp.parameters(), lr=self.lr, weight_decay=self.wd)
        best_mae, best_state, bad_epochs = val_mae_lin, deepcopy(self.shared_mlp.state_dict()), 0
        self.linear_is_best = torch.ones(self.N, self.H, dtype=torch.bool, device=self.device)

        for ep in range(1, self.max_epochs + 1):
            self.shared_mlp.train()
            for Zb, yb, yh_b in tr_dl:
                Zb, yb, yh_b = Zb.to(self.device), yb.to(self.device), yh_b.to(self.device)
                opt.zero_grad()
                r_lin = self._r_linear(Zb.detach())
                r_delta = self._r_delta(Zb)
                tau = self.tau_map.unsqueeze(0).expand_as(r_delta)
                r_delta_c = self._apply_dir_gate(r_delta, yb, yh_b, r_lin, tau)
                y_f = yh_b.detach() + r_lin + r_delta_c
                loss = F.l1_loss(y_f, yb) + self.l2_delta * (r_delta_c**2).mean()
                loss.backward(); opt.step()

            # 在验证集上进行评估 (直接使用已经创建的 Zvl_t)
            self.shared_mlp.eval()
            with torch.no_grad():
                r_lin_v = self._r_linear(Zvl_t)
                r_delta_v = self._r_delta(Zvl_t)
                tau_v = self.tau_map.unsqueeze(0).expand_as(r_delta_v)
                r_delta_v = torch.clamp(r_delta_v, -tau_v, tau_v)
                y_lin = yhvl_t + r_lin_v
                y_hyb = y_lin + r_delta_v
                err_lin = torch.mean(torch.abs(yvl_t - y_lin), dim=0)
                err_hyb = torch.mean(torch.abs(yvl_t - y_hyb), dim=0)
                better_hyb = (err_hyb + 1e-8) < (err_lin * (1 - self.improve_eps))
                self.linear_is_best = (~better_hyb).permute(1,0).contiguous()
                mae_v_mix = torch.mean(torch.where(better_hyb, err_hyb, err_lin)).item()
            print(f"[Hybrid] Epoch {ep:03d}: Val MAE (mixed)={mae_v_mix:.4f}")

            if mae_v_mix < best_mae:
                best_mae, best_state, bad_epochs = mae_v_mix, deepcopy(self.shared_mlp.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {ep}."); break
        
        if best_state: self.shared_mlp.load_state_dict(best_state)

        # 6) 融合权重校准或固定权重
        if not self.use_blend_search:
            self.W_blend = torch.ones(self.N, self.H, 2, device=self.device)
            print("[Hybrid] Fixed blend weights w1=w2=1.0 (no grid search).")
        else:
            print("[Hybrid] Calibrating per-(node,h) blending weights ...")
            with torch.no_grad():
                r_lin_v = self._r_linear(Zvl_t)
                r_delta_v = self._r_delta(Zvl_t)
                tau_v = self.tau_map.unsqueeze(0).expand_as(r_delta_v)
                r_delta_v = torch.clamp(r_delta_v, -tau_v, tau_v)
                base = yhvl_t; y_true = yvl_t
                W_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
                self.W_blend = torch.zeros(self.N, self.H, 2, device=self.device)
                base_mae_map = torch.mean(torch.abs(y_true - base), dim=0)

                for i in range(self.N):
                    for h in range(self.H):
                        best_mae_ih = base_mae_map[h,i].item(); best_w = (0.0, 0.0)
                        for w1 in W_grid:
                            for w2 in W_grid:
                                y_hat = base[:,h,i] + w1 * r_lin_v[:,h,i] + w2 * r_delta_v[:,h,i]
                                m = torch.mean(torch.abs(y_true[:,h,i] - y_hat)).item()
                                if m < best_mae_ih - 1e-12: best_mae_ih, best_w = m, (w1, w2)
                        if best_mae_ih > base_mae_map[h,i].item() * (1 - self.improve_eps):
                            best_w = (0.0, 0.0)
                        self.W_blend[i,h,0], self.W_blend[i,h,1] = best_w[0], best_w[1]
            print("[Hybrid] Blending calibration done.")

    # ---- inference ----
    def predict(
        self,
        W: np.ndarray,
        yhat: np.ndarray,
        return_nl_abs: bool = False,
        return_clip_mask: bool = False,
    ):
        Z_aug = self._encode_aug(W)
        Zt = torch.from_numpy(Z_aug).to(self.device)
        yh_t = torch.from_numpy(yhat).to(self.device)
        with torch.no_grad():
            r_lin = self._r_linear(Zt)
            r_delta = self._r_delta(Zt)
            tau = self.tau_map.unsqueeze(0).expand_as(r_delta)
            clip_mask = torch.abs(r_delta) > tau
            r_delta_c = torch.clamp(r_delta, -tau, tau)
            w1 = self.W_blend[:,:,0].permute(1,0).unsqueeze(0)  # (1,H,N)
            w2 = self.W_blend[:,:,1].permute(1,0).unsqueeze(0)
            nl = w2 * r_delta_c
            y_final = yh_t + w1 * r_lin + nl
        y_out = y_final.cpu().numpy()
        if return_nl_abs and return_clip_mask:
            return y_out, torch.abs(nl).cpu().numpy(), clip_mask.cpu().numpy()
        if return_nl_abs:
            return y_out, torch.abs(nl).cpu().numpy()
        if return_clip_mask:
            return y_out, clip_mask.cpu().numpy()
        return y_out


# =========================
# CRCConfig + ResidualCorrector (Phase 1 wrapper)
# =========================
CRC_CHECKPOINT_VERSION = "crc-slim-v1"


def q_val_to_quantile(q_val: int) -> float:
    """Legacy mapping from integer q_val to tau quantile."""
    q_int = int(q_val)
    if q_int <= 1:
        return 0.10
    if q_int == 2:
        return 0.30
    if q_int == 3:
        return 0.50
    if q_int == 4:
        return 0.70
    return 0.90


def resolve_crc_hparams(args) -> Tuple[int, float]:
    """Resolve STCL layers and tau quantile from new CLI or legacy q_val."""
    q_legacy = int(getattr(args, "q_val", 3))
    enc_layers = getattr(args, "crc_encoder_layers", None)
    tau_quantile = getattr(args, "crc_tau_quantile", None)
    if enc_layers is None:
        enc_layers = q_legacy
    if tau_quantile is None:
        tau_quantile = q_val_to_quantile(q_legacy)
    return int(enc_layers), float(tau_quantile)


@dataclass
class CRCConfig:
    """Feature flags default to legacy (all-on) behavior."""
    use_mtl: bool = True
    use_physics_priors: bool = True
    use_dir_gate: bool = True
    use_blend_search: bool = True
    use_stcl_encoder: bool = True
    minimal_priors: bool = False
    tau_quantile: float = 0.5
    encoder_layers: int = 3
    graph_topk: int = 3

    @classmethod
    def from_args(cls, args) -> "CRCConfig":
        enc_layers, tau_q = resolve_crc_hparams(args)
        def _flag(name: str, default: bool = True) -> bool:
            v = getattr(args, name, 1 if default else 0)
            return bool(int(v))
        return cls(
            use_mtl=_flag("crc_use_mtl"),
            use_physics_priors=_flag("crc_use_priors"),
            use_dir_gate=_flag("crc_use_dir_gate"),
            use_blend_search=_flag("crc_use_blend"),
            use_stcl_encoder=_flag("crc_use_stcl"),
            minimal_priors=_flag("crc_use_minimal_priors", default=False),
            tau_quantile=tau_q,
            encoder_layers=enc_layers,
            graph_topk=int(getattr(args, "top_k", 3)),
        )


class ResidualCorrector:
    """
    Phase-A wrapper: orchestrates MTL pretrain + HybridTrainer with CRCConfig.
    Behavior matches legacy HybridTrainer path when all flags are default-on.
    """
    VERSION = CRC_CHECKPOINT_VERSION

    def __init__(
        self,
        config: CRCConfig,
        A: np.ndarray,
        H: int,
        N: int,
        P: int,
        d_model: int,
        enc_in: int,
        *,
        F_mu: Optional[np.ndarray] = None,
        F_sd: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.A = A
        self.H, self.N, self.P = H, N, P
        self.d_model = d_model
        self.enc_in = enc_in
        self.F_mu = F_mu
        self.F_sd = F_sd
        self._hybrid: Optional[HybridTrainer] = None

    @classmethod
    def from_legacy_hybrid(cls, hybrid: HybridTrainer, config: Optional[CRCConfig] = None) -> "ResidualCorrector":
        obj = cls.__new__(cls)
        obj.config = config or CRCConfig()
        obj._hybrid = hybrid
        obj.A = hybrid.A
        obj.H, obj.N, obj.P = hybrid.H, hybrid.N, hybrid.P
        obj.d_model = hybrid.encoder.core.d if hasattr(hybrid.encoder, "core") else 64
        obj.enc_in = hybrid.N
        obj.F_mu, obj.F_sd = hybrid.F_mu, hybrid.F_sd
        return obj

    @property
    def hybrid(self) -> HybridTrainer:
        if self._hybrid is None:
            raise RuntimeError("ResidualCorrector not trained or loaded.")
        return self._hybrid

    def train(
        self,
        W_tr: np.ndarray,
        y_tr: np.ndarray,
        yhat_tr: np.ndarray,
        W_vl: np.ndarray,
        y_vl: np.ndarray,
        yhat_vl: np.ndarray,
    ) -> None:
        cfg = self.config
        if cfg.use_stcl_encoder:
            encoder = None
            if cfg.use_mtl:
                print("\n--- Stage A: MTL Pre-training ---")
                pre_encoder = PretrainableEncoderMTL(
                    self.A, self.P, self.H, self.enc_in,
                    self.d_model, cfg.encoder_layers,
                )
                mtl_pretrain_trainer(
                    pre_encoder,
                    W_tr, y_tr, W_vl, y_vl, yhat_tr, yhat_vl,
                    mask_ratio=0.0,
                )
                encoder = pre_encoder.encoder
            else:
                print("\n--- Stage A: MTL Pre-training SKIPPED (crc_use_mtl=0) ---")
                pre_encoder = PretrainableEncoderMTL(
                    self.A, self.P, self.H, self.enc_in,
                    self.d_model, cfg.encoder_layers,
                )
                encoder = pre_encoder.encoder
        else:
            print("\n--- Stage A: TemporalEncoder (no STCL, crc_use_stcl=0) ---")
            if cfg.use_mtl:
                print("[CRC] Note: MTL pretrain skipped with TemporalEncoder (A5 ablation).")
            encoder = TemporalEncoder(self.P, self.d_model, self.enc_in)

        prior_mode = "minimal" if cfg.minimal_priors else "full"
        print("\n--- Stage B: Hybrid Corrector ---")
        if cfg.use_physics_priors:
            if prior_mode == "minimal":
                F_tr, F_mu, F_sd, _ = build_physics_priors_minimal(W_tr, self.A)
                print(f"[Priors] Minimal priors dim = {F_tr.shape[-1]} (last, diff1, ema_12)")
            else:
                F_tr, F_mu, F_sd, _ = build_physics_priors_ms(
                    W_tr, self.A,
                    scales=[6, 12, 24, 48],
                    freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
                    use_robust=True,
                    top_m=None,
                    R_train=(y_tr - yhat_tr),
                    mu=None, sd=None,
                )
                print(f"[Priors] 多尺度先验维度 = {F_tr.shape[-1]}")
            self.F_mu, self.F_sd = F_mu, F_sd
        else:
            raise NotImplementedError("Disabling all priors is not supported.")

        print(f"[Hybrid] Using safety quantile_q={cfg.tau_quantile:.2f} "
              f"(encoder_layers={cfg.encoder_layers})")

        hybrid = HybridTrainer(
            encoder, self.A, F_mu=self.F_mu, F_sd=self.F_sd,
            H=self.H, N=self.N, P=self.P,
            node_emb_dim=16, use_dir_gate=cfg.use_dir_gate,
            use_blend_search=cfg.use_blend_search,
            quantile_q=cfg.tau_quantile, improve_eps=0.01,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True, top_m=None,
            prior_mode=prior_mode,
        )
        hybrid.train(W_tr, y_tr, yhat_tr, W_vl, y_vl, yhat_vl)
        self._hybrid = hybrid

    def predict(self, W: np.ndarray, yhat: np.ndarray, **kwargs):
        return self.hybrid.predict(W, yhat, **kwargs)


def save_crc_checkpoint(path: str, corrector: ResidualCorrector) -> None:
    payload = {
        "version": ResidualCorrector.VERSION,
        "config": corrector.config,
        "hybrid": corrector.hybrid,
    }
    torch.save(payload, path)


def load_crc_checkpoint(path: str, map_location=None) -> ResidualCorrector:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=map_location)

    if isinstance(obj, ResidualCorrector):
        return obj
    if isinstance(obj, HybridTrainer):
        print(f"[CRC] Loaded legacy HybridTrainer checkpoint from {path}")
        return ResidualCorrector.from_legacy_hybrid(obj)
    if isinstance(obj, dict):
        if "hybrid" in obj and isinstance(obj["hybrid"], HybridTrainer):
            cfg = obj.get("config")
            if cfg is None:
                cfg = CRCConfig()
            elif isinstance(cfg, dict):
                cfg = CRCConfig(**cfg)
            corrector = ResidualCorrector.from_legacy_hybrid(obj["hybrid"], cfg)
            ver = obj.get("version", "unknown")
            print(f"[CRC] Loaded checkpoint version={ver} from {path}")
            return corrector
    raise ValueError(
        f"Unrecognized CRC checkpoint format at {path}. "
        "Expected ResidualCorrector, HybridTrainer, or versioned dict."
    )
