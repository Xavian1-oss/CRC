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


warnings.filterwarnings('ignore')


def quick_val_series_from_windows(W_val):
    """Last timestep per window: (B, N)."""
    return W_val[:, -1, :]


class Exp_CRC(Exp_Long_Term_Forecast):
    """Two-stage CRC: train baseline, then hybrid corrector on baseline outputs."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.baseline_model = None
        self.hybrid_corrector = None

    def _build_model(self):
        return None

    def _get_predictions(self, model, data_loader):
        """Return preds (B,H,N_sel), trues (B,H,N_sel), inputs (B,P,N_all)."""
        model.eval()
        all_preds, all_trues, all_inputs = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                all_preds.append(outputs.detach().cpu().numpy())
                all_trues.append(batch_y.detach().cpu().numpy())
                all_inputs.append(batch_x.detach().cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        return preds, trues, inputs

    def train(self, setting):
        print(">>>>>>> CRC two-stage training <<<<<<<")

        print("\n" + "=" * 50)
        print("--- Stage 1: baseline forecaster ---")
        print("=" * 50)

        baseline_args = deepcopy(self.args)
        baseline_args.model = self.args.baseline_model

        if getattr(self.args, 'data', None) != 'custom':
            dataset_tag = self.args.data
        else:
            data_path = getattr(self.args, 'data_path', '')
            if data_path:
                dataset_tag = os.path.splitext(os.path.basename(data_path))[0]
            else:
                root_path = getattr(self.args, 'root_path', '')
                dataset_tag = os.path.basename(os.path.normpath(root_path)) if root_path else 'custom'

        baseline_setting = (
            f"baseline_{self.args.baseline_model}_{dataset_tag}"
            f"_pl{self.args.pred_len}_dm{self.args.d_model}_enc{self.args.enc_in}"
        )
        baseline_ckpt_dir = os.path.join(self.args.checkpoints, baseline_setting)
        baseline_ckpt_path = os.path.join(baseline_ckpt_dir, 'checkpoint.pth')

        baseline_exp = Exp_Long_Term_Forecast(baseline_args)

        if getattr(self.args, 'warm_start', False) and os.path.exists(baseline_ckpt_path):
            print(f"[WarmStart] Loading baseline checkpoint: {baseline_ckpt_path}")
            ckpt = torch.load(baseline_ckpt_path, map_location=self.device)
            try:
                baseline_exp.model.load_state_dict(ckpt)
                self.baseline_model = baseline_exp.model.to(self.device)
                print("[WarmStart] Baseline loaded.")
            except RuntimeError as e:
                msg = str(e)
                if 'enc_embedding.temporal_embedding.embed' in msg:
                    baseline_exp.model.load_state_dict(ckpt, strict=False)
                    self.baseline_model = baseline_exp.model.to(self.device)
                    print("[WarmStart] Baseline loaded (strict=False).")
                else:
                    print(f"[WarmStart] Load failed ({e}); retraining baseline.")
                    self.baseline_model = baseline_exp.train(baseline_setting)
                    print("Baseline training finished.")
        else:
            print(f"Training baseline: {baseline_args.model} (setting={baseline_setting})")
            self.baseline_model = baseline_exp.train(baseline_setting)
            print("Baseline training finished.")

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
            print("--- Stage 2: load hybrid corrector (skip training) ---")
            print("=" * 50)
            print(f"[CRC] Loading hybrid corrector from {crc_ckpt_path}")
            try:
                self.hybrid_corrector = torch.load(
                    crc_ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                self.hybrid_corrector = torch.load(crc_ckpt_path, map_location=self.device)
            print("[CRC] Stage 2 done (hybrid not retrained).")
            return self.hybrid_corrector

        print("\n" + "=" * 50)
        print("--- Stage 2: CRC hybrid corrector ---")
        print("=" * 50)

        train_set, train_loader = self._get_data(flag='train')
        val_set, val_loader = self._get_data(flag='val')

        print("Generating baseline predictions and windows on train/val...")
        yhat_tr_bl, y_tr_win, W_tr = self._get_predictions(self.baseline_model, train_loader)
        yhat_vl_bl, y_vl_win, W_vl = self._get_predictions(self.baseline_model, val_loader)

        print(f"DEBUG Train W:{W_tr.shape} y:{y_tr_win.shape} yhat:{yhat_tr_bl.shape}")
        print(f"DEBUG Val   W:{W_vl.shape} y:{y_vl_win.shape} yhat:{yhat_vl_bl.shape}")

        if val_ratio < 1.0:
            num_v = max(1, int(round(W_vl.shape[0] * val_ratio)))
            rng = np.random.RandomState(getattr(self.args, 'seed', 42))
            idx = rng.permutation(W_vl.shape[0])[:num_v]
            W_vl_sub = W_vl[idx]
            y_vl_sub = y_vl_win[idx]
            yhat_vl_sub = yhat_vl_bl[idx]
            print(f"[CRC] Hybrid val subset: ratio={val_ratio:.2f}, size={num_v}/{W_vl.shape[0]}")
        else:
            W_vl_sub, y_vl_sub, yhat_vl_sub = W_vl, y_vl_win, yhat_vl_bl

        raw_train_data = train_set.scaler.inverse_transform(W_tr.reshape(-1, W_tr.shape[-1]))
        A = build_corr_graph(raw_train_data, topk=self.args.top_k, thr=0.2).astype(np.float32)
        print(f"Graph built, density: {A.sum() / (A.shape[0] ** 2):.3f}")

        print("\n--- Stage A: MTL encoder pre-training ---")
        # q_val: STCL depth Q and mapped tau quantile (see quantile_q below)
        pre_encoder = PretrainableEncoderMTL(
            A, self.args.seq_len, self.args.pred_len, self.args.enc_in,
            self.args.d_model, self.args.q_val
        )
        mtl_pretrain_trainer(
            pre_encoder,
            W_tr, y_tr_win, W_vl, y_vl_win,
            yhat_tr_bl, yhat_vl_bl,
        )

        print("\n--- Stage B: hybrid corrector ---")
        F_tr, F_mu, F_sd, feat_names = build_physics_priors_ms(
            W_tr, A,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True,
            top_m=None,
            R_train=(y_tr_win - yhat_tr_bl),
            mu=None, sd=None
        )
        print(f"[Priors] feature dim = {F_tr.shape[-1]}")

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
        print(f"[Hybrid] safety quantile_q={quantile_q:.2f} (from q_val={q_int})")

        N_out = y_tr_win.shape[-1]
        self.hybrid_corrector = HybridTrainer(
            pre_encoder.encoder, A, F_mu=F_mu, F_sd=F_sd,
            H=self.args.pred_len, N=N_out, P=self.args.seq_len,
            node_emb_dim=16, use_dir_gate=True,
            quantile_q=quantile_q, improve_eps=0.01,
            scales=[6, 12, 24, 48],
            freq_bands=[(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
            use_robust=True,
            top_m=None
        )

        assert N_out == yhat_tr_bl.shape[-1], "HybridTrainer.N must match target channels"
        self.hybrid_corrector.train(W_tr, y_tr_win, yhat_tr_bl, W_vl_sub, y_vl_sub, yhat_vl_sub)

        torch.save(self.hybrid_corrector, crc_ckpt_path)
        print(f"[Checkpoint] Saved hybrid corrector to {crc_ckpt_path}")
        print("\n>>>>>>> CRC training complete <<<<<<<")
        return self.hybrid_corrector

    def _inverse_like_ltf(self, arr, ref_cols, scaler):
        """Inverse-transform (B,H,C_sel); tile channels when C_sel < ref_cols (MS)."""
        B, H, C_sel = arr.shape
        if C_sel == ref_cols:
            return scaler.inverse_transform(arr.reshape(B * H, C_sel)).reshape(B, H, C_sel)

        reps = int(np.ceil(ref_cols / max(C_sel, 1)))
        tiled = np.tile(arr, [1, 1, reps])[:, :, :ref_cols]
        inv_full = scaler.inverse_transform(tiled.reshape(B * H, ref_cols)).reshape(B, H, ref_cols)
        return inv_full[:, :, -C_sel:]

    def test(self, setting):
        print(">>>>>>> CRC test <<<<<<<")
        if self.hybrid_corrector is None or self.baseline_model is None:
            raise ValueError("Call train() before test().")

        test_set, test_loader = self._get_data(flag='test')

        print("Generating baseline predictions on test...")
        yhat_te_bl, y_te, W_te = self._get_predictions(self.baseline_model, test_loader)

        tau_scale = float(getattr(self.args, "tau_scale", 1.0))
        if hasattr(self.hybrid_corrector, "tau_scale"):
            self.hybrid_corrector.tau_scale = tau_scale

        print(f"Applying hybrid corrector (tau_scale={tau_scale:.3f})...")
        yfinal_te = self.hybrid_corrector.predict(W_te, yhat_te_bl)

        err_baseline = np.abs(y_te - yhat_te_bl)
        err_final = np.abs(y_te - yfinal_te)
        mae_baseline = err_baseline.mean(axis=0)
        mae_final = err_final.mean(axis=0)
        non_degraded_mask = mae_final <= (mae_baseline + 1e-8)
        paper_ndr = np.mean(non_degraded_mask) * 100.0

        if getattr(test_set, "scale", False) and getattr(self.args, "inverse", False):
            C_all = W_te.shape[-1]
            train_set, _ = self._get_data(flag='train')
            scaler = train_set.scaler
            y_te = self._inverse_like_ltf(y_te, C_all, scaler)
            yhat_te_bl = self._inverse_like_ltf(yhat_te_bl, C_all, scaler)
            yfinal_te = self._inverse_like_ltf(yfinal_te, C_all, scaler)
            try:
                W_te = scaler.inverse_transform(W_te.reshape(-1, C_all)).reshape(W_te.shape)
            except Exception:
                pass

        print("\n" + "=" * 50)
        print("--- Evaluation ---")
        print("=" * 50)
        base_mae, base_mse, _, _, _ = metric(yhat_te_bl, y_te)
        print(f"Baseline ({self.args.baseline_model}) -> MSE: {base_mse:.4f}, MAE: {base_mae:.4f}")

        final_mae, final_mse, _, _, _ = metric(yfinal_te, y_te)
        print(f"CRC corrected -> MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")

        if self.args.use_dtw:
            print("Computing DTW (flatten H*C)...")
            manhattan = lambda x, y: np.abs(x - y)
            dtw_list = []
            for i in range(yfinal_te.shape[0]):
                x = yfinal_te[i].reshape(-1, 1)
                y = y_te[i].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan)
                dtw_list.append(d)
            print(f"CRC corrected -> DTW: {np.mean(dtw_list):.4f}")

        print("-" * 50)
        mae_improve = (base_mae - final_mae) / max(base_mae, 1e-8) * 100
        mse_improve = (base_mse - final_mse) / max(base_mse, 1e-8) * 100
        print(f"MAE improvement (%): {mae_improve:.2f}")
        print(f"MSE improvement (%): {mse_improve:.2f}")
        print(f"NDR (%): {paper_ndr:.1f}")
        print("=" * 50)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'metrics.npy', np.array([final_mae, final_mse]))
        np.save(folder_path + 'pred.npy', yfinal_te)
        np.save(folder_path + 'true.npy', y_te)
        np.save(folder_path + 'pred_baseline.npy', yhat_te_bl)
        return
