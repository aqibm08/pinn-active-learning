"""
Sequential AL trainer for the PSA PINN.

"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _is_pinn(model) -> bool:
    return hasattr(model, 'loss_evidential') and hasattr(model, 'loss_pde')


class PSAActiveLearningTrainerV3:
    def __init__(
            self,
            model,
            data_selector,
            coll_selector,
            n0:               int   = 3,
            k_per_round:      int   = 2,
            budget:           int   = 10,
            epochs_per_round: int   = 300,
            n_final_epochs:   int   = 1500,
            lambda_pde_max:   float = 0.01,
            warmup_epochs:    int   = None,
            lam_evid:         float = 0.01,
            lr:               float = 1e-3,
            weight_decay:     float = 1e-5,
            device:           str   = 'cpu',
            verbose:          bool  = True,
            train_committee:  bool  = True,
    ):
        self.model            = model
        self.data_sel         = data_selector
        self.coll_sel         = coll_selector
        self.n0               = n0
        self.k_per_round      = k_per_round
        self.budget           = budget
        self.epochs_per_round = epochs_per_round
        self.n_final_epochs   = n_final_epochs
        self.lambda_pde_max   = lambda_pde_max
        self.warmup_epochs    = warmup_epochs if warmup_epochs is not None else epochs_per_round
        self.lam_evid         = lam_evid
        self.lr               = lr
        self.weight_decay     = weight_decay
        self.device           = device
        self.verbose          = verbose
        self.train_committee  = train_committee

        self.is_pinn = _is_pinn(model)

    # -- lambda_pde schedule --------------------------------------------------------
    def _lam_at(self, global_ep: int) -> float:
        if not self.is_pinn or self.lambda_pde_max <= 0.0:
            return 0.0
        if global_ep >= self.warmup_epochs:
            return self.lambda_pde_max
        return self.lambda_pde_max * (global_ep / max(1, self.warmup_epochs))

    # -- Validation ------------------------------------------------------------
    def _validate(self, x_val, y_val):
        self.model.eval()
        with torch.no_grad():
            pred = self.model.forward_deterministic(x_val)
            mse  = float(nn.functional.mse_loss(pred, y_val))
            mae  = float((pred - y_val).abs().mean())
            if self.is_pinn:
                from psa_pinn_model_v3 import evidential_loss
                g, nu, a, b = self.model.forward_evidential(x_val)
                vl  = float(evidential_loss(g, nu, a, b, y_val, lam=self.lam_evid))
                unc = float((b / (a * nu + 1e-8)).mean())
            else:
                vl, unc = mse, 0.0
            pmse = [float(nn.functional.mse_loss(pred[:, i], y_val[:, i])) for i in range(4)]
        self.model.train()
        return vl, mse, mae, unc, pmse

    # -- Training step ---------------------------------------------------------
    def _step(self, opt, x_lab, y_lab, x_coll, lam):
        self.model.train()
        opt.zero_grad()

        if self.is_pinn:
            # Evidential head loss
            d_evid = self.model.loss_evidential(x_lab, y_lab, lam_evid=self.lam_evid)
            # Deterministic committee MSE (separate parameters)
            d_det  = self.model.loss_deterministic_mse(x_lab, y_lab) if self.train_committee \
                     else torch.tensor(0.0, dtype=torch.float64, device=self.device)
            d_loss = d_evid + d_det
            # PDE loss
            if lam > 0.0 and len(x_coll) > 0:
                p_loss = self.model.loss_pde(x_coll)
                loss = d_loss + lam * p_loss
                p_val = float(p_loss.item())
            else:
                loss, p_val = d_loss, 0.0
        else:
            d_loss = self.model.loss_mse(x_lab, y_lab)
            loss   = d_loss
            p_val  = 0.0

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        opt.step()
        return float(d_loss.item()), p_val, float(loss.item())

    # -- Train block -----------------------------------------------------------
    def _train_block(self, opt, scheduler, x_lab, y_lab, x_coll,
                     n_ep, global_ep_start, x_val=None, y_val=None):
        h = {k: [] for k in ('train_data', 'train_pde', 'train_total',
                              'val_loss', 'val_mse', 'val_mae', 'uncertainty',
                              'lambda_pde')}
        h['val_mse_per_output'] = [[] for _ in range(4)]

        for local_ep in range(n_ep):
            global_ep = global_ep_start + local_ep
            lam = self._lam_at(global_ep)
            dl, pl, tl = self._step(opt, x_lab, y_lab, x_coll, lam)
            h['train_data'].append(dl); h['train_pde'].append(pl)
            h['train_total'].append(tl); h['lambda_pde'].append(lam)

            if x_val is not None and (local_ep % 5 == 0 or local_ep == n_ep - 1):
                vl, vmse, vmae, unc, pmse = self._validate(x_val, y_val)
                h['val_loss'].append(vl); h['val_mse'].append(vmse)
                h['val_mae'].append(vmae); h['uncertainty'].append(unc)
                for i in range(4): h['val_mse_per_output'][i].append(pmse[i])

            if scheduler is not None:
                scheduler.step()

            if self.verbose and (local_ep + 1) % 200 == 0:
                vs = f'  val_mse={h["val_mse"][-1]:.3e}' if h['val_mse'] else ''
                print(f'    ep {global_ep + 1:5d}  data={dl:.3e}  pde={pl:.3e}  lambda={lam:.4f}{vs}')
        return h

    # -- Main run --------------------------------------------------------------
    def run(
            self,
            pool_scenarios:   list,
            x_pool:           torch.Tensor,    # 102-dim (model accepts this)
            x_pool_orig:      torch.Tensor,    # same as x_pool in v3 (no compression)
            y_pool:           torch.Tensor,
            x_val:            torch.Tensor,
            y_val:            torch.Tensor,
            seed:             int = 42,
    ) -> dict:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        dev = self.device

        x_pool      = x_pool.double().to(dev)
        x_pool_orig = x_pool_orig.double().to(dev)
        y_pool      = y_pool.double().to(dev)
        x_val = x_val.double().to(dev); y_val = y_val.double().to(dev)

        opt = optim.Adam(self.model.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)

        rng = np.random.default_rng(seed)

        # -- Diverse seed selection (same as v1/v2) ---------------------------
        y1 = np.array([s['ic_summary'][0] for s in pool_scenarios])
        sort_idx = np.argsort(y1)
        sp = [pool_scenarios[i] for i in sort_idx]
        bsz = max(1, len(sp) // self.n0)
        seed_scens = []
        for b in range(self.n0):
            band = sp[b*bsz : (b+1)*bsz if b < self.n0 - 1 else len(sp)]
            seed_scens.append(band[int(rng.integers(0, len(band)))])
        seed_ids = {id(s) for s in seed_scens}
        pool_rem = [s for s in pool_scenarios if id(s) not in seed_ids]
        pool_rem = [pool_rem[i] for i in rng.permutation(len(pool_rem))]
        lab_scens = list(seed_scens)

        seed_idx = np.concatenate([s['indices'] for s in seed_scens]).tolist()
        x_lab = x_pool[seed_idx]; y_lab = y_pool[seed_idx]

        n_rounds = max(0, (self.budget - self.n0 + self.k_per_round - 1)
                       // self.k_per_round)
        total_epochs = self.epochs_per_round * (n_rounds + 1) + self.n_final_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_epochs, eta_min=1e-6)

        if self.verbose:
            mode = 'PINN+committee+PDE' if self.is_pinn else 'ANN'
            print(f'[v3 trainer] {mode}  budget={self.budget}  '
                  f'n0={self.n0}  k={self.k_per_round}  rounds={n_rounds}  '
                  f'total_ep={total_epochs}  lambda_max={self.lambda_pde_max}')

        # Histories
        dl_h=[]; pl_h=[]; tl_h=[]; lam_h=[]
        vl_h=[]; vm_h=[]; vma_h=[]; u_h=[]
        vmo_h={i: [] for i in range(4)}
        tsz_h=[]; csz_h=[]
        global_ep = 0

        for rnd in range(n_rounds + 1):
            if self.verbose:
                print(f'\n--- Round {rnd}/{n_rounds} | {len(lab_scens)} scen = {len(x_lab)} rows ---')

            # -- Build collocation set (PINN only) ----------------------------
            if self.is_pinn and self.lambda_pde_max > 0.0:
                cur_idx = np.concatenate([s['indices'] for s in lab_scens]).tolist()
                x_lab_orig = x_pool_orig[cur_idx]
                # v3 selector signature: select(model, labeled_x, rng_seed, round_idx)
                try:
                    x_coll = self.coll_sel.select(self.model, x_lab_orig,
                                                   rng_seed=seed + rnd,
                                                   round_idx=rnd).to(dev)
                except TypeError:
                    # v1/v2 selector signature
                    x_coll = self.coll_sel.select(self.model, x_lab_orig,
                                                   rng_seed=seed + rnd).to(dev)
                if x_coll.shape[0] > 0:
                    self.model.update_residual_scales(
                        x_coll, momentum=0.9 if rnd > 0 else 0.0)
                    if self.verbose:
                        rs = self.model.residual_scales.cpu().tolist()
                        print(f'    coll={x_coll.shape[0]}  '
                              f'scales=[{rs[0]:.2e}, {rs[1]:.2e}, '
                              f'{rs[2]:.2e}, {rs[3]:.2e}]')
            else:
                x_coll = torch.empty(0, 102, dtype=torch.float64, device=dev)
            n_coll = len(x_coll)

            # -- Train this round ---------------------------------------------
            h = self._train_block(opt, scheduler, x_lab, y_lab, x_coll,
                                   self.epochs_per_round, global_ep,
                                   x_val=x_val, y_val=y_val)

            dl_h.extend(h['train_data']); pl_h.extend(h['train_pde'])
            tl_h.extend(h['train_total']); lam_h.extend(h['lambda_pde'])
            vl_h.extend(h['val_loss']); vm_h.extend(h['val_mse'])
            vma_h.extend(h['val_mae']); u_h.extend(h['uncertainty'])
            for i in range(4): vmo_h[i].extend(h['val_mse_per_output'][i])
            n_local = len(h['train_data'])
            tsz_h.extend([len(lab_scens)] * n_local)
            csz_h.extend([n_coll] * n_local)
            global_ep += n_local

            # -- Selection ----------------------------------------------------
            if rnd < n_rounds and pool_rem:
                # Try v4 signature (with x_val for EER), fall back to v3, v2, v1
                try:
                    sel = self.data_sel.select(
                        self.model, pool_rem, lab_scens,
                        x_pool_orig=x_pool_orig,
                        x_val=x_val,
                        rng_seed=seed + rnd,
                        round_idx=rnd, max_rounds=n_rounds)
                except TypeError:
                    try:
                        sel = self.data_sel.select(
                            self.model, pool_rem, lab_scens,
                            x_pool_orig=x_pool_orig,
                            rng_seed=seed + rnd,
                            round_idx=rnd, max_rounds=n_rounds)
                    except TypeError:
                        try:
                            sel = self.data_sel.select(self.model, pool_rem, lab_scens,
                                                        x_pool, x_pool_orig=x_pool_orig,
                                                        rng_seed=seed + rnd)
                        except TypeError:
                            sel = self.data_sel.select(self.model, pool_rem, lab_scens,
                                                        x_pool, rng_seed=seed + rnd)
                sel_ids = {id(s) for s in sel}
                pool_rem = [s for s in pool_rem if id(s) not in sel_ids]
                for s in sel:
                    lab_scens.append(s)
                    ni = s['indices'].tolist()
                    x_lab = torch.cat([x_lab, x_pool[ni]], dim=0)
                    y_lab = torch.cat([y_lab, y_pool[ni]], dim=0)
                if self.verbose:
                    print(f'    -> +{len(sel)} scen  (total {len(lab_scens)} = {len(x_lab)} rows)')

        # -- Final polish ------------------------------------------------------
        if self.verbose:
            print(f'\n--- Final polish | {self.n_final_epochs} ep ---')

        if self.is_pinn and self.lambda_pde_max > 0.0:
            cur_idx = np.concatenate([s['indices'] for s in lab_scens]).tolist()
            x_lab_orig = x_pool_orig[cur_idx]
            try:
                x_coll_f = self.coll_sel.select(self.model, x_lab_orig,
                                                rng_seed=seed + 9999,
                                                round_idx=n_rounds).to(dev)
            except TypeError:
                x_coll_f = self.coll_sel.select(self.model, x_lab_orig,
                                                rng_seed=seed + 9999).to(dev)
            if x_coll_f.shape[0] > 0:
                self.model.update_residual_scales(x_coll_f, momentum=0.9)
        else:
            x_coll_f = torch.empty(0, 102, dtype=torch.float64, device=dev)

        hf = self._train_block(opt, scheduler, x_lab, y_lab, x_coll_f,
                                self.n_final_epochs, global_ep,
                                x_val=x_val, y_val=y_val)
        dl_h.extend(hf['train_data']); pl_h.extend(hf['train_pde'])
        tl_h.extend(hf['train_total']); lam_h.extend(hf['lambda_pde'])
        vl_h.extend(hf['val_loss']); vm_h.extend(hf['val_mse'])
        vma_h.extend(hf['val_mae']); u_h.extend(hf['uncertainty'])
        for i in range(4): vmo_h[i].extend(hf['val_mse_per_output'][i])
        n_fep = len(hf['train_data'])
        tsz_h.extend([len(lab_scens)] * n_fep)
        csz_h.extend([len(x_coll_f)] * n_fep)

        wall = time.time() - t0
        fval = vm_h[-1] if vm_h else float('nan')
        if self.verbose:
            print(f'\n[Done v3]  {wall:.1f}s  {len(lab_scens)} scen  '
                  f'{len(x_lab)} rows  val_mse={fval:.4e}')

        return dict(x_labeled=x_lab.cpu(), y_labeled=y_lab.cpu(),
                    data_loss_history=dl_h, pde_loss_history=pl_h,
                    total_loss_history=tl_h, val_loss_history=vl_h,
                    val_mse_history=vm_h, val_mae_history=vma_h,
                    uncertainty_history=u_h, lambda_pde_history=lam_h,
                    val_mse_per_output_history=vmo_h,
                    train_size_history=tsz_h, collocation_size_history=csz_h,
                    wall_time=wall, final_val_mse=fval)
