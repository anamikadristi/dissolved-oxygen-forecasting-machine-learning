"""Step 3: train the deep learning model (PyTorch LSTM, CPU) for each site.

One multi-horizon LSTM per site reads the last 14 days of daily variables and
predicts the change in daily minimum DO 1, 2 and 3 days ahead. Inputs are
standardized with training-block statistics; missing values become 0 after
standardization and a missing-DO indicator channel is added. Training uses
the training block only, with early stopping on the validation block. Three
random seeds are trained and their predictions averaged.

This runs as its own script because PyTorch and LightGBM ship separate
OpenMP runtimes that can crash when both are loaded in one macOS process.

Writes outputs/tables/lstm_predictions.csv (validation and test rows).
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from do_config import SITES, HORIZONS, TAB, SEED, SEQ_LEN
from do_common import site_frame, sequences, SEQ_BASE, OPTIONAL

torch.set_num_threads(4)
HIDDEN = 32


class LSTMNet(nn.Module):
    def __init__(self, k, hidden=HIDDEN, out=3):
        super().__init__()
        self.lstm = nn.LSTM(k, hidden, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(self.drop(h[:, -1]))


def train_lstm(Str, Dtr, Mtr, Sva, Dva, Mva, seed, max_epochs=300, patience=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = LSTMNet(Str.shape[2], out=Dtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt, Mt = map(torch.tensor, (Str, Dtr, Mtr))
    Xv, Yv, Mv = map(torch.tensor, (Sva, Dva, Mva))
    best, best_state, wait, ep_best = np.inf, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = (((net(Xt[b]) - Yt[b]) ** 2) * Mt[b]).sum() / Mt[b].sum()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float((((net(Xv) - Yv) ** 2) * Mv).sum() / Mv.sum())
        if vl < best - 1e-5:
            best, wait, ep_best = vl, 0, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, ep_best + 1


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    out, settings = [], []
    for s in SITES:
        t0 = time.time()
        sf = site_frame(s["short"])
        F, Y, X, ok, sp = sf["F"], sf["Y"], sf["X"], sf["ok"], sf["split"]
        seq_vars = [c for c in SEQ_BASE + OPTIONAL if c in X and X[c].notna().mean() > 0.3]
        tr_days = X.index[X.index + pd.Timedelta(days=max(HORIZONS)) < sf["d1"]]
        mu = X.loc[tr_days, seq_vars].mean()
        sd = X.loc[tr_days, seq_vars].std().replace(0, 1)
        rows = {k: F.index[ok & (sp == k)] for k in ["train", "val", "test"]}
        S = {k: sequences(X, seq_vars, v, mu, sd) for k, v in rows.items()}
        D, M, dsd = {}, {}, None
        for k in ["train", "val", "test"]:
            delta = Y.loc[rows[k]].values - F.loc[rows[k], "do_min"].values[:, None]
            if dsd is None:  # scale from the training block only
                dsd = np.nanstd(delta, axis=0)
            M[k] = (~np.isnan(delta)).astype(np.float32)
            D[k] = np.nan_to_num(delta / dsd).astype(np.float32)
        preds, eps = {"val": [], "test": []}, []
        for seed in [SEED, SEED + 1, SEED + 2]:
            net, ep = train_lstm(S["train"], D["train"], M["train"],
                                 S["val"], D["val"], M["val"], seed)
            eps.append(ep)
            for k in preds:
                with torch.no_grad():
                    preds[k].append(net(torch.tensor(S[k])).numpy() * dsd)
        for k in preds:
            arr = np.stack(preds[k])  # seeds x n x horizons
            base = F.loc[rows[k], "do_min"].values
            for j, h in enumerate(HORIZONS):
                df = pd.DataFrame({"site": s["short"], "issue_date": rows[k], "block": k,
                                   "horizon": h,
                                   "lstm": base + arr[:, :, j].mean(0)})
                for i in range(arr.shape[0]):
                    df[f"lstm_seed{i}"] = base + arr[i, :, j]
                out.append(df)
        nparam = sum(p.numel() for p in net.parameters())
        settings.append(dict(site=s["short"], model="lstm", horizon="1-3",
                             setting=f"hidden={HIDDEN}; seq_len={SEQ_LEN}; inputs={len(seq_vars) + 1}; "
                                     f"params={nparam}; best_epochs={eps}; seeds=3"))
        print(f"{s['short']}: n_train={len(rows['train'])} inputs={seq_vars} "
              f"epochs={eps} ({time.time() - t0:.0f}s)")
    res = pd.concat(out)
    num = res.select_dtypes("number").columns.drop("horizon")
    res[num] = res[num].round(4)
    res.to_csv(TAB / "lstm_predictions.csv", index=False)
    pd.DataFrame(settings).to_csv(TAB / "lstm_settings.csv", index=False)


if __name__ == "__main__":
    main()
