"""Polynomial regression for BT2024095 (var1: sparse/Lasso, var2: Ridge).
Usage: python train.py --data_dir DATA --out_dir OUT
"""
import argparse, json, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")
ROLL = "BT2024095"

def feats(X, deg): return PolynomialFeatures(deg, include_bias=False).fit_transform(X)

def make(kind, a): return Lasso(alpha=a, max_iter=5000, tol=1e-3) if kind == "lasso" else Ridge(alpha=a)

def cv_score(Z, y, kind, a, kf):
    pr = np.zeros_like(y)
    for tr, te in kf.split(Z):
        sc = StandardScaler().fit(Z[tr])
        pr[te] = make(kind, a).fit(sc.transform(Z[tr]), y[tr]).predict(sc.transform(Z[te]))
    return mean_squared_error(y, pr), r2_score(y, pr)

def search(X, y, kind, degs, alphas):
    kf = KFold(5, shuffle=True, random_state=0); rows = []
    for deg in degs:
        Z = feats(X, deg)
        for a in alphas:
            mse, r2 = cv_score(Z, y, kind, a, kf)
            rows.append(dict(deg=deg, alpha=a, cv_mse=mse, cv_r2=r2))
            print(kind, deg, a, "MSE %.4f R2 %.5f" % (mse, r2), flush=True)
    return pd.DataFrame(rows)

def fit_predict(Xtr, ytr, Xte, kind, deg, a):
    sc = StandardScaler().fit(feats(Xtr, deg)); m = make(kind, a)
    m.fit(sc.transform(feats(Xtr, deg)), ytr)
    return m.predict(sc.transform(feats(Xte, deg))), m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", default="data"); ap.add_argument("--out_dir", default="outputs")
    A = ap.parse_args(); import os; os.makedirs(A.out_dir, exist_ok=True)
    cfg = {1: ("lasso", [3, 4, 5, 6, 7], [0.004, 0.008, 0.012, 0.018]),
           2: ("ridge", list(range(4, 15)), [1e-3, 1e-2, 1e-1, 1, 10])}
    summary = {}
    for v, (kind, degs, alphas) in cfg.items():
        tr = pd.read_csv(f"{A.data_dir}/{ROLL}_train_var{v}.csv"); te = pd.read_csv(f"{A.data_dir}/{ROLL}_test_var{v}.csv")
        X, y = tr.drop(columns="y").values, tr.y.values
        res = search(X, y, kind, degs, alphas); res.to_csv(f"{A.out_dir}/cv_var{v}.csv", index=False)
        b = res.loc[res.cv_mse.idxmin()]; deg, a = int(b.deg), float(b.alpha)
        # independent sanity check: 80/20 holdout with the chosen config
        rng = np.random.RandomState(1); p = rng.permutation(len(y)); k = int(.8 * len(y))
        ph, _ = fit_predict(X[p[:k]], y[p[:k]], X[p[k:]], kind, deg, a)
        pred, m = fit_predict(X, y, te.values, kind, deg, a)
        pd.DataFrame({"y": pred}).to_csv(f"{A.out_dir}/{ROLL}_pred_var{v}.csv", index=False)
        summary[v] = dict(model=kind, degree=deg, alpha=a, cv_mse=float(b.cv_mse), cv_r2=float(b.cv_r2),
                          holdout_mse=float(mean_squared_error(y[p[k:]], ph)), holdout_r2=float(r2_score(y[p[k:]], ph)),
                          n_terms=int(len(m.coef_)), n_nonzero=int((m.coef_ != 0).sum()))
        print(summary[v])
    json.dump(summary, open(f"{A.out_dir}/summary.json", "w"), indent=2)
