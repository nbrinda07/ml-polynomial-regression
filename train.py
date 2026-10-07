"""Polynomial regression for BT2024095.
For each problem, 5-fold CV searches over method (OLS / Ridge / Lasso), polynomial degree and
penalty strength alpha. The configuration with the lowest CV MSE is selected automatically,
refit on all training rows, and used to predict the test set.
Usage: python train.py --data_dir data --out_dir outputs
"""
import argparse, json, os, warnings, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")
ROLL = "BT2024095"

# search grids (degrees and alphas per method)
GRID = {1: dict(degs=range(1, 8),  ridge=[1e-2, 1e-1, 1, 10, 100], lasso=[0.004, 0.008, 0.012, 0.018]),
        2: dict(degs=range(2, 15), ridge=[1e-2, 1e-1, 1, 10],      lasso=[3e-4, 1e-3, 3e-3, 1e-2])}

def feats(X, deg): return PolynomialFeatures(deg, include_bias=False).fit_transform(X)

def make(kind, a):
    if kind == "ols": return LinearRegression()
    if kind == "ridge": return Ridge(alpha=a)
    return Lasso(alpha=a, max_iter=5000, tol=1e-3)

def oof_pred(Z, y, kind, a, kf):
    pr = np.zeros_like(y)
    for tr, te in kf.split(Z):
        sc = StandardScaler().fit(Z[tr])
        pr[te] = make(kind, a).fit(sc.transform(Z[tr]), y[tr]).predict(sc.transform(Z[te]))
    return pr

def search(X, y, g):
    kf = KFold(5, shuffle=True, random_state=0); rows = []
    for deg in g["degs"]:
        Z = feats(X, deg)
        for kind, a in [("ols", 0.0)] + [("ridge", a) for a in g["ridge"]] + [("lasso", a) for a in g["lasso"]]:
            p = oof_pred(Z, y, kind, a, kf)
            rows.append(dict(model=kind, deg=deg, alpha=a, n_terms=Z.shape[1],
                             cv_mse=mean_squared_error(y, p), cv_r2=r2_score(y, p)))
            print(kind, deg, a, "MSE %.4f" % rows[-1]["cv_mse"], flush=True)
    return pd.DataFrame(rows)

def fit_predict(Xtr, ytr, Xte, kind, deg, a):
    sc = StandardScaler().fit(feats(Xtr, deg)); m = make(kind, a)
    m.fit(sc.transform(feats(Xtr, deg)), ytr)
    return m.predict(sc.transform(feats(Xte, deg))), m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", default="data"); ap.add_argument("--out_dir", default="outputs")
    A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True); summary = {}
    for v, g in GRID.items():
        tr = pd.read_csv(f"{A.data_dir}/{ROLL}_train_var{v}.csv"); te = pd.read_csv(f"{A.data_dir}/{ROLL}_test_var{v}.csv")
        X, y = tr.drop(columns="y").values, tr.y.values
        res = search(X, y, g); res.to_csv(f"{A.out_dir}/cv_var{v}.csv", index=False)
        b = res.loc[res.cv_mse.idxmin()]; kind, deg, a = b.model, int(b.deg), float(b.alpha)
        # sanity check: 80/20 holdout with the selected configuration
        p = np.random.RandomState(1).permutation(len(y)); k = int(.8 * len(y))
        ph, _ = fit_predict(X[p[:k]], y[p[:k]], X[p[k:]], kind, deg, a)
        pred, m = fit_predict(X, y, te.values, kind, deg, a)
        pd.DataFrame({"y": pred}).to_csv(f"{A.out_dir}/{ROLL}_pred_var{v}.csv", index=False)
        pd.DataFrame({"y_oof": oof_pred(feats(X, deg), y, kind, a, KFold(5, shuffle=True, random_state=0))}).to_csv(f"{A.out_dir}/oof_var{v}.csv", index=False)
        summary[v] = dict(model=kind, degree=deg, alpha=a, cv_mse=float(b.cv_mse), cv_r2=float(b.cv_r2),
                          holdout_mse=float(mean_squared_error(y[p[k:]], ph)), holdout_r2=float(r2_score(y[p[k:]], ph)),
                          n_terms=int(len(m.coef_)), n_nonzero=int((m.coef_ != 0).sum()))
        print("SELECTED", v, summary[v], flush=True)
    json.dump(summary, open(f"{A.out_dir}/summary.json", "w"), indent=2)
