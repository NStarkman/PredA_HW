import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

SEED = 0
TARGET = "rd_pct_gdp"
ALPHAS = np.logspace(-3, 4, 40)


PART_B = {
    "Australia": ("AUS", 2),
    "Brazil": ("BRA", 3),
    "Canada": ("CAN", 1),
    "Chile": ("CHL", 2),
    "China": ("CHN", 1),
    "Colombia": ("COL", 3),
    "Estonia": ("EST", 1),
    "European Union": (None, 2),
    "Finland": ("FIN", 1),
    "France": ("FRA", 1),
    "Germany": ("DEU", 1),
    "India": ("IND", 3),
    "Israel": ("ISR", 2),
    "Japan": ("JPN", 2),
    "Kenya": ("KEN", 3),
    "Mexico": ("MEX", 3),
    "Netherlands": ("NLD", 1),
    "Rwanda": ("RWA", 4),
    "Saudi Arabia": ("SAU", 1),
    "Singapore": ("SGP", 2),
    "South Africa": ("ZAF", 3),
    "South Korea": ("KOR", 1),
    "Taiwan": ("TWN", 1),
    "UAE": ("ARE", 1),
    "United Kingdom": ("GBR", 2),
    "United States": ("USA", 2),
    "Vietnam": ("VNM", 3),
}
NAMES = {1: "the Legislators", 2: "the Bougie", 3: "Power in Population", 4: "Rwanda"}

t = pd.read_csv("partC_table.csv")
t = t[t[TARGET].notna()].reset_index(drop=True)

DROP = ["iso3", "country", "wb_region", "income_group", "iso_region", TARGET]
PRED = []
for column in t.columns:
    if column not in DROP and not column.endswith("_year"):
        PRED.append(column)
X, y = t[PRED], t[TARGET]
print(f"{len(t)} countries, {len(PRED)} predictors, target = {TARGET}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"train {len(Xtr)}   held-out test {len(Xte)}\n")

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

PRE = [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]

models = {
    "baseline (predict the average)": Pipeline(
        PRE + [("m", DummyRegressor(strategy="mean"))]),
    "ridge on raw features": Pipeline(
        PRE + [("m", RidgeCV(alphas=ALPHAS))]),
    "random forest on raw features": Pipeline(
        PRE + [("m", RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1))]),
    "ridge on principal components": Pipeline(
        PRE + [("pca", PCA(n_components=0.90, random_state=SEED)),
               ("m", RidgeCV(alphas=ALPHAS))]),
}

#parts 4-5:

results = {}
for name, pipe in models.items():
    folds = cross_validate(pipe, Xtr, ytr, cv=cv, scoring="r2")["test_score"]
    results[name] = folds
    print(f"{name}")
    print("   R2 by fold: " + "  ".join(f"{v:+.3f}" for v in folds))
    print(f"   mean {folds.mean():+.3f}\n")

base = results["baseline (predict the average)"].mean()
print(f"The baseline is R2 = {base:+.3f}. It always predicts the training mean.\n")

for name in list(models)[1:]:
    m = results[name].mean()
    if m > base:
        verdict = "beats"
    else:
        verdict = "does NOT beat"
    print(f"  {name:32s} {m:+.3f}   {verdict} the baseline")

pca_probe = Pipeline(PRE + [("pca", PCA(n_components=0.90, random_state=SEED))]).fit(Xtr)
ncomp = pca_probe.named_steps["pca"].n_components_
print(f"\nItem 6: {ncomp} components carry 90% of the variance in {len(PRED)} predictors.")

print("\nHeld-out test set, scored once")
for name, pipe in models.items():
    pipe.fit(Xtr, ytr)
    print(f"  {name:32s} R2 {r2_score(yte, pipe.predict(Xte)):+.3f}")

print("\nItem 7: what the models use")

ridge = models["ridge on raw features"]
coefs = pd.Series(ridge.named_steps["m"].coef_, index=PRED).sort_values(key=abs, ascending=False)
print("\nridge coefficients (standardised units, largest first):")
for k, v in coefs.items():
    print(f"  {k:24s}{v:+.4f}")

rf = models["random forest on raw features"]
perm = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=SEED, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=PRED).sort_values(ascending=False)
print("\nrandom forest permutation importance (drop in R2 when shuffled):")
for k, v in imp.items():
    print(f"  {k:24s}{v:+.4f}")

top_r = list(coefs.head(5).index)
top_f = list(imp.head(5).index)
shared = []
for column in top_r:
    if column in top_f:
        shared.append(column)
if shared:
    shared_names = ", ".join(shared)
else:
    shared_names = "nothing"
print(f"\n  top five agree on {len(shared)} of 5: {shared_names}")

rows = []
for country, (iso, g) in PART_B.items():
    if iso:
        hit = t[t.iso3 == iso]
    else:
        hit = t.iloc[0:0]
    if len(hit) == 0:
        rows.append({"country": country, "group": g, "in_table": False})
        continue
    i = hit.index[0]
    rows.append({
        "country": country,
        "group": g,
        "in_table": True,
        "actual": hit[TARGET].iloc[0],
        "predicted": ridge.predict(X.loc[[i]])[0],
    })
own = pd.DataFrame(rows)

missing = own[~own.in_table]
print(f"\n{len(missing)} of 27 are not in the table: {', '.join(missing.country)}")

scored = own[own.in_table].copy()
scored["error"] = scored.predicted - scored.actual
scored["abs_error"] = scored.error.abs()

print(f"\n{len(scored)} scored")
print(f"{'country':16s}{'actual':>8s}{'predicted':>11s}{'error':>8s}  group")
for _, r in scored.sort_values("error").iterrows():
    print(f"{r.country:16s}{r.actual:8.2f}{r.predicted:11.2f}{r.error:+8.2f}  {NAMES[r.group]}")

print("\nerror by Part B group:")
print(f"{'group':24s}{'n':>3s}{'mean error':>12s}{'mean |error|':>14s}")
for g, sub in scored.groupby("group"):
    print(f"{NAMES[g]:24s}{len(sub):3d}{sub.error.mean():+12.2f}{sub.abs_error.mean():14.2f}")

r = np.corrcoef(scored.actual, scored.error)[0, 1]
print(f"\ncorrelation: {r:+.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = list(models)
for i, name in enumerate(labels):
    f = results[name]
    ax.scatter([i] * len(f), f, s=42, zorder=3, color="0.35")
    ax.plot([i - .22, i + .22], [f.mean()] * 2, lw=2.4, color="firebrick", zorder=4)
ax.axhline(0, ls="--", lw=1, c="0.6")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels([l.replace(" on ", "\non ").replace(" (", "\n(") for l in labels],
                   fontsize=8.5)
ax.set_ylabel("cross-validated $R^2$")
ax.set_title("Figure C1. Individual fold scores, five-fold CV on the training set.\n"
             "Red bar is the mean.", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig("figC1_folds.png", dpi=200)
print("\nsaved figC1_folds.png")

scored.to_csv("partC_own27_errors.csv", index=False)