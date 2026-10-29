import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SEED = 0
RESTARTS = 50
TARGET = "rd_pct_gdp"

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

CLUSTER_FEATS = []
for column in PRED:
    if not column.startswith("region_"):
        CLUSTER_FEATS.append(column)
print(f"{len(PRED)} Part C features, {len(CLUSTER_FEATS)} used for clustering "
      f"({len(PRED)-len(CLUSTER_FEATS)} region indicators held out)")

prep = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
Z = prep.fit_transform(t[CLUSTER_FEATS])
print(f"clustering {Z.shape[0]} countries on {Z.shape[1]} features\n")

print("Item 2: choosing k by silhouette")
ks = list(range(2, 11))
sil = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=RESTARTS, random_state=SEED).fit(Z)
    sil.append(silhouette_score(Z, km.labels_))
print(f"{'k':>3s}{'silhouette':>13s}")
for k, s in zip(ks, sil):
    print(f"{k:3d}{s:13.3f}")

K = ks[int(np.argmax(sil))]
runner_up = ks[int(np.argsort(sil)[-2])]
print(f"\nsilhouette is highest at k = {K} ({max(sil):.3f}); next best is k = {runner_up}.")

km = KMeans(n_clusters=K, n_init=RESTARTS, random_state=SEED).fit(Z)
t["cluster"] = km.labels_ + 1

print("\nItem 3: every feature's average against the overall average")
prof = t.groupby("cluster")[CLUSTER_FEATS].mean()
prof.loc["ALL"] = t[CLUSTER_FEATS].mean()
pd.set_option("display.width", 250)
print(prof.round(2).to_string())

print(f"\n{'cluster':>8s}{'n':>5s}  {'R&D % GDP':>10s}   example members")
for c in sorted(t.cluster.unique()):
    sub = t[t.cluster == c]
    ex = ", ".join(sub.nlargest(4, "energy_use_pc_kwh").country)
    print(f"{c:8d}{len(sub):5d}  {sub[TARGET].mean():10.2f}   {ex}")



def locate(labels):
    out = []
    for country, (iso, g) in PART_B.items():
        hit = t[t.iso3 == iso] if iso else t.iloc[0:0]
        if len(hit):
            out.append({"country": country,
                        "partB": NAMES[g],
                        "partD": int(labels[hit.index[0]])})
    return pd.DataFrame(out)


found = locate(t.cluster.values)
absent = [c for c in PART_B if c not in set(found.country)]
print(f"{len(found)} of 27 are in the table ({', '.join(absent)} are not).\n")
print(pd.crosstab(found.partB, found.partD,
                  rownames=["Part B group"], colnames=["Part D cluster"]).to_string())

share = found.partD.value_counts(normalize=True)
big = share.index[0]
bigshare = share.iloc[0]
world = (t.cluster == big).mean()
print(f"\nThe largest single landing place is cluster {big}: "
      f"{int(bigshare*len(found))} of {len(found)} of mine ({bigshare:.0%}), "
      f"against {world:.0%} of the world.")
print(f"My 27 occupy {found.partD.nunique()} of the {K} clusters.")

km2 = KMeans(n_clusters=runner_up, n_init=RESTARTS, random_state=SEED).fit(Z)
check = locate(km2.labels_ + 1)
print(f"\nSame question at the runner-up k = {runner_up}:")
print(pd.crosstab(check.partB, check.partD,
                  rownames=["Part B group"], colnames=[f"cluster (k={runner_up})"]).to_string())

zp = pd.DataFrame(Z, columns=CLUSTER_FEATS)
zp["cluster"] = km2.labels_ + 1
print(f"\nk={runner_up} clusters, standard deviations from the overall average:")
print(zp.groupby("cluster").mean().round(2).to_string())
for c in sorted(set(km2.labels_ + 1)):
    members = t.country[(km2.labels_ + 1) == c]
    mine = ", ".join(check.country[check.partD == c])
    print(f"\ncluster {c} (n={len(members)}): {', '.join(members)}")
    print(f"  mine: {mine if mine else 'none'}")

fig, ax = plt.subplots(figsize=(10, 4.6))
cmap = plt.get_cmap("tab10")
for c in sorted(t.cluster.unique()):
    sub = t[t.cluster == c]
    ax.scatter(sub.energy_use_pc_kwh, sub[TARGET], s=26, alpha=.55,
               color=cmap(c - 1), label=f"cluster {c} (n={len(sub)})")
own = t[t.iso3.isin([iso for iso, _ in PART_B.values() if iso])]
ax.scatter(own.energy_use_pc_kwh, own[TARGET], s=96, facecolors="none",
           edgecolors="k", lw=1.3, label="my 27", zorder=5)
ax.set_xlabel("Energy use per person (kWh)")
ax.set_ylabel("R&D spending, % of GDP")
ax.set_title(f"Figure D1. {len(t)} countries in {K} k-means clusters. "
             f"Circled points are the class's 27.", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("figD1_world.png", dpi=200)

t[["iso3", "country", "cluster", TARGET]].to_csv("partD_clusters.csv", index=False)
print("\nsaved figD1_world.png, partD_clusters.csv")