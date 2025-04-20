#!/usr/bin/env python3
# HealthCare.py

import os
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ─── 0) Ensure we run from this script’s folder ────────────────────────────
os.chdir(os.path.dirname(__file__))

# ─── 1) Read the IPUMS `.dat` in 5 M‑row chunks ─────────────────────────────
colspecs = [
    (0, 4),    # year
    (54, 56),  # statefip
    (73, 83),  # perwt
    (83, 84),  # hcovany
]
colnames = ["year", "statefip", "perwt", "hcovany"]

print("📥 Reading fixed‑width `.dat` in chunks…")
reader = pd.read_fwf(
    "usa_00001.dat",
    colspecs=colspecs,
    names=colnames,
    engine="python",
    chunksize=5_000_000
)

parts = []
for chunk in reader:
    # scale the person weight
    chunk["perwt"] = chunk["perwt"] / 100
    # flag uninsured (hcovany==1 means NO coverage)
    chunk["uninsured"] = (chunk["hcovany"] == 1).astype(int)
    parts.append(chunk[["statefip", "year", "perwt", "uninsured"]])

df = pd.concat(parts, ignore_index=True)
print(f"✅ Loaded {len(df):,} person‑level rows.")

del parts  # free memory

# ─── 2) Collapse to state–year uninsured rate ───────────────────────────────
print("🔢 Aggregating to state–year uninsured rate…")
panel = (
    df
    .groupby(["statefip", "year"])
    .apply(lambda g: (g["uninsured"] * g["perwt"]).sum() / g["perwt"].sum())
    .reset_index(name="uninsured_rate")
)

del df  # free memory

# ─── 3) Read & prepare expansion_status.csv ─────────────────────────────────
print("📊 Reading expansion_status.csv…")
exp = pd.read_csv(
    "expansion_status.csv",
    dtype={"statefip": str, "year": int}
)
# ensure both sides of merge are same dtype: string of two digits
exp["statefip"] = exp["statefip"].str.zfill(2)
panel["statefip"] = panel["statefip"].astype(str).str.zfill(2)

# ─── 4) Merge panel + expansion info ────────────────────────────────────────
panel = panel.merge(exp, on=["statefip", "year"], how="left")

# ─── 5) Parallel‑trends check ───────────────────────────────────────────────
print("📈 Plotting pre‑expansion trends by expansion status…")
fig, ax = plt.subplots()
(
    panel.query("year < 2014")
         .groupby(["year", "expansion"])["uninsured_rate"]
         .mean()
         .unstack()
         .plot(marker="o", ax=ax)
)
ax.set_ylabel("Avg uninsured rate")
ax.set_title("Pre‑expansion trends by expansion status")
plt.tight_layout()
plt.show()

# ─── 6) Difference‑in‑Differences with robust SEs ──────────────────────────
print("⚖️  Running DiD regression (HC1 SE)…")
did = smf.ols(
    "uninsured_rate ~ expansion * post + C(statefip) + C(year)",
    data=panel
).fit(cov_type="HC1")
print(did.summary())

# ─── 6.1) Extract DiD coefficient, SE, and 95% CI ───────────────────────────
coef = did.params["expansion:post"]
se = did.bse["expansion:post"]
ci_lower, ci_upper = did.conf_int().loc["expansion:post"]
print(
    f"DiD effect = {coef:.4f} "
    f"(95% CI {ci_lower:.4f}, {ci_upper:.4f}), "
    f"p = {did.pvalues['expansion:post']:.3f}"
)

# ─── 7) Save the final panel ────────────────────────────────────────────────
panel.to_csv("state_year_uninsured_panel.csv", index=False)
print("🏁 Done! Panel saved → state_year_uninsured_panel.csv")