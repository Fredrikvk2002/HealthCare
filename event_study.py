#!/usr/bin/env python3
# event_study.py

import os
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ─── 0) Make sure we’re running in this script’s folder ────────────────
os.chdir(os.path.dirname(__file__))

# ─── 1) Load the aggregated state–year panel ────────────────────────────
panel = pd.read_csv("state_year_uninsured_panel.csv", dtype={"statefip": str})

# ─── 2) Define “event time” relative to the 2014 expansion ─────────────
panel["event_time"] = panel["year"] - 2014

# ─── 3) Build lead/lag dummies from –4 to +9 ───────────────────────────
max_lead = 4   # years before
max_lag  = 9   # years after (through 2023)

for k in range(-max_lead, max_lag+1):
    if k < 0:
        panel[f"D_m{abs(k)}"] = (panel["event_time"] == k).astype(int)
    elif k == 0:
        panel["D_0"] = (panel["event_time"] == 0).astype(int)
    else:
        panel[f"D_p{k}"] = (panel["event_time"] == k).astype(int)

# ─── 4) Specify the formula ─────────────────────────────────────────────
leads = [f"D_m{m}" for m in range(max_lead, 0, -1)]
lags  = [f"D_p{p}" for p in range(1, max_lag+1)]
all_dummies = leads + ["D_0"] + lags

formula = (
    "uninsured_rate ~ " +
    " + ".join(all_dummies) +
    " + C(statefip) + C(year)"
)

# ─── 5) Fit the event‐study regression with HC1 SE ──────────────────────
model = smf.ols(formula, data=panel).fit(cov_type="HC1")
print(model.summary())

# ─── 6) Plot the dynamic effects ────────────────────────────────────────
effects = model.params[all_dummies]
cis     = model.conf_int().loc[all_dummies]

x = list(range(-max_lead, max_lag+1))
y = effects.values
yerr_lower = y - cis[0].values
yerr_upper = cis[1].values - y

fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(
    x, y,
    yerr=[yerr_lower, yerr_upper],
    fmt="o", capsize=4
)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xlabel("Years since expansion (2014 = 0)")
ax.set_ylabel("Change in uninsured rate")
ax.set_title("Event‐Study: Dynamic Effects of Medicaid Expansion")
plt.tight_layout()
plt.show()
