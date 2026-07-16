import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from reproduce import price_series, CFG
from age_buckets import cohort_arrays_bucketed
from age_std_v2 import estimators, sums, FINE
P = price_series(); rows=[]
print("count-weighted (CMC) standardization robustness, 800 reps")
for y in CFG:
    acc = cohort_arrays_bucketed(y, *CFG[y], P['CMC'], buckets=FINE, weight='count')
    C = len(next(iter(acc.values()))[0]); pt = estimators(sums(acc))
    rng = np.random.default_rng(7); d=[]
    for _ in range(800):
        r = estimators(sums(acc, rng.integers(0,C,C)))
        if np.isfinite(r['DR_G']): d.append(r['DR_G'])
    lo,hi = np.percentile(d,[2.5,97.5])
    share = 100*pt['comp']/pt['diff'] if abs(pt['diff'])>2e-5 else float('nan')
    print(f"  {y}: pooled={pt['pooled']:.2f}  DR*_G={pt['DR_G']:.2f} [{lo:.2f},{hi:.2f}]  comp%={share:.0f}" if np.isfinite(share) else f"  {y}: pooled={pt['pooled']:.2f}  DR*_G={pt['DR_G']:.2f} [{lo:.2f},{hi:.2f}]  (diff~0)")
    rows.append(dict(year=y, weight='count', pooled=round(pt['pooled'],3), DR_G=round(pt['DR_G'],3),
                     DR_G_lo=round(lo,3), DR_G_hi=round(hi,3)))
pd.DataFrame(rows).to_csv('age_std_count.csv', index=False); print("wrote age_std_count.csv")
