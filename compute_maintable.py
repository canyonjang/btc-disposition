import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from reproduce import price_series, CFG, REGIME
from age_buckets import cohort_arrays_bucketed, ratio_boot, BUCKETS

P = price_series()
print("MAIN RESULTS TABLE (CMC, value-weighted, pooled STH [1,155)); PGR/PLR are daily realization rates")
print(f"{'Year':5}{'PGR':>9}{'PLR':>9}{'PGR-PLR':>10}{'Ratio':>7}{'95% CI':>15}{'Ngain(M)':>11}{'Nloss(M)':>11}{'cohorts':>8}")
rows=[]
for y in CFG:
    accV = cohort_arrays_bucketed(y, *CFG[y], P['CMC'], weight='value')
    accN = cohort_arrays_bucketed(y, *CFG[y], P['CMC'], weight='count')
    RG=sum(accV[b][0].sum() for b in BUCKETS); PG=sum(accV[b][1].sum() for b in BUCKETS)
    RL=sum(accV[b][2].sum() for b in BUCKETS); PL=sum(accV[b][3].sum() for b in BUCKETS)
    RGn=sum(accN[b][0].sum() for b in BUCKETS); PGn=sum(accN[b][1].sum() for b in BUCKETS)
    RLn=sum(accN[b][2].sum() for b in BUCKETS); PLn=sum(accN[b][3].sum() for b in BUCKETS)
    pgr=RG/(RG+PG); plr=RL/(RL+PL)
    poolV=[sum(accV[b][k] for b in BUCKETS) for k in range(4)]
    pt,draws=ratio_boot(*poolV); lo,hi=np.percentile(draws,[2.5,97.5])
    Ng=(RGn+PGn)/1e6; Nl=(RLn+PLn)/1e6
    C=int(np.sum((poolV[0]+poolV[1])>0))
    print(f"{y:<5}{pgr:9.4f}{plr:9.4f}{(pgr-plr):10.4f}{pt:7.2f}  [{lo:4.2f},{hi:4.2f}]  {Ng:10.1f}{Nl:11.1f}{C:8d}")
    rows.append(dict(year=y,PGR=round(pgr,4),PLR=round(plr,4),diff=round(pgr-plr,4),ratio=round(pt,2),
                     lo=round(lo,2),hi=round(hi,2),Ngain_M=round(Ng,1),Nloss_M=round(Nl,1),cohorts=C))
pd.DataFrame(rows).to_csv('main_results_table.csv',index=False)
print("\nwrote main_results_table.csv")
