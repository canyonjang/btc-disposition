"""Monthly standardized series (all years) + calendar-time block bootstrap (2022)
+ min-holding sensitivity (2022). Shares one day-level accumulation pass per year."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from reproduce import price_series, CFG, FLOOR
from age_buckets import cohort_arrays_bucketed
from age_std_v2 import estimators, sums, FINE

def day_arrays(year, snap_f, new_f, matrix_f, price, deadband=0.05, min_lag=1):
    """One pass; returns days index and dayacc[D, nb, 4] value-weighted sums."""
    y0,y1 = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
    snap = pd.read_csv(snap_f); snap['create_d']=pd.to_datetime(snap['create_d'])
    new  = pd.read_csv(new_f);  new['create_d']=pd.to_datetime(new['create_d'])
    m = pd.read_csv(matrix_f)
    m['create_d']=pd.to_datetime(m['create_d']); m['spend_d']=pd.to_datetime(m['spend_d'])
    m = m[m.create_d>=FLOOR]; m = m[(m.spend_d-m.create_d).dt.days>=min_lag]
    coh = pd.Index(sorted(set(snap.create_d)|set(new.create_d)))
    ci = {c:i for i,c in enumerate(coh)}; C=len(coh)
    pc = np.array([price[pd.Timestamp(c)] for c in coh])
    cord = np.array([pd.Timestamp(c).toordinal() for c in coh])
    a0=np.zeros(C); newamt=np.zeros(C)
    for c,v in zip(snap.create_d, snap['btc_alive']): a0[ci[c]]=v
    for c,v in zip(new.create_d, new['btc_new']):
        if c in ci: newamt[ci[c]]=v
    days = pd.date_range(y0,y1,freq='D'); D=len(days)
    pdv = np.array([price[x] for x in days]); di={x:i for i,x in enumerate(days)}
    S = np.zeros((C,D))
    for cd,sd,v in zip(m.create_d,m.spend_d,m['btc']):
        if cd in ci and sd in di: S[ci[cd],di[sd]]+=v
    nb=len(FINE); dayacc=np.zeros((D,nb,4))
    rem=a0.copy()
    for j in range(D):
        dord=days[j].toordinal(); rem=rem+np.where(cord==dord,newamt,0.0)
        sold=S[:,j]; held=np.clip(rem-sold,0,None); age=dord-cord
        ret=pdv[j]/pc-1; gain=ret>=deadband; loss=ret<=-deadband
        for bi,(lo,hi) in enumerate(FINE):
            inb=(age>=lo)&(age<hi)
            g=inb&gain; l=inb&loss
            dayacc[j,bi,0]=sold[g].sum(); dayacc[j,bi,1]=held[g].sum()
            dayacc[j,bi,2]=sold[l].sum(); dayacc[j,bi,3]=held[l].sum()
        rem=held
    return days, dayacc

P = price_series()

# ---------- 1) Monthly pooled vs standardized series ----------
rows=[]; dayacc2022=None; days2022=None
for y in CFG:
    days, dacc = day_arrays(y, *CFG[y], P['CMC'])
    if y==2022: dayacc2022, days2022 = dacc, days
    mo = pd.Series(days.month, index=range(len(days)))
    for mth in range(1,13):
        idx = mo[mo==mth].index.values
        Smo = dacc[idx].sum(axis=0)          # (nb,4)
        e = estimators(Smo)
        rows.append(dict(year=y, month=mth, pooled=e['pooled'], DR_G=e['DR_G'],
                         diff_per10k=1e4*e['diff'], comp_per10k=1e4*e['comp']))
mdf = pd.DataFrame(rows); mdf.to_csv('monthly_std_series.csv', index=False)
print("monthly series: wrote monthly_std_series.csv,", len(mdf), "rows")
print("  2022 monthly pooled range:", round(mdf[mdf.year==2022].pooled.min(),2), "-",
      round(mdf[mdf.year==2022].pooled.max(),2),
      "| DR*_G range:", round(mdf[mdf.year==2022].DR_G.min(),2), "-",
      round(mdf[mdf.year==2022].DR_G.max(),2))

# figure
t = np.arange(len(mdf)); lab=[f"{r.year}-{r.month:02d}" for r in mdf.itertuples()]
fig,ax=plt.subplots(figsize=(11,3.8))
ax.plot(t, mdf.pooled, color='#c0392b', lw=1.8, label='Pooled PGR/PLR (monthly)')
ax.plot(t, mdf.DR_G,  color='#2c3e50', lw=1.8, label='Age-standardized DR*_G (monthly)')
ax.axhline(1, color='#7f8c8d', ls='--', lw=1)
for yy in [2022]:
    i0=mdf.index[(mdf.year==yy)&(mdf.month==1)][0]; i1=mdf.index[(mdf.year==yy)&(mdf.month==12)][0]
    ax.axvspan(i0,i1,color='#f39c12',alpha=0.12)
ax.set_yscale('log'); ax.set_yticks([0.5,1,2,4]); ax.set_yticklabels(['0.5','1','2','4'])
ax.set_xticks(t[::6]); ax.set_xticklabels([lab[i] for i in t[::6]], rotation=45, fontsize=8)
ax.legend(frameon=False, fontsize=9); ax.grid(axis='y', alpha=0.25)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.set_title("Monthly pooled vs age-standardized disposition ratio (CMC, value-weighted; shaded = 2022)", fontsize=10)
plt.tight_layout(); plt.savefig('fig_monthly_std.png', dpi=150); plt.savefig('fig_monthly_std.pdf')
print("  wrote fig_monthly_std.png/.pdf")

# ---------- 2) Calendar-time block bootstrap, 2022 ----------
D = dayacc2022.shape[0]; blocks=[np.arange(i,min(i+7,D)) for i in range(0,D,7)]
rng=np.random.default_rng(7); bp,bg=[],[]
for _ in range(2000):
    pick=rng.integers(0,len(blocks),len(blocks))
    idx=np.concatenate([blocks[k] for k in pick])
    e=estimators(dayacc2022[idx].sum(axis=0))
    if np.isfinite(e['pooled']): bp.append(e['pooled'])
    if np.isfinite(e['DR_G']): bg.append(e['DR_G'])
print(f"\nblock bootstrap 2022 (weekly blocks, 2000 reps):")
print(f"  pooled 2.62  cohort CI [1.99,3.49]  vs  block CI [{np.percentile(bp,2.5):.2f},{np.percentile(bp,97.5):.2f}]")
print(f"  DR*_G  0.93  cohort CI [0.72,1.19]  vs  block CI [{np.percentile(bg,2.5):.2f},{np.percentile(bg,97.5):.2f}]")

# ---------- 3) Min-holding sensitivity, 2022 ----------
print("\nmin-holding sensitivity 2022 (CMC, value): pooled | DR*_G | comp share")
for lag in (2,7):
    acc = cohort_arrays_bucketed(2022, *CFG[2022], P['CMC'], buckets=FINE, weight='value', min_lag=lag)
    e = estimators(sums(acc))
    share = 100*e['comp']/e['diff'] if abs(e['diff'])>2e-5 else np.nan
    print(f"  lag>={lag}d: pooled={e['pooled']:.2f}  DR*_G={e['DR_G']:.2f}  comp={share:.0f}%")
