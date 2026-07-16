"""
Coin-day realization logit (grouped binomial hazard) — revision analysis, FRL-D-26-03326.

The referee-grade version of the age-bucket result. For every (cohort c, day j) with
STH age in [1,155) and a clean gain/loss label (|ret|>=5%), we form the daily
realization event: among the n_atrisk UTXOs of cohort c alive at the start of day j,
n_realized were spent that day. We model the realization probability with a logit and
ask whether coins IN GAIN are realized faster than coins IN LOSS (the disposition
effect = positive 'gain' coefficient), (i) unconditionally, (ii) with continuous age
control, and (iii) within age buckets. Inference: cohort-clustered robust SE
(the paper's bootstrap unit is the creation-day cohort). Grouped binomial on UTXO
counts is the natural, honest weighting (a real count of realization events).

Reconciliation: the unconditional 'gain' odds ratio tracks the pooled PGR/PLR, because
PGR = sum(realized_gain)/sum(atrisk_gain) is exactly the atrisk-weighted mean
realization rate on gain cohort-days (and PLR the same on loss cohort-days).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, patsy
import statsmodels.api as sm
from reproduce import price_series, CFG, REGIME, FLOOR

DEADBAND = 0.05
BUCKETS = [(1, 7), (7, 30), (30, 90), (90, 155)]
BLAB = {(1,7):"1-7d",(7,30):"7-30d",(30,90):"30-90d",(90,155):"90-155d"}


def build_panel(year, snap_f, new_f, matrix_f, price, min_lag=1):
    """Return a (cohort, day) DataFrame with count + value at-risk / realized volumes."""
    y0, y1 = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
    snap = pd.read_csv(snap_f); snap['create_d'] = pd.to_datetime(snap['create_d'])
    new  = pd.read_csv(new_f);  new['create_d']  = pd.to_datetime(new['create_d'])
    m = pd.read_csv(matrix_f)
    m['create_d'] = pd.to_datetime(m['create_d']); m['spend_d'] = pd.to_datetime(m['spend_d'])
    m = m[m.create_d >= FLOOR]
    m = m[(m.spend_d - m.create_d).dt.days >= min_lag]
    coh = pd.Index(sorted(set(snap.create_d) | set(new.create_d)))
    ci = {c:i for i,c in enumerate(coh)}; C = len(coh)
    pc   = np.array([price[pd.Timestamp(c)] for c in coh])
    cord = np.array([pd.Timestamp(c).toordinal() for c in coh])
    days = pd.date_range(y0, y1, freq='D')
    pdv = np.array([price[x] for x in days]); di = {x:i for i,x in enumerate(days)}; D = len(days)

    def build_amounts(acol, icol, qcol):
        a0 = np.zeros(C); newamt = np.zeros(C); S = np.zeros((C, D))
        for c,v in zip(snap.create_d, snap[acol]): a0[ci[c]] = v
        for c,v in zip(new.create_d, new[icol]):
            if c in ci: newamt[ci[c]] = v
        for cd,sd,v in zip(m.create_d, m.spend_d, m[qcol]):
            if cd in ci and sd in di: S[ci[cd], di[sd]] += v
        return a0, newamt, S
    a0n, newn, Sn = build_amounts('n_alive','n_new','n')
    a0v, newv, Sv = build_amounts('btc_alive','btc_new','btc')

    rows = []
    remn = a0n.copy(); remv = a0v.copy()
    for j in range(D):
        dord = days[j].toordinal()
        addmask = (cord == dord)
        remn = remn + np.where(addmask, newn, 0.0); remv = remv + np.where(addmask, newv, 0.0)
        soldn = Sn[:, j]; soldv = Sv[:, j]
        age = dord - cord
        ret = pdv[j] / pc - 1
        keep = (age >= 1) & (age < 155) & (remn > 0) & (np.abs(ret) >= DEADBAND)
        idx = np.where(keep)[0]
        if len(idx):
            gain = (ret[idx] >= DEADBAND).astype(int)
            rows.append(pd.DataFrame(dict(
                cohort=cord[idx], day=dord, age=age[idx], gain=gain,
                atrisk_n=remn[idx], realized_n=np.clip(soldn[idx], 0, remn[idx]),
                atrisk_v=remv[idx], realized_v=np.clip(soldv[idx], 0, remv[idx]))))
        remn = np.clip(remn - soldn, 0, None); remv = np.clip(remv - soldv, 0, None)
    df = pd.concat(rows, ignore_index=True)
    df['realized_n'] = np.round(df['realized_n']).astype(int)
    df['atrisk_n']   = np.round(df['atrisk_n']).astype(int)
    df = df[df.atrisk_n >= df.realized_n]
    df = df[df.atrisk_n > 0]
    df['bucket'] = pd.cut(df.age, [1,7,30,90,155], right=False,
                          labels=[BLAB[b] for b in BUCKETS])
    df['logage'] = np.log(df.age)
    df['month']  = pd.to_datetime(df.day.map(lambda o: pd.Timestamp.fromordinal(o))).dt.month
    return df


def fit_grouped(df, rhs):
    """Grouped binomial logit; return dict for the 'gain' term: OR, ci_lo, ci_hi, p, n_obs."""
    y = np.column_stack([df.realized_n.values, (df.atrisk_n - df.realized_n).values]).astype(float)
    X = patsy.dmatrix(rhs, df, return_type='dataframe')
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': df.cohort.values})
    if 'gain' not in X.columns:  # gain entered via interaction only; caller handles
        return res, X
    b = res.params['gain']; se = res.bse['gain']
    return dict(OR=np.exp(b), lo=np.exp(b-1.96*se), hi=np.exp(b+1.96*se),
                p=res.pvalues['gain'], n=int(df.cohort.nunique()))


def gain_or_at_meanage(df):
    """Age-adjusted pooled gain OR: gain coef in realize ~ gain + c_logage + gain:c_logage,
    with logage centered so 'gain' is the effect at the sample-mean (log) age."""
    d = df.copy(); d['clogage'] = d.logage - df.logage.mean()
    r = fit_grouped(d, "gain * clogage")
    return r


def main():
    P = price_series()
    print("COIN-DAY REALIZATION LOGIT (grouped binomial on UTXO counts, cohort-clustered SE)\n"
          "Disposition effect = gain odds ratio (OR) > 1: coins in gain realized faster.\n"
          "Reconciliation: unconditional gain OR tracks pooled PGR/PLR.\n")
    summary = []
    for src in ['CMC']:
        for y in CFG:
            df = build_panel(y, *CFG[y], P[src])
            uncond = fit_grouped(df, "gain")
            adj    = gain_or_at_meanage(df)
            print(f"{'='*78}\n{y}  [{REGIME[y]}]   ({src}; obs={len(df):,} cohort-days, "
                  f"cohorts={df.cohort.nunique()})")
            print(f"  (U) UNCONDITIONAL     gain OR = {uncond['OR']:5.2f}  "
                  f"[{uncond['lo']:4.2f}, {uncond['hi']:4.2f}]  p={uncond['p']:.3g}")
            print(f"  (C) AGE-ADJUSTED      gain OR = {adj['OR']:5.2f}  "
                  f"[{adj['lo']:4.2f}, {adj['hi']:4.2f}]  p={adj['p']:.3g}   "
                  f"(logage centered; continuous age control)")
            print("  (B) WITHIN AGE BUCKET  gain OR:")
            brow = {}
            for b in BUCKETS:
                sub = df[df.bucket == BLAB[b]]
                if sub.gain.nunique() < 2 or len(sub) < 20:
                    print(f"        {BLAB[b]:8} (insufficient)"); continue
                r = fit_grouped(sub, "gain")
                star = '*' if r['lo'] > 1 else ('rev' if r['hi'] < 1 else 'ns')
                print(f"        {BLAB[b]:8} OR = {r['OR']:5.2f}  "
                      f"[{r['lo']:4.2f}, {r['hi']:4.2f}]  {star}")
                brow[BLAB[b]] = r
            summary.append((y, uncond, adj, brow))
            # month-FE robustness on the pivotal year
            if y == 2022:
                mfe = fit_grouped(df, "gain + C(month)")
                print(f"  (U+monthFE) gain OR = {mfe['OR']:5.2f} "
                      f"[{mfe['lo']:4.2f}, {mfe['hi']:4.2f}]  p={mfe['p']:.3g}")
    df_out = []
    for y, u, a, brow in summary:
        row = dict(year=y, regime=REGIME[y],
                   OR_uncond=round(u['OR'],3), uncond_lo=round(u['lo'],3), uncond_hi=round(u['hi'],3),
                   OR_ageadj=round(a['OR'],3), ageadj_lo=round(a['lo'],3), ageadj_hi=round(a['hi'],3))
        for bl in [BLAB[b] for b in BUCKETS]:
            if bl in brow:
                row[f'OR_{bl}'] = round(brow[bl]['OR'],3)
                row[f'{bl}_lo'] = round(brow[bl]['lo'],3)
                row[f'{bl}_hi'] = round(brow[bl]['hi'],3)
        df_out.append(row)
    pd.DataFrame(df_out).to_csv('hazard_logit_results.csv', index=False)
    print("\nwrote hazard_logit_results.csv")


if __name__ == '__main__':
    main()
