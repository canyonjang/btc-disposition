"""
Age-bucket re-estimation of PGR/PLR — revision analysis for FRL-D-26-03326.

Addresses reviewer point (1): age-status confound. The headline STH ratio pools
coins aged 0-155 days. If age and gain/loss status are correlated (they are, under
a trend), the pooled ratio can reflect age composition rather than a within-age
realization asymmetry. We therefore re-estimate PGR/PLR *within* age buckets
[1,7), [7,30), [30,90), [90,155) days, holding age approximately constant.

Also folds in reviewer point (4): for every cell we report the raw value-weighted
PGR and PLR, the count-based gain/loss exposure (N_gain, N_loss) and realized
counts, the number of contributing cohorts C (the real unit of bootstrap
resampling / power), and a TOST equivalence verdict for cells whose CI covers 1.

Reuses price_series() and the accounting logic from reproduce.py verbatim; the
ONLY change to cohort accounting is replacing the single STH mask with per-bucket
masks accumulated in one pass. Supply carryover (rem/held) is unchanged and global.
"""
import os
import numpy as np, pandas as pd
from reproduce import price_series, CFG, REGIME, FLOOR, CRISIS

BUCKETS = [(1, 7), (7, 30), (30, 90), (90, 155)]   # [lo, hi) in days, per reviewer
BLABEL  = {(1,7):"1-7d", (7,30):"7-30d", (30,90):"30-90d", (90,155):"90-155d"}


def cohort_arrays_bucketed(year, snap_f, new_f, matrix_f, price, buckets=BUCKETS,
                           deadband=0.05, weight='value', min_lag=1,
                           exclude_crisis=False):
    """Identical to reproduce.cohort_arrays but returns, per age bucket, the
    length-C arrays (RG, PG, RL, PL). One pass; supply evolution is global."""
    y0, y1 = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31')
    snap = pd.read_csv(snap_f); snap['create_d'] = pd.to_datetime(snap['create_d'])
    new  = pd.read_csv(new_f);  new['create_d']  = pd.to_datetime(new['create_d'])
    m = pd.read_csv(matrix_f)
    m['create_d'] = pd.to_datetime(m['create_d']); m['spend_d'] = pd.to_datetime(m['spend_d'])
    m = m[m.create_d >= FLOOR]
    m = m[(m.spend_d - m.create_d).dt.days >= min_lag]
    qcol = 'btc' if weight == 'value' else 'n'
    acol = 'btc_alive' if weight == 'value' else 'n_alive'
    icol = 'btc_new'   if weight == 'value' else 'n_new'
    coh = pd.Index(sorted(set(snap.create_d) | set(new.create_d)))
    ci = {c:i for i,c in enumerate(coh)}; C = len(coh)
    pc   = np.array([price[pd.Timestamp(c)] for c in coh])
    cord = np.array([pd.Timestamp(c).toordinal() for c in coh])
    a0 = np.zeros(C); newamt = np.zeros(C)
    for c,v in zip(snap.create_d, snap[acol]): a0[ci[c]] = v
    for c,v in zip(new.create_d, new[icol]):
        if c in ci: newamt[ci[c]] = v
    days = pd.date_range(y0, y1, freq='D')
    pdv = np.array([price[x] for x in days]); di = {x:i for i,x in enumerate(days)}; D = len(days)
    S = np.zeros((C, D))
    for cd,sd,v in zip(m.create_d, m.spend_d, m[qcol]):
        if cd in ci and sd in di: S[ci[cd], di[sd]] += v
    excl = np.zeros(D, bool)
    if exclude_crisis:
        for a,b in CRISIS: excl |= (days >= a) & (days <= b)

    acc = {b: [np.zeros(C) for _ in range(4)] for b in buckets}   # b -> [RG,PG,RL,PL]
    rem = a0.copy()
    for j in range(D):
        dord = days[j].toordinal(); rem = rem + np.where(cord == dord, newamt, 0.0)
        sold = S[:, j]; held = np.clip(rem - sold, 0, None); age = dord - cord
        ret = pdv[j] / pc - 1
        gain = ret >= deadband; loss = ret <= -deadband
        if not excl[j]:
            for b in buckets:
                lo, hi = b
                inb = (age >= lo) & (age < hi)
                g = inb & gain; l = inb & loss
                RG, PG, RL, PL = acc[b]
                RG += np.where(g, sold, 0); PG += np.where(g, held, 0)
                RL += np.where(l, sold, 0); PL += np.where(l, held, 0)
        rem = held
    return acc


def ratio_boot(RG, PG, RL, PL, reps=2000, seed=7):
    """Point ratio + bootstrap draws of the ratio (cohort resampling, fixed seed).
    Same seed across buckets within a (year,src) => paired resamples across buckets."""
    C = len(RG)
    def pt_ratio(rg, pg, rl, pl):
        if rg+pg <= 0 or rl+pl <= 0 or rl <= 0: return np.nan
        return (rg/(rg+pg)) / (rl/(rl+pl))
    pt = pt_ratio(RG.sum(), PG.sum(), RL.sum(), PL.sum())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        i = rng.integers(0, C, C)
        r = pt_ratio(RG[i].sum(), PG[i].sum(), RL[i].sum(), PL[i].sum())
        if np.isfinite(r): draws.append(r)
    return pt, np.array(draws)


def tag(lo, hi):
    return '*' if lo > 1 else ('rev' if hi < 1 else 'ns')


def tost(draws, delta):
    """Percentile-bootstrap TOST for H1: ratio in (1/delta, delta).
    Returns (equiv_bool, p_tost, ci90_lo, ci90_hi). Equivalence <=> 90% CI in margin."""
    lo90, hi90 = np.percentile(draws, [5, 95])
    p_upper = np.mean(draws >= delta)      # evidence against ratio>=delta is 1-p_upper
    p_lower = np.mean(draws <= 1.0/delta)
    p_tost = max(p_upper, p_lower)
    equiv = (lo90 > 1.0/delta) and (hi90 < delta)
    return equiv, p_tost, lo90, hi90


def summarize(RG, PG, RL, PL):
    """Value/count summaries for one cell. Inputs are whichever weighting was used."""
    rg, pg, rl, pl = RG.sum(), PG.sum(), RL.sum(), PL.sum()
    pgr = rg/(rg+pg) if rg+pg > 0 else np.nan
    plr = rl/(rl+pl) if rl+pl > 0 else np.nan
    Cnz_gain = int(np.sum((RG+PG) > 0)); Cnz_loss = int(np.sum((RL+PL) > 0))
    return pgr, plr, (rg+pg), (rl+pl), rg, rl, Cnz_gain, Cnz_loss


def main():
    P = price_series()
    print("AGE-BUCKET RE-ESTIMATION of PGR/PLR  (baseline filters: |ret|>=5%, "
          "no same-day, value-weighted; 2000-rep cohort bootstrap, seed=7)\n")
    print("Buckets [lo,hi) days:", ", ".join(BLABEL[b] for b in BUCKETS),
          " | pooled = [1,155)\n")

    for src in ['CMC', 'Bitstamp']:
        print("=" * 92)
        print(f"PRICE SERIES: {src}")
        print("=" * 92)
        for y in CFG:
            accV = cohort_arrays_bucketed(y, *CFG[y], P[src], weight='value')
            accN = cohort_arrays_bucketed(y, *CFG[y], P[src], weight='count')
            # pooled [1,155) = sum of the four buckets (same partition)
            poolV = [sum(accV[b][k] for b in BUCKETS) for k in range(4)]
            poolN = [sum(accN[b][k] for b in BUCKETS) for k in range(4)]

            print(f"\n  {y}  [{REGIME[y]}]")
            print(f"    {'bucket':8} {'ratio':>5} {'95% CI':>13} {'':3} "
                  f"{'PGR':>5} {'PLR':>5} {'Ngain':>10} {'Nloss':>10} {'C_g':>4} {'C_l':>4}")
            rows = [(BLABEL[b], accV[b], accN[b]) for b in BUCKETS] + \
                   [("POOLED", poolV, poolN)]
            for name, cv, cn in rows:
                pt, draws = ratio_boot(*cv)
                lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
                pgr, plr, ng_v, nl_v, rg_v, rl_v, _, _ = summarize(*cv)
                _, _, ng_n, nl_n, rg_n, rl_n, Cg, Cl = summarize(*cn)
                print(f"    {name:8} {pt:5.2f} [{lo:5.2f},{hi:5.2f}] {tag(lo,hi):3} "
                      f"{pgr:5.3f} {plr:5.3f} {ng_n:10.0f} {nl_n:10.0f} {Cg:4d} {Cl:4d}")

    # ---- TOST equivalence panel for null buckets (CMC only, to keep it compact) ----
    print("\n" + "=" * 92)
    print("TOST EQUIVALENCE (CMC): for each bucket, 90% CI and PASS/FAIL vs margins "
          "delta=1.25 and 1.50")
    print("  equivalence 'PASS' <=> 90% CI subset of (1/delta, delta); p = max one-sided boot p")
    print("=" * 92)
    for y in CFG:
        accV = cohort_arrays_bucketed(y, *CFG[y], P['CMC'], weight='value')
        poolV = [sum(accV[b][k] for b in BUCKETS) for k in range(4)]
        print(f"\n  {y} [{REGIME[y]}]")
        for name, cv in [(BLABEL[b], accV[b]) for b in BUCKETS] + [("POOLED", poolV)]:
            pt, draws = ratio_boot(*cv)
            out = []
            for delta in (1.25, 1.50):
                eq, p, lo90, hi90 = tost(draws, delta)
                out.append(f"d={delta:.2f}:{'PASS' if eq else 'FAIL':4}(p={p:.3f})")
            _, _, lo90, hi90 = tost(draws, 1.25)
            print(f"    {name:8} ratio={pt:5.2f}  90%CI[{lo90:5.2f},{hi90:5.2f}]  " + "  ".join(out))


if __name__ == '__main__':
    main()
