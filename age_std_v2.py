"""
Age-standardized disposition ratio v2 — overlap-aware.
PRIMARY: DR*_G = PGR / PLR*(G), PLR*(G) = sum_a w^G_a h^L_a  (indirect standardization onto
the GAIN age distribution: never extrapolates h^G outside its own support; h^L is well
estimated at all ages in every year). SECONDARY: DR*_L (loss mix; unstable where gain
exposure is thin) and trimmed-total (bins kept iff each side's exposure share >= 0.5%).
Kitagawa decomposition unchanged. One cohort-array build per year; all estimators share
the same 2000 bootstrap replicates (seed 7).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from reproduce import price_series, CFG
from age_buckets import cohort_arrays_bucketed

FINE = [(a, min(a+7,155)) for a in range(1,155,7)]
SEED, REPS, TRIM = 7, 2000, 0.005

def sums(acc, idx=None):
    out = np.zeros((len(acc),4))
    for bi,b in enumerate(acc):
        for k in range(4):
            v = acc[b][k]; out[bi,k] = v.sum() if idx is None else v[idx].sum()
    return out

def estimators(S):
    RG,PG,RL,PL = S[:,0],S[:,1],S[:,2],S[:,3]
    EG,EL = RG+PG, RL+PL; NG,NL = EG.sum(), EL.sum()
    PGR = RG.sum()/NG; PLR = RL.sum()/NL; pooled = PGR/PLR if PLR>0 else np.nan
    hG = np.divide(RG, EG, out=np.full_like(RG,np.nan), where=EG>0)
    hL = np.divide(RL, EL, out=np.full_like(RL,np.nan), where=EL>0)
    wG, wL = EG/NG, EL/NL
    # (1) gain-mix standardization: PLR reweighted to gain age distribution
    m = EL>0
    PLR_G = np.nansum(np.where(m,wG,0)*hL)/np.where(wG[m].sum()>0,wG[m].sum(),np.nan)
    DR_G = PGR/PLR_G if PLR_G>0 else np.nan
    # (2) loss-mix: PGR reweighted to loss age distribution
    m2 = EG>0
    PGR_L = np.nansum(np.where(m2,wL,0)*hG)/np.where(wL[m2].sum()>0,wL[m2].sum(),np.nan)
    DR_L = PGR_L/PLR if PLR>0 else np.nan
    # (3) trimmed total-exposure weights (overlap-restricted)
    keep = (wG>=TRIM)&(wL>=TRIM)
    if keep.sum()>0:
        q = (EG+EL)*keep; q = q/q.sum()
        DR_T = np.nansum(q*hG)/np.nansum(q*hL)
        suppG, suppL = wG[keep].sum(), wL[keep].sum()
    else:
        DR_T, suppG, suppL = np.nan, 0.0, 0.0
    # Kitagawa (full support, exact)
    hG0 = np.where(EG>0,hG,0.0); hL0 = np.where(EL>0,hL,0.0)
    within = np.sum(0.5*(wG+wL)*(hG0-hL0)); comp = np.sum(0.5*(hG0+hL0)*(wG-wL))
    return dict(pooled=pooled, DR_G=DR_G, DR_L=DR_L, DR_T=DR_T,
                suppG=suppG, suppL=suppL, diff=PGR-PLR, within=within, comp=comp)

def main():
    P = price_series()
    rows=[]
    print("v2: DR*_G = gain-mix standardized (PRIMARY) | DR*_L = loss-mix | DR*_T = trimmed-total")
    print(f"{'yr':4} {'pooled':>7} {'DR*_G':>6} {'[95% CI]':>14} {'DR*_L':>6} {'DR*_T':>6} "
          f"{'suppG/L%':>9} {'diff/10k':>9} {'within':>7} {'comp':>6} {'comp share':>10}")
    for y in CFG:
        acc = cohort_arrays_bucketed(y, *CFG[y], P['CMC'], buckets=FINE, weight='value')
        C = len(next(iter(acc.values()))[0])
        pt = estimators(sums(acc))
        rng = np.random.default_rng(SEED)
        dG,dL,dT,dP = [],[],[],[]
        for _ in range(REPS):
            r = estimators(sums(acc, rng.integers(0,C,C)))
            for lst,k in ((dG,'DR_G'),(dL,'DR_L'),(dT,'DR_T'),(dP,'pooled')):
                if np.isfinite(r[k]): lst.append(r[k])
        ciG = np.percentile(dG,[2.5,97.5]); ciL = np.percentile(dL,[2.5,97.5])
        ciT = np.percentile(dT,[2.5,97.5]); ciP = np.percentile(dP,[2.5,97.5])
        share = 100*pt['comp']/pt['diff'] if abs(pt['diff'])>2e-5 else np.nan
        sh = f"{share:9.0f}%" if np.isfinite(share) else "   (diff~0)"
        print(f"{y:4} {pt['pooled']:7.2f} {pt['DR_G']:6.2f} [{ciG[0]:5.2f},{ciG[1]:5.2f}] "
              f"{pt['DR_L']:6.2f} {pt['DR_T']:6.2f} {100*pt['suppG']:4.0f}/{100*pt['suppL']:3.0f} "
              f"{1e4*pt['diff']:9.2f} {1e4*pt['within']:7.2f} {1e4*pt['comp']:6.2f} {sh}")
        rows.append(dict(year=y, pooled=round(pt['pooled'],3),
            pooled_lo=round(ciP[0],3), pooled_hi=round(ciP[1],3),
            DR_G=round(pt['DR_G'],3), DR_G_lo=round(ciG[0],3), DR_G_hi=round(ciG[1],3),
            DR_L=round(pt['DR_L'],3), DR_L_lo=round(ciL[0],3), DR_L_hi=round(ciL[1],3),
            DR_T=round(pt['DR_T'],3), DR_T_lo=round(ciT[0],3), DR_T_hi=round(ciT[1],3),
            suppG_pct=round(100*pt['suppG'],1), suppL_pct=round(100*pt['suppL'],1),
            diff_per10k=round(1e4*pt['diff'],2), within_per10k=round(1e4*pt['within'],2),
            comp_per10k=round(1e4*pt['comp'],2),
            comp_share_pct=round(share,1) if np.isfinite(share) else np.nan))
    pd.DataFrame(rows).to_csv('age_std_results.csv', index=False)
    print("\nwrote age_std_results.csv (v2)")


if __name__ == '__main__':
    main()
