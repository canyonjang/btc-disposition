"""
Reproduction script for
"Identifying the Disposition Effect on the Bitcoin Blockchain: Real but Regime-Specific".

Running `python reproduce.py` regenerates, from the bundled data in ./data,
with no external services required:
  * Figure 1  - five-year short-term-holder (STH) PGR/PLR, both price series
  * Table S1  - 2022 robustness (8 perturbations)
  * Tables S2/S3 - STH age-threshold sensitivity (CMC / Bitstamp)
  * Table S4  - churn-filter coverage

DATA (in ./data, all derived from the public bigquery-public-data.crypto_bitcoin
dataset via the queries in Supplementary Material Section S4):
  prices_cmc.xlsx, prices_bitstamp.csv          two daily BTC/USD price series
  matrix_<year>.csv.gz    realized (create_d x spend_d) BTC volume; cols: create_d,spend_d,n,btc
  snapshot_<year>.csv.gz  year-start unspent UTXO volume; cols: create_d,n_alive,btc_alive
  newdaily_<year>.csv.gz  daily new creation;            cols: create_d,n_new,btc_new
Creation floor 2018-08-23 (common start of continuous daily price coverage).

Dependencies: pandas, numpy, openpyxl  (see requirements.txt).
"""
import os
import pandas as pd, numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
def d(name): return os.path.join(DATA, name)

# CFG[year] = (snapshot, newdaily, matrix)
CFG = {y: (d(f"snapshot_{y}.csv.gz"), d(f"newdaily_{y}.csv.gz"), d(f"matrix_{y}.csv.gz"))
       for y in (2021, 2022, 2023, 2024, 2025)}
REGIME = {2021:'bull(pre-ETF)',2022:'bear(pre-ETF)',2023:'bull(pre-ETF)',
          2024:'bull(ETF)',2025:'mixed(ETF)'}
FLOOR  = pd.Timestamp('2018-08-23')
CRISIS = [(pd.Timestamp('2022-05-07'), pd.Timestamp('2022-05-20')),   # Terra/LUNA
          (pd.Timestamp('2022-11-06'), pd.Timestamp('2022-11-20'))]   # FTX
rng = np.random.default_rng(7)

def price_series():
    cmc = pd.read_excel(d('prices_cmc.xlsx'))
    cmc['date'] = pd.to_datetime(pd.to_datetime(cmc['timeClose'], unit='ms').dt.date)
    cmc = cmc[['date','priceClose']].rename(columns={'priceClose':'p'}).sort_values('date')
    bs = pd.read_csv(d('prices_bitstamp.csv'), skiprows=1)
    bs['date'] = pd.to_datetime(pd.to_datetime(bs['date']).dt.date)
    bs = bs[['date','close']].rename(columns={'close':'p'}).sort_values('date')
    def fill(df):
        full = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        return df.set_index('date').reindex(full).ffill().bfill()['p']
    return {'CMC': fill(cmc), 'Bitstamp': fill(bs)}

def cohort_arrays(year, snap_f, new_f, matrix_f, price, sth_days=155, deadband=0.05,
                  weight='value', min_lag=1, exclude_crisis=False):
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
    RG=np.zeros(C);PG=np.zeros(C);RL=np.zeros(C);PL=np.zeros(C); rem=a0.copy()
    for j in range(D):
        dord = days[j].toordinal(); rem = rem + np.where(cord == dord, newamt, 0.0)
        sold = S[:, j]; held = np.clip(rem - sold, 0, None); age = dord - cord
        mask = (age >= 0) & (age < sth_days)
        ret = pdv[j] / pc - 1
        g = mask & (ret >= deadband); l = mask & (ret <= -deadband)
        if not excl[j]:
            RG += np.where(g, sold, 0); PG += np.where(g, held, 0)
            RL += np.where(l, sold, 0); PL += np.where(l, held, 0)
        rem = held
    return RG, PG, RL, PL

def ratio_ci(RG, PG, RL, PL, reps=500):
    C = len(RG)
    pt = (RG.sum()/(RG.sum()+PG.sum())) / (RL.sum()/(RL.sum()+PL.sum()))
    rs = []
    for _ in range(reps):
        i = rng.integers(0, C, C); rg,pg,rl,pl = RG[i].sum(),PG[i].sum(),RL[i].sum(),PL[i].sum()
        if rg+pg > 0 and rl+pl > 0 and rl > 0:
            rs.append((rg/(rg+pg)) / (rl/(rl+pl)))
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return pt, lo, hi

def tag(lo, hi):
    return '*' if lo > 1 else ('rev' if hi < 1 else 'ns')

def main():
    P = price_series()
    ov = pd.DataFrame({'cmc': P['CMC'], 'bs': P['Bitstamp']}).dropna()
    print(f"Price overlap {len(ov)} days: corr={ov.cmc.corr(ov.bs):.5f}  "
          f"MAPE={(np.abs(ov.cmc-ov.bs)/ov.cmc).mean()*100:.2f}%\n")

    print("=== Figure 1 - five-year STH PGR/PLR (baseline: |ret|>=5%, no same-day, value-wtd) ===")
    for src in ['CMC','Bitstamp']:
        print(f"  [{src}]")
        for y in CFG:
            pt,lo,hi = ratio_ci(*cohort_arrays(y, *CFG[y], P[src]))
            print(f"    {y} {REGIME[y]:14} {pt:5.2f} [{lo:.2f},{hi:.2f}] {tag(lo,hi)}")

    print("\n=== Table S1 - 2022 robustness (CMC | Bitstamp) ===")
    specs = [("baseline",            dict()),
             ("deadband 10%",        dict(deadband=0.10)),
             ("deadband 20%",        dict(deadband=0.20)),
             ("equal-weighted",      dict(weight='count')),
             ("min holding >=2d",    dict(min_lag=2)),
             ("min holding >=7d",    dict(min_lag=7)),
             ("exclude crisis",      dict(exclude_crisis=True)),
             ("strict 10%+7d+excl",  dict(deadband=0.10, min_lag=7, exclude_crisis=True))]
    for name,kw in specs:
        out = []
        for src in ['CMC','Bitstamp']:
            pt,lo,hi = ratio_ci(*cohort_arrays(2022, *CFG[2022], P[src], **kw))
            out.append(f"{pt:.2f}[{lo:.2f},{hi:.2f}]{tag(lo,hi)}")
        print(f"    {name:20} {out[0]:22} {out[1]}")

    print("\n=== Tables S2/S3 - STH threshold sensitivity ===")
    for src in ['CMC','Bitstamp']:
        print(f"  [{src}]  " + "".join(f"{t:>18}" for t in [90,120,155,180,365]))
        for y in CFG:
            cells = []
            for t in [90,120,155,180,365]:
                pt,lo,hi = ratio_ci(*cohort_arrays(y, *CFG[y], P[src], sth_days=t))
                cells.append(f"{pt:.2f}[{lo:.2f},{hi:.2f}]{tag(lo,hi)}")
            print(f"    {y}  " + " ".join(cells))

    print("\n=== Table S4 - churn-filter coverage (share of STH BTC volume) ===")
    for y in CFG:
        m = pd.read_csv(CFG[y][2])
        m['create_d'] = pd.to_datetime(m['create_d']); m['spend_d'] = pd.to_datetime(m['spend_d'])
        m = m[m.create_d >= FLOOR]
        y0,y1 = pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y}-12-31')
        m = m[(m.spend_d >= y0) & (m.spend_d <= y1)]
        age = (m.spend_d - m.create_d).dt.days
        sth = m[(age >= 0) & (age < 155)]
        same = sth[sth.create_d == sth.spend_d]['btc'].sum() / sth['btc'].sum() * 100
        nsd = sth[sth.create_d != sth.spend_d].copy()
        pc = P['CMC'].reindex(nsd.create_d).values; ps = P['CMC'].reindex(nsd.spend_d).values
        inband = np.abs(ps/pc - 1) < 0.05
        db = nsd['btc'].values[inband].sum() / sth['btc'].sum() * 100
        print(f"    {y}: same-day {same:4.1f}%  deadband {db:4.1f}%")

if __name__ == '__main__':
    main()
