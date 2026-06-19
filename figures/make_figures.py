# Generates Figure 1 (forest plot) and Figure 2 (identification ladder).
# Point estimates and CIs are the values reported by reproduce.py.
# Usage: python make_figures.py   (writes fig1_forest.pdf/.png and fig2_ladder.pdf/.png)

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':0.8,
    'xtick.direction':'out','ytick.direction':'out','axes.edgecolor':'#333333',
    'pdf.fonttype':42,'ps.fonttype':42,
})

# canonical numbers (reproduce_results.py / brief sec.2)
data = {
 2021:('Bull (pre-ETF)', (0.89,0.71,1.12), (0.90,0.72,1.13)),
 2022:('Bear (pre-ETF)', (2.62,1.95,3.48), (2.93,2.25,3.89)),
 2023:('Bull (pre-ETF)', (0.84,0.66,1.08), (0.81,0.62,1.05)),
 2024:('Bull (ETF)',     (1.00,0.80,1.27), (1.02,0.81,1.26)),
 2025:('Mixed (ETF)',    (0.82,0.64,1.04), (0.82,0.67,1.03)),
}
years=[2021,2022,2023,2024,2025]

SIG='#b22222'      # firebrick for the one robust effect
NS ='#5a5a5a'      # muted grey for non-significant
fig,ax=plt.subplots(figsize=(6.4,3.7))

ypos={y:len(years)-1-i for i,y in enumerate(years)}   # 2021 top
off=0.16
for y in years:
    yc=ypos[y]; reg,cmc,bsd=data[y]
    sig = cmc[1]>1 or bsd[1]>1
    col = SIG if sig else NS
    for (m,lo,hi),dy,mk,fc in [(cmc,+off,'o',col),(bsd,-off,'s','white')]:
        ax.plot([lo,hi],[yc+dy,yc+dy],'-',color=col,lw=1.3,zorder=2)
        ax.plot([lo,lo],[yc+dy-0.05,yc+dy+0.05],'-',color=col,lw=1.3)
        ax.plot([hi,hi],[yc+dy-0.05,yc+dy+0.05],'-',color=col,lw=1.3)
        ax.plot(m,yc+dy,mk,mfc=(col if mk=='o' else 'white'),mec=col,ms=6.5,mew=1.3,zorder=3)

ax.axvline(1.0,color='#999999',ls='--',lw=1.0,zorder=1)
ax.set_yticks([ypos[y] for y in years])
ax.set_yticklabels([f'{y}\n{data[y][0]}' for y in years],fontsize=9)
ax.set_xlim(0.45,4.05); ax.set_ylim(-0.6,len(years)-0.4)
ax.set_xlabel('STH PGR / PLR  (value-weighted, |ret|\u22655%, 95% bootstrap CI)',fontsize=9.5)
ax.text(1.02,len(years)-0.55,'no effect',fontsize=8,color='#777777',ha='left',va='center',style='italic')
ax.annotate('robust disposition effect\n(only regime above 1.0)',
            xy=(2.93,ypos[2022]-off), xytext=(2.35,ypos[2022]+0.46),
            fontsize=8.5,color=SIG,ha='center',va='bottom',fontweight='bold',
            arrowprops=dict(arrowstyle='-',color=SIG,lw=0.8,shrinkA=0,shrinkB=3))

# legend (marker = price source)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker='o',color='none',mfc=NS,mec=NS,ms=6.5,label='CMC price'),
     Line2D([0],[0],marker='s',color='none',mfc='white',mec=NS,ms=6.5,mew=1.3,label='Bitstamp price')]
ax.legend(handles=leg,loc='lower right',frameon=False,fontsize=8.5,handletextpad=0.4)

ax.spines[['top','right']].set_visible(False)
ax.tick_params(length=3)
plt.tight_layout()
plt.savefig('fig1_forest.pdf',bbox_inches='tight')
plt.savefig('fig1_forest.png',dpi=200,bbox_inches='tight')
print("saved fig1_forest.pdf / .png")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':0.8,
    'axes.edgecolor':'#333333','pdf.fonttype':42,'ps.fonttype':42})
HL='#1f6f6f'; NV='#9a9a9a'
panels=[
 ('2022 · Bear market', 'restricted ratio 2.62 — naive all-age supply understates the real effect', [
    ('All-age supply, naive',         (1.27,1.15,1.41)),
    ('Active supply (STH), naive', (1.41,1.35,1.49)),
    ('Active supply, restricted',  (2.62,1.99,3.50))]),
 ('2024 · ETF bull', 'restricted ratio = null — naive all-age supply fabricates a false reversal', [
    ('All-age supply, naive',         (0.26,0.22,0.29)),
    ('Active supply (STH), naive', (0.73,0.64,0.84)),
    ('Active supply, restricted',  (1.00,0.80,1.29))]),
]
fig,axes=plt.subplots(2,1,figsize=(6.8,4.6),sharex=True,
                      gridspec_kw={'hspace':0.55})
for ax,(title,sub,rungs) in zip(axes,panels):
    for i,(name,(m,lo,hi)) in enumerate(rungs):
        ident=(i==len(rungs)-1); col=HL if ident else NV; y=len(rungs)-1-i
        ax.plot([lo,hi],[y,y],'-',color=col,lw=1.5,zorder=2)
        for x in (lo,hi): ax.plot([x,x],[y-0.14,y+0.14],'-',color=col,lw=1.5)
        ax.plot(m,y,'o',mfc=col,mec=col,ms=7.5 if ident else 6,zorder=3)
    ax.axvline(1.0,color='#bbbbbb',ls='--',lw=1.0,zorder=1)
    ax.set_yticks(range(len(rungs)))
    ax.set_yticklabels([n for n,_ in rungs][::-1],fontsize=8.7)
    for t,(n,_) in zip(ax.get_yticklabels()[::-1],rungs):
        if 'identified' in n: t.set_fontweight('bold'); t.set_color('#111')
        else: t.set_color('#666')
    ax.set_ylim(-0.6,len(rungs)-0.4)
    ax.spines[['top','right']].set_visible(False); ax.tick_params(length=3)
    ax.set_title(title,loc='left',fontsize=10.5,fontweight='bold',pad=12)
    ax.text(0.0,1.01,sub,transform=ax.transAxes,fontsize=8.4,color='#1f6f6f',va='bottom')
axes[0].set_xscale('log'); axes[0].set_xlim(0.18,6.0)
axes[1].set_xticks([0.25,0.5,1,2,4]); axes[1].set_xticklabels(['0.25','0.5','1','2','4'])
axes[1].set_xlabel('PGR / PLR  (log scale; 95% bootstrap CI; CMC price)',fontsize=9.5)
axes[0].text(1.03,len(panels[0][2])-0.55,'no effect',fontsize=7.8,style='italic',color='#999',va='center')
leg=[Line2D([0],[0],marker='o',color='none',mfc=HL,mec=HL,ms=7.5,label='restricted estimator'),
     Line2D([0],[0],marker='o',color='none',mfc=NV,mec=NV,ms=6,label='naive specification')]
axes[1].legend(handles=leg,loc='lower right',frameon=False,fontsize=8.3)
plt.savefig('fig2_ladder.pdf',bbox_inches='tight'); plt.savefig('fig2_ladder.png',dpi=200,bbox_inches='tight')
print('saved')
