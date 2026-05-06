"""
CIVE 580 — Community Seismic Resilience Dashboard
Berkeley (High-Code) vs Coalinga (Low-Code)
Run with: streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Seismic Resilience Dashboard", page_icon="\U0001f3d9\ufe0f",
                   layout="wide", initial_sidebar_state="expanded")

NAVY="#0D2137"; BLUE="#065A82"; TEAL="#1C7293"; MINT="#02C39A"
BK_COL="#1C7293"; CL_COL="#C62828"; WHITE="#FFFFFF"; LGRAY="#EEF4F8"
DS_COLORS=["#4CAF50","#8BC34A","#FFC107","#FF5722","#B71C1C"]
CAT_COLOR={"Residential":"#1565C0","Hospital":"#D32F2F","Fire Stn":"#E64A19",
           "Police":"#6A1B9A","Electric":"#F57F17","Water":"#00838F",
           "Bridge":"#2E7D32","School":"#558B2F","Gas Stn":"#795548"}
LOSS_RATIOS={"wood":[0.02,0.10,0.40,0.75],"conc":[0.015,0.08,0.35,0.70],
             "steel":[0.015,0.08,0.35,0.70],"mason":[0.015,0.08,0.40,0.80],
             "mh":[0.030,0.15,0.65,0.95],"essfac":[0.015,0.08,0.35,0.70],
             "infra":[0.010,0.05,0.25,0.60]}

st.markdown(f"""<style>
header[data-testid="stHeader"]{{visibility:hidden;height:0;min-height:0}}
div[data-testid="stToolbar"]{{visibility:hidden}}
#MainMenu{{visibility:hidden}}
div[data-testid="stMainBlockContainer"]{{padding-top:1rem}}
.block-container{{padding-top:1rem}}
.stApp{{background-color:{LGRAY}}}
.metric-card{{background:{NAVY};border-radius:10px;padding:14px 16px;margin-bottom:8px;border-left:4px solid {TEAL}}}
.section-hdr{{background:{NAVY};color:{WHITE};font-size:14px;font-weight:700;padding:6px 14px;border-radius:6px;margin:12px 0 6px;letter-spacing:.5px}}
.city-hdr-bk{{background:{BK_COL};color:{WHITE};text-align:center;font-size:15px;font-weight:800;padding:6px;border-radius:6px;margin-bottom:8px}}
.city-hdr-cl{{background:{CL_COL};color:{WHITE};text-align:center;font-size:15px;font-weight:800;padding:6px;border-radius:6px;margin-bottom:8px}}
.callout-banner{{background:{NAVY};color:{WHITE};text-align:center;font-size:13px;font-weight:600;padding:10px 16px;border-radius:8px;margin:10px 0;border-left:5px solid {MINT}}}
div[data-testid="stMetric"]{{background:{NAVY};border-radius:8px;padding:10px;border-left:3px solid {TEAL}}}
div[data-testid="stMetric"] label{{color:{MINT} !important;font-size:11px !important}}
div[data-testid="stMetric"] div{{color:{WHITE} !important}}
</style>""", unsafe_allow_html=True)

BERKELEY={
    "W1-R":  dict(n=20,cat="Residential",ppb=1*2.5, lrt="wood",  rv=280000,    params=[(0.30,0.64),(0.60,0.64),(1.20,0.64),(2.40,0.64)]),
    "W2-R":  dict(n=8, cat="Residential",ppb=8*2.5, lrt="wood",  rv=2240000,   params=[(0.28,0.64),(0.56,0.64),(1.12,0.64),(2.24,0.64)]),
    "C1L-R": dict(n=3, cat="Residential",ppb=6*2.5, lrt="conc",  rv=1680000,   params=[(0.20,0.64),(0.40,0.64),(0.80,0.64),(1.60,0.64)]),
    "S1L-R": dict(n=2, cat="Residential",ppb=6*2.5, lrt="steel", rv=1680000,   params=[(0.15,0.64),(0.25,0.64),(0.75,0.64),(1.44,0.64)]),
    "S1M-R": dict(n=1, cat="Residential",ppb=20*2.5,lrt="steel", rv=5600000,   params=[(0.12,0.64),(0.22,0.64),(0.75,0.64),(1.38,0.64)]),
    "S4L-R": dict(n=2, cat="Residential",ppb=6*2.5, lrt="steel", rv=1680000,   params=[(0.20,0.64),(0.35,0.64),(0.85,0.64),(1.28,0.64)]),
    "MH-R":  dict(n=2, cat="Residential",ppb=1*2.5, lrt="mh",   rv=60000,     params=[(0.13,0.64),(0.26,0.64),(0.50,0.64),(0.90,0.64)]),
    "HOSP":  dict(n=1, cat="Hospital",   ppb=0,     lrt="essfac",rv=15000000,  params=[(0.30,0.50),(0.60,0.50),(1.20,0.50),(2.40,0.50)]),
    "FIRE1": dict(n=1, cat="Fire Stn",   ppb=0,     lrt="essfac",rv=5000000,   params=[(0.30,0.50),(0.60,0.50),(1.20,0.50),(2.40,0.50)]),
    "FIRE2": dict(n=1, cat="Fire Stn",   ppb=0,     lrt="essfac",rv=5000000,   params=[(0.30,0.50),(0.60,0.50),(1.20,0.50),(2.40,0.50)]),
    "FIRE3": dict(n=1, cat="Fire Stn",   ppb=0,     lrt="essfac",rv=5000000,   params=[(0.30,0.50),(0.60,0.50),(1.20,0.50),(2.40,0.50)]),
    "POLICE":dict(n=1, cat="Police",     ppb=0,     lrt="essfac",rv=8000000,   params=[(0.30,0.50),(0.60,0.50),(1.20,0.50),(2.40,0.50)]),
    "EPPS":  dict(n=1, cat="Electric",   ppb=0,     lrt="infra", rv=50000000,  params=[(0.13,0.55),(0.24,0.55),(0.67,0.55),(1.20,0.55)]),
    "PWTS":  dict(n=1, cat="Water",      ppb=0,     lrt="infra", rv=15000000,  params=[(0.18,0.40),(0.30,0.40),(0.80,0.45),(1.45,0.45)]),
    "PWTL":  dict(n=1, cat="Water",      ppb=0,     lrt="infra", rv=25000000,  params=[(0.21,0.40),(0.35,0.40),(0.84,0.45),(1.55,0.45)]),
    "BRIDGE":dict(n=1, cat="Bridge",     ppb=0,     lrt="infra", rv=20000000,  params=[(0.25,0.60),(0.50,0.60),(0.90,0.60),(1.60,0.60)]),
}
COALINGA={
    "W1-R":  dict(n=18,cat="Residential",ppb=1*2.9, lrt="wood",  rv=180000,    params=[(0.18,0.64),(0.36,0.64),(0.72,0.64),(1.44,0.64)]),
    "RM2L-R":dict(n=6, cat="Residential",ppb=6*2.9, lrt="mason", rv=1080000,   params=[(0.10,0.64),(0.20,0.64),(0.50,0.64),(1.00,0.64)]),
    "C2L-R": dict(n=4, cat="Residential",ppb=6*2.9, lrt="conc",  rv=1080000,   params=[(0.12,0.64),(0.23,0.64),(0.60,0.64),(1.20,0.64)]),
    "MH-R":  dict(n=5, cat="Residential",ppb=1*2.9, lrt="mh",   rv=60000,     params=[(0.10,0.64),(0.20,0.64),(0.40,0.64),(0.80,0.64)]),
    "URM-R": dict(n=3, cat="Residential",ppb=4*2.9, lrt="mason", rv=720000,    params=[(0.08,0.64),(0.15,0.64),(0.35,0.64),(0.60,0.64)]),
    "HOSP":  dict(n=1, cat="Hospital",   ppb=0,     lrt="essfac",rv=8000000,   params=[(0.20,0.50),(0.40,0.50),(0.80,0.50),(1.60,0.50)]),
    "FIRE":  dict(n=1, cat="Fire Stn",   ppb=0,     lrt="essfac",rv=3000000,   params=[(0.15,0.50),(0.30,0.50),(0.70,0.50),(1.40,0.50)]),
    "POLICE":dict(n=1, cat="Police",     ppb=0,     lrt="essfac",rv=3000000,   params=[(0.15,0.50),(0.30,0.50),(0.70,0.50),(1.40,0.50)]),
    "EPPS":  dict(n=1, cat="Electric",   ppb=0,     lrt="infra", rv=20000000,  params=[(0.10,0.55),(0.18,0.55),(0.50,0.55),(0.90,0.55)]),
    "PWTS":  dict(n=1, cat="Water",      ppb=0,     lrt="infra", rv=10000000,  params=[(0.12,0.40),(0.22,0.40),(0.60,0.45),(1.10,0.45)]),
    "PWTL":  dict(n=1, cat="Water",      ppb=0,     lrt="infra", rv=15000000,  params=[(0.14,0.40),(0.26,0.40),(0.64,0.45),(1.18,0.45)]),
    "BRIDGE":dict(n=1, cat="Bridge",     ppb=0,     lrt="infra", rv=10000000,  params=[(0.15,0.60),(0.30,0.60),(0.65,0.60),(1.20,0.60)]),
    "SCHOOL":dict(n=2, cat="School",     ppb=0,     lrt="essfac",rv=5000000,   params=[(0.12,0.64),(0.24,0.64),(0.60,0.64),(1.20,0.64)]),
    "GAS":   dict(n=1, cat="Gas Stn",    ppb=0,     lrt="infra", rv=2000000,   params=[(0.12,0.60),(0.25,0.60),(0.65,0.60),(1.20,0.60)]),
}
BK_IMP={"C1L-R":[(0.24,0.64),(0.45,0.64),(0.90,0.64),(1.55,0.64)],
         "S1L-R":[(0.19,0.64),(0.31,0.64),(0.80,0.64),(1.49,0.64)],
         "S1M-R":[(0.14,0.64),(0.26,0.64),(0.80,0.64),(1.43,0.64)],
         "S4L-R":[(0.24,0.64),(0.39,0.64),(0.90,0.64),(1.33,0.64)],
         "PWTS": [(0.22,0.40),(0.35,0.40),(0.87,0.45),(1.57,0.45)]}
CL_IMP={"MH-R":  [(0.15,0.64),(0.28,0.64),(0.55,0.64),(0.95,0.64)],
         "URM-R": [(0.12,0.64),(0.22,0.64),(0.50,0.64),(0.80,0.64)],
         "RM2L-R":[(0.14,0.64),(0.26,0.64),(0.60,0.64),(1.10,0.64)],
         "EPPS":  [(0.13,0.55),(0.24,0.55),(0.62,0.55),(1.05,0.55)]}

def p_exc(pga,theta,beta):
    return float(norm.cdf(np.log(max(pga,1e-9)/theta)/beta))

def discrete_ds(pga,params):
    c=[p_exc(pga,params[i][0],params[i][1]) for i in range(4)]
    return [1-c[0],c[0]-c[1],c[1]-c[2],c[2]-c[3],c[3]]

def node_func(pga,params):
    return p_exc(pga,params[2][0],params[2][1])<=0.50

def get_params(name,d,imp,retro):
    return imp[name] if retro and name in imp else d["params"]

def compute(nodes,imp,pga,retro=False):
    tot=sum(d["n"] for d in nodes.values())
    tot_pop=sum(d["n"]*d["ppb"] for d in nodes.values())
    func=0;disp=0.0;loss=0.0;ds=[0.0]*5;nstat={}
    for nm,d in nodes.items():
        p=get_params(nm,d,imp,retro)
        probs=discrete_ds(pga,p)
        fn=node_func(pga,p)
        func+=d["n"]*int(fn)
        disp+=d["n"]*(probs[3]+probs[4])*d["ppb"]
        lr=LOSS_RATIOS[d["lrt"]]
        mdr=sum(probs[i+1]*lr[i] for i in range(4))
        loss+=d["n"]*mdr*d["rv"]
        for k in range(5): ds[k]+=d["n"]*probs[k]
        nstat[nm]=dict(fn=fn,cat=d["cat"],n=d["n"],
                       p3=p_exc(pga,p[2][0],p[2][1]))
    return dict(R=func/tot,disp=disp,pct_disp=disp/tot_pop*100 if tot_pop else 0,
                loss=loss,ds=ds,tot=tot,pct_ds3=sum(ds[3:])/tot*100,nstat=nstat)

def crit(nodes,imp,pga,retro):
    tot=sum(d["n"] for d in nodes.values())
    base=compute(nodes,imp,pga,retro)["R"]
    rows=[]
    for nm,d in nodes.items():
        fn=sum(d2["n"]*int(node_func(pga,get_params(n2,d2,imp,retro)))
               for n2,d2 in nodes.items() if n2!=nm)
        rows.append(dict(node=nm,cat=d["cat"],dR=(base-fn/tot)*100))
    return sorted(rows,key=lambda x:-x["dR"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='background:{NAVY};padding:14px;border-radius:8px;text-align:center'>"
                f"<span style='color:{MINT};font-size:20px;font-weight:800'>\U0001f3d9\ufe0f CIVE 580</span><br>"
                f"<span style='color:{WHITE};font-size:12px'>Seismic Resilience Dashboard</span></div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Scenario Presets**")
    c1,c2,c3=st.columns(3)
    preset=None
    if c1.button("Low\n0.15g",use_container_width=True): preset=0.15
    if c2.button("Mod\n0.44g",use_container_width=True): preset=0.44
    if c3.button("High\n0.75g",use_container_width=True): preset=0.75
    if "pga" not in st.session_state: st.session_state.pga=0.30
    if preset: st.session_state.pga=preset
    pga=st.slider("**PGA (g)**",0.05,0.75,st.session_state.pga,0.01,format="%.2fg")
    st.session_state.pga=pga
    st.markdown("---")
    retro=st.toggle("\U0001f527 **Post-Retrofit Network**",value=False)
    st.markdown("---")
    st.markdown(f"<div style='background:{NAVY};padding:10px;border-radius:6px;color:{WHITE};font-size:11px'>"
                f"<b style='color:{MINT}'>How values are computed</b><br><br>"
                f"Exact analytical evaluation at every PGA:<br><br>"
                f"<code style='color:{MINT}'>P(DS\u2265ds|PGA) = \u03a6[ln(PGA/\u03b8)/\u03b2]</code><br><br>"
                f"No interpolation \u2014 smooth lognormal CDF.</div>",unsafe_allow_html=True)
    st.caption("Berkeley avg HH: 2.5 | Coalinga: 2.9\nDS\u22653 \u2192 uninhabitable (HAZUS \u00a713)")

# ── Compute ───────────────────────────────────────────────────────────────────
bk=compute(BERKELEY,BK_IMP,pga,retro)
cl=compute(COALINGA,CL_IMP,pga,retro)
bk_cr=crit(BERKELEY,BK_IMP,pga,retro)
cl_cr=crit(COALINGA,CL_IMP,pga,retro)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"<h2 style='color:{NAVY};margin-bottom:4px'>Community Seismic Resilience Dashboard</h2>"
            f"<p style='color:{TEAL};margin-top:0;font-size:13px'>"
            f"Berkeley (High-Code) vs Coalinga (Low-Code) &nbsp;|&nbsp; "
            f"PGA = <b>{pga:.2f}g</b> &nbsp;|&nbsp; "
            f"{'Post-Retrofit' if retro else 'Pre-Retrofit'} Network</p>",
            unsafe_allow_html=True)

# ── Row 1: Headline metrics ───────────────────────────────────────────────────
st.markdown(f"<div class='section-hdr'>\U0001f4ca Headline Metrics at PGA = {pga:.2f}g</div>",
            unsafe_allow_html=True)
cbk,ccl=st.columns(2)
with cbk:
    st.markdown("<div class='city-hdr-bk'>\U0001f535 BERKELEY \u2014 High-Code</div>",unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Resilience R",f"{bk['R']*100:.1f}%")
    m2.metric("Displaced (people)",f"{bk['disp']:.0f}",f"{bk['pct_disp']:.1f}% of pop")
    m3.metric("Severe Damage (DS\u22653)",f"{bk['pct_ds3']:.1f}%")
    m4.metric("Economic Loss (USD)",f"${bk['loss']/1e6:.1f}M")
with ccl:
    st.markdown("<div class='city-hdr-cl'>\U0001f534 COALINGA \u2014 Low-Code</div>",unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Resilience R",f"{cl['R']*100:.1f}%")
    m2.metric("Displaced (people)",f"{cl['disp']:.0f}",f"{cl['pct_disp']:.1f}% of pop")
    m3.metric("Severe Damage (DS\u22653)",f"{cl['pct_ds3']:.1f}%")
    m4.metric("Economic Loss (USD)",f"${cl['loss']/1e6:.1f}M")

gap_R=(bk["R"]-cl["R"])*100; gap_d=cl["pct_disp"]-bk["pct_disp"]
st.markdown(f"<div class='callout-banner'>"
            f"PGA = {pga:.2f}g \u2014 Berkeley R = {bk['R']*100:.1f}% vs Coalinga R = {cl['R']*100:.1f}% "
            f"(gap: {gap_R:.1f} pp) &nbsp;|&nbsp; Displacement gap: {gap_d:.1f} pp"
            f"</div>",unsafe_allow_html=True)

# ── Row 2: Resilience curve + Node grid ──────────────────────────────────────
st.markdown("<div class='section-hdr'>\U0001f4c8 Resilience Curve & Network Status</div>",unsafe_allow_html=True)
c_curve,c_nodes=st.columns([1.15,0.85])

with c_curve:
    sweep=np.linspace(0.05,0.75,300)
    bkR=[compute(BERKELEY,BK_IMP,p,retro)["R"]*100 for p in sweep]
    clR=[compute(COALINGA,CL_IMP,p,retro)["R"]*100 for p in sweep]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sweep,y=bkR,name="Berkeley",
        line=dict(color=BK_COL,width=3),fill="tozeroy",fillcolor="rgba(6,90,130,0.08)"))
    fig.add_trace(go.Scatter(x=sweep,y=clR,name="Coalinga",
        line=dict(color=CL_COL,width=3),fill="tozeroy",fillcolor="rgba(198,40,40,0.08)"))
    fig.add_vline(x=pga,line_dash="dash",line_color=MINT,line_width=2)
    fig.add_annotation(x=pga,y=107,text=f"\u2190 {pga:.2f}g",
        font=dict(color=MINT,size=11),showarrow=False)
    fig.add_trace(go.Scatter(x=[pga],y=[bk["R"]*100],mode="markers",
        marker=dict(color=BK_COL,size=12),showlegend=False))
    fig.add_trace(go.Scatter(x=[pga],y=[cl["R"]*100],mode="markers",
        marker=dict(color=CL_COL,size=12),showlegend=False))
    fig.update_layout(title="Resilience R vs PGA",xaxis_title="PGA (g)",yaxis_title="R (%)",
        yaxis=dict(range=[0,110]),xaxis=dict(range=[0.05,0.75]),
        paper_bgcolor="white",plot_bgcolor=LGRAY,
        legend=dict(orientation="h",y=1.12),margin=dict(l=40,r=20,t=50,b=40),height=310)
    st.plotly_chart(fig,use_container_width=True)

with c_nodes:
    def node_fig(nodes,imp,pga,retro,title,ccol):
        nm_l,x_l,y_l,col_l,sz_l,hov_l=[],[],[],[],[],[]
        cats=list(dict.fromkeys(d["cat"] for d in nodes.values()))
        for i,(nm,d) in enumerate(nodes.items()):
            p=get_params(nm,d,imp,retro)
            fn=node_func(pga,p); p3=p_exc(pga,p[2][0],p[2][1])
            nm_l.append(nm);x_l.append(i);y_l.append(cats.index(d["cat"]))
            col_l.append("#43A047" if fn else "#E53935")
            sz_l.append(max(10,min(38,d["n"]*3)))
            _status = '✅ Functional' if fn else '❌ Failed'
            hov_l.append(f"<b>{nm}</b><br>{d['cat']}<br>n={d['n']}<br>P(DS≥3)={p3*100:.1f}%<br>{_status}")
        fig=go.Figure(go.Scatter(x=x_l,y=y_l,mode="markers+text",text=nm_l,
            textposition="top center",textfont=dict(size=7,color=NAVY),
            marker=dict(color=col_l,size=sz_l,line=dict(color="white",width=1.5)),
            hovertemplate="%{customdata}<extra></extra>",customdata=hov_l))
        fig.update_layout(title=dict(text=title,font=dict(color=ccol,size=12)),
            xaxis=dict(visible=False),yaxis=dict(visible=False),
            paper_bgcolor="white",plot_bgcolor=LGRAY,
            margin=dict(l=8,r=8,t=32,b=8),height=148,showlegend=False)
        return fig
    st.plotly_chart(node_fig(BERKELEY,BK_IMP,pga,retro,
        "Berkeley  (\U0001f7e2 functional / \U0001f534 failed)",BK_COL),use_container_width=True)
    st.plotly_chart(node_fig(COALINGA,CL_IMP,pga,retro,
        "Coalinga  (\U0001f7e2 functional / \U0001f534 failed)",CL_COL),use_container_width=True)

# ── Row 3: Damage distribution + Displacement by type ────────────────────────
st.markdown("<div class='section-hdr'>\U0001f3d7\ufe0f Building Damage & Population Displacement</div>",unsafe_allow_html=True)
c_dmg,c_disp=st.columns(2)

with c_dmg:
    DS_L=["DS1 Slight","DS2 Moderate","DS3 Extensive","DS4 Complete"]
    tbk=sum(d["n"] for d in BERKELEY.values()); tcl=sum(d["n"] for d in COALINGA.values())
    bp=[v/tbk*100 for v in bk["ds"][1:]]; cp=[v/tcl*100 for v in cl["ds"][1:]]
    fig=go.Figure()
    for i,(lb,color) in enumerate(zip(DS_L,DS_COLORS[1:])):
        fig.add_trace(go.Bar(name=lb,x=["Berkeley","Coalinga"],y=[bp[i],cp[i]],
            marker_color=color,text=[f"{bp[i]:.1f}%",f"{cp[i]:.1f}%"],
            textposition="inside",insidetextanchor="middle",
            textfont=dict(size=9,color="white")))
    fig.update_layout(barmode="stack",title="Building Damage Distribution (%)",
        yaxis_title="% buildings",paper_bgcolor="white",plot_bgcolor=LGRAY,
        legend=dict(orientation="h",y=-0.28,font=dict(size=9)),
        margin=dict(l=40,r=20,t=45,b=85),height=330)
    st.plotly_chart(fig,use_container_width=True)

with c_disp:
    def disp_rows(nodes,imp,pga,retro):
        r=[]
        for nm,d in nodes.items():
            if d["ppb"]==0: continue
            p=get_params(nm,d,imp,retro); probs=discrete_ds(pga,p)
            ppl=d["n"]*(probs[3]+probs[4])*d["ppb"]
            if ppl>0.05: r.append(dict(node=nm,ppl=ppl))
        return sorted(r,key=lambda x:-x["ppl"])
    bdr=disp_rows(BERKELEY,BK_IMP,pga,retro)
    cdr=disp_rows(COALINGA,CL_IMP,pga,retro)
    fig=go.Figure()
    if bdr:
        fig.add_trace(go.Bar(name="Berkeley",x=[r["node"] for r in bdr],
            y=[r["ppl"] for r in bdr],marker_color=BK_COL,opacity=0.9,
            text=[f"{r['ppl']:.0f}" for r in bdr],textposition="outside",textfont=dict(size=9)))
    if cdr:
        fig.add_trace(go.Bar(name="Coalinga",x=[r["node"] for r in cdr],
            y=[r["ppl"] for r in cdr],marker_color=CL_COL,opacity=0.9,
            text=[f"{r['ppl']:.0f}" for r in cdr],textposition="outside",textfont=dict(size=9)))
    fig.update_layout(barmode="group",title="Displaced Persons by Building Type",
        yaxis_title="Displaced persons",paper_bgcolor="white",plot_bgcolor=LGRAY,
        legend=dict(orientation="h",y=1.12),margin=dict(l=40,r=20,t=45,b=40),height=330)
    if not bdr and not cdr:
        st.info("No displaced persons at this PGA level.")
    else:
        st.plotly_chart(fig,use_container_width=True)

# ── Row 4: Node criticality ───────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>\U0001f3af Node Criticality Ranking (\u0394R when forced to fail)</div>",
            unsafe_allow_html=True)
c_bkc,c_clc=st.columns(2)

def crit_fig(rows,title,ccol):
    top=[r for r in rows if r["dR"]>0.001][:10]
    if not top:
        return go.Figure().update_layout(title=f"{title} \u2014 all nodes failed",
            paper_bgcolor="white",height=290)
    cols=[CAT_COLOR.get(r["cat"],"#607D8B") for r in top]
    fig=go.Figure(go.Bar(x=[r["dR"] for r in top][::-1],y=[r["node"] for r in top][::-1],
        orientation="h",marker_color=cols[::-1],
        text=[f"{r['dR']:.1f} pp" for r in top][::-1],
        textposition="outside",textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>\u0394R = %{x:.2f} pp<extra></extra>"))
    fig.update_layout(title=dict(text=title,font=dict(color=ccol,size=13)),
        xaxis_title="\u0394R (pp)",paper_bgcolor="white",plot_bgcolor=LGRAY,
        margin=dict(l=80,r=60,t=45,b=40),height=320,
        xaxis=dict(range=[0,max(r["dR"] for r in top)*1.3]))
    return fig

with c_bkc: st.plotly_chart(crit_fig(bk_cr,"Berkeley \u2014 Node Criticality",BK_COL),use_container_width=True)
with c_clc: st.plotly_chart(crit_fig(cl_cr,"Coalinga \u2014 Node Criticality",CL_COL),use_container_width=True)

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:{TEAL};font-size:11px'>"
            f"CIVE 580 Community Resilience &nbsp;|&nbsp; HAZUS 6.1 lognormal fragility "
            f"P(DS\u2265ds|PGA) = \u03a6[ln(PGA/\u03b8)/\u03b2] &nbsp;|&nbsp; "
            f"Exact analytical evaluation \u2014 no interpolation</div>",unsafe_allow_html=True)
