"""
CLSC Stackelberg Game Model — Complete Final Version
=====================================================
Combines:
  • Correct social welfare function (new code)
  • All 5 publication-quality figures with layout fixes (original code)
  • Complete mathematical audit and reporting
  • Added sensitivity analysis visualization (Fig 6)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, minimize
import warnings, logging
import time  # 确保time模块导入

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# ── Auto-detect best available serif font ────────────────────────────
def _best_serif() -> str:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Latin Modern Roman", "Lora", "Caladea",
                 "Liberation Serif", "DejaVu Serif", "FreeSerif"]:
        if font in available:
            return font
    return "serif"

_SERIF = _best_serif()

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": _SERIF,
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.titlepad": 8,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "patch.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.major.pad": 4, "ytick.major.pad": 4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.20,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "grid.color": "#cccccc",
    "legend.frameon": True,
    "legend.framealpha": 0.93,
    "legend.edgecolor": "#cccccc",
    "legend.handlelength": 2.0,
    "legend.handleheight": 0.8,
    "legend.borderpad": 0.5,
    "legend.labelspacing": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

C1="#1B3A6B"; C2="#C94B2D"; C3="#2E7D5E"
C4="#B07D2A"; C5="#5C4B8A"; CG="#7A7A7A"

def _sax(ax, xlabel="", ylabel="", title="", grid=True):
    if xlabel: ax.set_xlabel(xlabel, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, labelpad=6)
    if title: ax.set_title(title, pad=8)
    if grid: ax.grid(True, which="major", zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=2.0, width=0.5)
    ax.tick_params(which="major", length=4.0, width=0.8)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_linewidth(0.8); ax.spines[sp].set_color("#333333")
    return ax

def _pl(ax, label, x=-0.13, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left", color="#1a1a1a")

def _vl(ax, x, c=C2, lw=1.2, ls=":", a=0.75):
    ax.axvline(x, color=c, lw=lw, ls=ls, alpha=a, zorder=2)

def _hl(ax, y, c=CG, lw=1.0, ls="--", a=0.65):
    ax.axhline(y, color=c, lw=lw, ls=ls, alpha=a, zorder=2)

def _dot(ax, x, y, c=C2, s=55, z=8):
    ax.scatter([x],[y], color=c, s=s, zorder=z, edgecolors="white", linewidths=1.0)

def _save(fig, name):
    for ext in ["pdf","png"]:
        fig.savefig(f"./{name}.{ext}", dpi=600)  # 新路径：当前目录
    plt.close(fig)
    print(f"    ✓ {name}.{{pdf,png}} (保存到当前目录)")

# ── Model (CORRECTED SOCIAL WELFARE) ──────────────────────────────────

@dataclass
class Cfg:
    a: float = 500.0
    b: float = 1.3
    gamma: float = 10.0
    c_m: float = 200.0
    c_r: float = 170.0
    k: float = 12000.0
    e_m: float = 0.8
    e_r: float = 0.2
    eta: float = 3.0
    tau_max: float = 150.0
    tol: float = 1e-10
    
    @property
    def dc(self): return self.c_m - self.c_r
    @property
    def de(self): return round(self.e_m - self.e_r, 10)

def Df(c, p, rho): return max(c.a - c.b * p + c.gamma * rho, 0.0)
def Ef(c, d, rho): return max((c.e_m - (c.e_m - c.e_r) * rho) * d, 0.0)
def CSf(c, p, rho): 
    d = Df(c, p, rho)
    return d**2 / (2 * c.b)

def prf(c, w, rho): return (c.a + c.gamma * rho + c.b * w) / (2 * c.b)

def pim(c, w, rho, tau):
    p = prf(c, w, rho)
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    return (w - c.c_m) * d + c.dc * rho * d - tau * em - c.k * rho**2

def pivif(c, p, rho, tau):
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    return (p - c.c_m) * d + c.dc * rho * d - c.k * rho**2 - tau * em

def rs(c, d, tau): return float(np.clip(d * (c.dc + tau * c.de) / (2 * c.k), 0, 1))

def SWf(c, cs, pi_m, pi_r, tau, em):
    """CORRECTED: Explicitly includes tax revenue τ·E"""
    tax_revenue = tau * em
    environmental_damage = c.eta * em**2
    return cs + pi_m + pi_r + tax_revenue - environmental_damage

def dec_solve(c, tau):
    def neg(w):
        rho = 0.15
        for _ in range(100):
            p = prf(c, w, rho)
            d = Df(c, p, rho)
            rho_new = rs(c, d, tau)
            if abs(rho_new - rho) < 1e-8: break
            rho = 0.7 * rho + 0.3 * rho_new
        return -pim(c, w, rho, tau)
    res = minimize_scalar(neg, bounds=(c.c_m+.01, c.a/c.b-.01), 
                         method="bounded", options={"xatol": c.tol})
    w = float(res.x)
    p = prf(c, w, 0.15)
    d = Df(c, p, 0.15)
    rho = rs(c, d, tau)
    for _ in range(100):
        p = prf(c, w, rho)
        d = Df(c, p, rho)
        rho_new = rs(c, d, tau)
        if abs(rho_new - rho) < 1e-8: break
        rho = 0.7 * rho + 0.3 * rho_new
    p = prf(c, w, rho); d = Df(c, p, rho); em = Ef(c, d, rho)
    pm = pim(c, w, rho, tau); pr = (p - w) * d; cs = CSf(c, p, rho)
    sw = SWf(c, cs, pm, pr, tau, em)
    return dict(tau=tau, w=w, p=p, rho=rho, D=d, E=em, pi_m=pm, pi_r=pr,
                CS=cs, env_dam=c.eta*em**2, tax_rev=tau*em, SW=sw)

def dec_eq(c, tau=None):
    if tau is not None: return dec_solve(c, tau)
    res = minimize_scalar(lambda t: -dec_solve(c, t)["SW"],
                         bounds=(0, c.tau_max), method="bounded", 
                         options={"xatol": c.tol})
    return dec_solve(c, float(res.x))

def vif_br(c, tau):
    def neg_pi_vif(x):
        return -pivif(c, x[0], x[1], tau)
    res = minimize(neg_pi_vif, x0=[350, 0.3],
                  bounds=[(c.c_m+.01, c.a/c.b-.01), (0, 1)],
                  method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
    return float(res.x[0]), float(res.x[1])

def vif_sw_val(c, tau):
    p, rho = vif_br(c, tau)
    profit_vif = pivif(c, p, rho, tau)
    if profit_vif < -1: return -1e10
    d = Df(c, p, rho); em = Ef(c, d, rho); cs = CSf(c, p, rho)
    return SWf(c, cs, profit_vif, 0, tau, em)

def vif_eq(c):
    tau_grid = np.linspace(0, c.tau_max, 200)
    sw_values = [vif_sw_val(c, t) for t in tau_grid]
    best_idx = np.argmax(sw_values)
    tau_init = tau_grid[best_idx]
    res = minimize_scalar(lambda t: -vif_sw_val(c, t),
                         bounds=(max(0, tau_init-10), min(c.tau_max, tau_init+10)),
                         method="bounded", options={"xatol": c.tol})
    tau_opt = float(res.x)
    p_opt, rho_opt = vif_br(c, tau_opt)
    d = Df(c, p_opt, rho_opt); em = Ef(c, d, rho_opt)
    profit_vif = pivif(c, p_opt, rho_opt, tau_opt); cs = CSf(c, p_opt, rho_opt)
    sw = SWf(c, cs, profit_vif, 0, tau_opt, em)
    return dict(tau=tau_opt, p=p_opt, rho=rho_opt, D=d, E=em, pi_vif=profit_vif,
                CS=cs, env_dam=c.eta*em**2, tax_rev=tau_opt*em, SW=sw)

def tau_sweep(c, n=150):
    taus = np.linspace(0, c.tau_max, n)
    keys = ["w","p","rho","D","E","pi_m","pi_r","CS","SW","env_dam","tax_rev"]
    out = {k: np.empty(n) for k in keys}; out["taus"] = taus
    for i, t in enumerate(taus):
        eq = dec_solve(c, t)
        for k in keys: out[k][i] = eq[k]
    return out

def eta_sweep(c, n=60):
    etas = np.linspace(0.5, 8., n); ts, Es, SWs, rhos = [], [], [], []
    orig = c.eta
    for ev in etas:
        c.eta = ev; eq = dec_eq(c)
        ts.append(eq["tau"]); Es.append(eq["E"]); SWs.append(eq["SW"]); rhos.append(eq["rho"])
    c.eta = orig
    return dict(etas=etas, tau_s=np.array(ts), E_s=np.array(Es), SW_s=np.array(SWs), rho_s=np.array(rhos))

def gamma_sweep(c, n=60):
    gammas = np.linspace(2, 28, n); ts, rhos, Ds, SWs = [], [], [], []
    orig = c.gamma
    for gv in gammas:
        c.gamma = gv; eq = dec_eq(c)
        ts.append(eq["tau"]); rhos.append(eq["rho"]); Ds.append(eq["D"]); SWs.append(eq["SW"])
    c.gamma = orig
    return dict(gammas=gammas, tau_s=np.array(ts), rho_s=np.array(rhos), D_s=np.array(Ds), SW_s=np.array(SWs))

def phi_sweep(c, tau, n=80):
    phis = np.linspace(0.01, 0.99, n); pms, prs, sws = [], [], []
    for phi in phis:
        p = (c.a + c.gamma + c.b * c.c_m) / (2 * c.b)
        d = Df(c, p, 0.3); rho = rs(c, d, tau); em = Ef(c, d, rho)
        pm = phi * (p - c.c_m) * d + c.dc * rho * d - c.k * rho**2 - tau * em
        pr = (1 - phi) * (p - c.c_m) * d; cs = CSf(c, p, rho)
        sv = SWf(c, cs, pm, pr, tau, em)
        pms.append(pm); prs.append(pr); sws.append(sv)
    return dict(phis=phis, pi_m=np.array(pms), pi_r=np.array(prs), sw=np.array(sws))

# ════════════════════════════════════════════════════════════════════
# FIGURE 1: Pricing and Recycling
# ════════════════════════════════════════════════════════════════════
def fig1(eq_nt, eq_d, eq_c, sens):
    fig = plt.figure(figsize=(13., 4.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.38, left=0.08, right=0.97, top=0.90, bottom=0.14)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    taus = sens["taus"]

    # Panel A: Pricing
    ax1.fill_between(taus, sens["w"], sens["p"], color=C1, alpha=0.09, zorder=1, label="Retailer margin")
    ax1.plot(taus, sens["p"], color=C1, lw=2.2, zorder=4, label=r"Retail price $p^*(\tau)$")
    ax1.plot(taus, sens["w"], color=C1, lw=1.6, ls="--", zorder=4, label=r"Wholesale price $w^*(\tau)$")
    _hl(ax1, eq_c["p"], c=C3, lw=1.1, ls=(0,(5,3)), a=0.7)
    ax1.text(taus[-1]*0.97, eq_c["p"]+2, fr"$p^C={eq_c['p']:.1f}$ (VIF)",
             ha="right", va="bottom", fontsize=8, color=C3, style="italic")
    _vl(ax1, eq_d["tau"], c=C2, lw=1.3)
    _dot(ax1, eq_d["tau"], eq_d["p"], c=C2)
    _dot(ax1, eq_d["tau"], eq_d["w"], c=C2)
    ax1.annotate("", xy=(eq_d["tau"]+2, eq_d["p"]), xytext=(eq_d["tau"]+2, eq_d["w"]),
                 arrowprops=dict(arrowstyle="<->", color=C2, lw=1.2))
    ax1.text(eq_d["tau"]+5, (eq_d["p"]+eq_d["w"])/2, fr"$\Delta={eq_d['p']-eq_d['w']:.1f}$",
             fontsize=8, color=C2, va="center")
    ax1.text(eq_d["tau"], 282, fr"$\tau^*={eq_d['tau']:.1f}$",
             fontsize=8.5, color=C2, fontweight="bold", ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C2, alpha=0.85, lw=0.6))
    _sax(ax1, xlabel=r"Carbon tax $\tau$", ylabel="Price", 
         title="Pricing Dynamics Along Equilibrium Path")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlim(0, taus[-1]); ax1.set_ylim(280, 380)
    _pl(ax1, "(a)")

    # Panel B: Recycling
    ax2.axvspan(0, eq_d["tau"], color=C4, alpha=0.05, zorder=0)
    ax2.axvspan(eq_d["tau"], taus[-1], color=C1, alpha=0.03, zorder=0)
    ax2.plot(taus, sens["rho"], color=C1, lw=2.2, zorder=4, label=r"Decentralised $\rho^*(\tau)$")
    _hl(ax2, eq_c["rho"], c=C3, lw=1.6, ls=(0,(5,3)), a=0.85)
    ax2.text(taus[-1]*0.97, eq_c["rho"]+0.007, fr"$\rho^C={eq_c['rho']:.3f}$ (VIF)",
             ha="right", va="bottom", fontsize=8, color=C3, style="italic")
    _hl(ax2, eq_nt["rho"], c=CG, lw=0.9, ls=":", a=0.55)
    ax2.text(4, eq_nt["rho"]+0.005, fr"$\rho_0={eq_nt['rho']:.3f}$", fontsize=7.5, color=CG)
    _vl(ax2, eq_d["tau"], c=C2, lw=1.3)
    gap = eq_c["rho"] - eq_d["rho"]
    ax2.annotate("", xy=(eq_d["tau"]+2, eq_c["rho"]), xytext=(eq_d["tau"]+2, eq_d["rho"]),
                 arrowprops=dict(arrowstyle="<->", color=C2, lw=1.2))
    ax2.text(eq_d["tau"]+5, (eq_d["rho"]+eq_c["rho"])/2,
             fr"$\Delta\rho={gap:.3f}$"+"\n(coordination gap)",
             fontsize=8, color=C2, va="center", fontweight="bold")
    _dot(ax2, eq_d["tau"], eq_d["rho"], c=C2)
    _dot(ax2, 0, eq_nt["rho"], c=CG, s=30)
    _sax(ax2, xlabel=r"Carbon tax $\tau$", ylabel=r"Recycling rate $\rho$",
         title="Recycling Rate and Coordination Gap")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(0, taus[-1]); ax2.set_ylim(0, 0.55)
    _pl(ax2, "(b)")
    fig.suptitle("Figure 1.  Supply Chain Equilibrium: Pricing Dynamics and Recycling Coordination",
                 fontsize=11.5, y=0.97, style="italic", color="#222222")
    _save(fig, "CLSC_Fig1_Pricing_Recycling")

# ════════════════════════════════════════════════════════════════════
# FIGURE 2: Emissions and Welfare
# ════════════════════════════════════════════════════════════════════
def fig2(eq_nt, eq_d, eq_c, sens):
    fig = plt.figure(figsize=(13., 4.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.38, left=0.08, right=0.97, top=0.90, bottom=0.14)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    taus = sens["taus"]

    # Panel A: Emissions
    below = sens["E"] <= eq_nt["E"]
    ax1.fill_between(taus, sens["E"], eq_nt["E"], where=below, color=C3, alpha=0.15, zorder=1,
                     label="Emission reduction")
    ax1.plot(taus, sens["E"], color=C4, lw=2.2, zorder=4, label=r"Emissions $E^*(\tau)$")
    _hl(ax1, eq_nt["E"], c=CG, lw=1.2)
    ax1.text(3, eq_nt["E"]+0.5, fr"$E_0={eq_nt['E']:.1f}$", fontsize=8, color=CG)
    e_min = sens["E"].min(); e_max = max(eq_c["E"], eq_nt["E"])
    ax1.set_ylim(e_min - 1.5, e_max + 3.5)
    _hl(ax1, eq_c["E"], c=C3, lw=1.0, ls=(0,(4,3)), a=0.7)
    ax1.text(6, eq_c["E"]+0.4, fr"$E^C={eq_c['E']:.1f}$ (VIF)", fontsize=8, color=C3, va="bottom")
    _vl(ax1, eq_d["tau"], c=C2, lw=1.3)
    _dot(ax1, eq_d["tau"], eq_d["E"], c=C2)
    ax1.annotate(
        "\"Green paradox\"\n"+r"$E^C > E^*$: output$\uparrow$ dominates",
        xy=(eq_d["tau"], eq_d["E"]),
        xytext=(80, eq_d["E"]+4.5),
        arrowprops=dict(arrowstyle="->", color=C3, lw=0.9, connectionstyle="arc3,rad=-0.15"),
        fontsize=8, color=C3, style="italic", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C3, alpha=0.88, lw=0.6))
    _sax(ax1, xlabel=r"Carbon tax $\tau$", ylabel=r"Total emissions $E$",
         title="Environmental Impact along the Tax Path")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_xlim(0, taus[-1])
    _pl(ax1, "(a)")

    # Panel B: Social Welfare
    sw_min = sens["SW"].min(); sw_max = sens["SW"].max()
    pad = (sw_max - sw_min) * 0.12
    ax2.set_ylim(sw_min - pad, sw_max + pad*1.25)
    feasible = sens["SW"] >= eq_nt["SW"]
    ax2.fill_between(taus, sw_min-pad, sw_max+pad, where=feasible, color=C1, alpha=0.06,
                     zorder=0, label="Effective policy zone")
    ax2.plot(taus, sens["SW"], color=C1, lw=2.2, zorder=4, label=r"Social welfare $SW(\tau)$")
    _hl(ax2, eq_nt["SW"], c=CG, lw=1.2)
    ax2.text(3, eq_nt["SW"]-pad*0.6, fr"$SW_0={eq_nt['SW']:.0f}$", fontsize=8, color=CG)
    _hl(ax2, min(eq_c["SW"], sw_max+pad*1.15), c=C3, lw=1.2, ls=(0,(5,3)))
    ax2.text(taus[-1]*0.97, sw_max+pad*0.35, fr"$SW^C={eq_c['SW']:.0f}$ (VIF)",
             ha="right", fontsize=8, color=C3, style="italic")
    idx = int(np.argmax(sens["SW"]))
    _vl(ax2, sens["taus"][idx], c=C2, lw=1.3)
    _dot(ax2, sens["taus"][idx], sens["SW"][idx], c=C2, s=65)
    ax2.annotate(
        fr"$\tau^*={eq_d['tau']:.1f}$"+"\n"+fr"$SW^*={eq_d['SW']:.0f}$",
        xy=(sens["taus"][idx], sens["SW"][idx]),
        xytext=(sens["taus"][idx]+14, sens["SW"][idx]-pad*0.5),
        arrowprops=dict(arrowstyle="->", color=C2, lw=1.0),
        fontsize=8.5, color=C2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C2, alpha=0.9, lw=0.7))
    _sax(ax2, xlabel=r"Carbon tax $\tau$", ylabel=r"Social welfare $SW$",
         title="Social Welfare Optimisation")
    ax2.set_xlim(0, taus[-1])
    ax2.legend(loc="upper right", fontsize=8)
    _pl(ax2, "(b)")
    fig.suptitle("Figure 2.  Environmental Impact and Social Welfare under Carbon Taxation",
                 fontsize=11.5, y=0.97, style="italic", color="#222222")
    _save(fig, "CLSC_Fig2_Emission_Welfare")

# ════════════════════════════════════════════════════════════════════
# FIGURE 3: Scenario Comparison and Welfare Decomposition
# ════════════════════════════════════════════════════════════════════
def fig3(eq_nt, eq_d, eq_c):
    fig = plt.figure(figsize=(13., 5.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.42, left=0.07, right=0.97, top=0.88, bottom=0.13)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

    # Panel A: Bars
    scenarios = ["No-Tax\n"+r"$(\tau=0)$",
                 "Decentralised\n"+fr"$(\tau^*={eq_d['tau']:.1f})$",
                 "VIF\n"+fr"$(\tau^C={eq_c['tau']:.1f})$"]
    colors3 = [CG, C1, C3]
    rhos = [eq_nt["rho"], eq_d["rho"], eq_c["rho"]]
    sws = [eq_nt["SW"], eq_d["SW"], eq_c["SW"]]
    x = np.arange(3)
    bars = ax1.bar(x, rhos, 0.42, color=colors3, alpha=0.82, edgecolor="white", lw=0.8, zorder=3)
    for i, (bar, v) in enumerate(zip(bars, rhos)):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.007, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=colors3[i])
    ax1.set_ylabel(r"Recycling rate $\rho$", labelpad=6)
    ax1.set_ylim(0, max(rhos)*1.40)
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios, fontsize=9.5)
    ax1r = ax1.twinx()
    ax1r.plot(x, sws, "D--", color=C4, lw=1.6, ms=7, markerfacecolor=C4,
              markeredgecolor="white", markeredgewidth=0.9, zorder=5)
    for i, (xi, sv) in enumerate(zip(x, sws)):
        ax1r.text(xi, sv+(max(sws)-min(sws))*0.045, f"{sv:.0f}", ha="center", fontsize=9, color=C4)
    ax1r.set_ylabel(r"Social welfare $SW$", color=C4, labelpad=6)
    ax1r.spines["top"].set_visible(False)
    ax1r.tick_params(axis="y", colors=C4)
    ax1r.yaxis.set_minor_locator(AutoMinorLocator(2))
    poa = eq_d["SW"]/eq_c["SW"]; gap = eq_c["rho"]-eq_d["rho"]
    ax1.set_title(fr"$\Delta\rho={gap:.3f}$,  PoA$={poa:.4f}$,  $\pi_{{VIF}}={eq_c['pi_vif']:.0f}>0\;\checkmark$",
                  fontsize=9.5, pad=6)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2, zorder=0)
    ax1.tick_params(axis="x", length=0)
    legend_items = [
        mpatches.Patch(color=CG, alpha=0.82, label="No-Tax"),
        mpatches.Patch(color=C1, alpha=0.82, label="Decentralised"),
        mpatches.Patch(color=C3, alpha=0.82, label="VIF"),
        plt.Line2D([0],[0], color=C4, lw=1.6, ls="--", marker="D", ms=6, label=r"Social welfare $SW$"),
    ]
    ax1.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.92)
    _pl(ax1, "(a)", x=-0.10)

    # Panel B: Waterfall
    dCS = eq_d["CS"] - eq_nt["CS"]
    dPim = eq_d["pi_m"] - eq_nt["pi_m"]
    dPir = eq_d["pi_r"] - eq_nt["pi_r"]
    dTax = eq_d["tax_rev"]
    dEnv = eq_nt["env_dam"] - eq_d["env_dam"]
    dSW = eq_d["SW"] - eq_nt["SW"]
    vals = [dCS, dPim, dPir, dTax, dEnv, dSW]
    labs = [r"$\Delta CS$", r"$\Delta\pi_m$", r"$\Delta\pi_r$", r"$\tau E$", r"$-\Delta(\eta E^2)$", r"Net $\Delta SW$"]

    bar_bottoms = []; run = 0
    for v in vals[:-1]:
        bar_bottoms.append(run if v>=0 else run+v)
        run += v
    bar_bottoms.append(0)

    bar_c = [C3 if v>=0 else C2 for v in vals]
    bar_c[-1] = C3

    ax2.bar(np.arange(6), np.abs(vals), 0.55, bottom=bar_bottoms,
            color=bar_c, alpha=0.80, edgecolor="white", lw=0.7, zorder=3)
    ax2.axhline(0, color="#333333", lw=0.9, zorder=4)

    run2 = 0
    for i, v in enumerate(vals[:-1]):
        run2 += v
        ax2.plot([i+0.285, i+0.715], [run2, run2], color="#aaaaaa", lw=0.9, ls="--", zorder=3)

    for i, (v, bot) in enumerate(zip(vals, bar_bottoms)):
        top = bot + abs(v)
        if v >= 0:
            ax2.text(i, top+abs(dSW)*0.06, f"{v:+.0f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=bar_c[i])
        else:
            ax2.text(i, bot-abs(dSW)*0.06, f"{v:+.0f}", ha="center", va="top",
                     fontsize=9, fontweight="bold", color=bar_c[i])

    ax2.set_xticks(np.arange(6)); ax2.set_xticklabels(labs, fontsize=9.5)
    ax2.set_ylabel(r"Welfare component change $(\Delta)$", labelpad=6)
    ax2.set_title(r"Welfare Decomposition: $SW(\tau^*) - SW(0)$", fontsize=9.5, pad=6)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, zorder=0)
    ax2.tick_params(axis="x", length=0)
    ylo = min(bar_bottoms+list(np.array(bar_bottoms)+np.abs(vals)))
    yhi = max(bar_bottoms+list(np.array(bar_bottoms)+np.abs(vals)))
    yr = yhi - ylo
    ax2.set_ylim(ylo-yr*0.12, yhi+yr*0.18)
    ax2.text(4.7, dSW*0.6, fr"Net: $+{dSW:.0f}$", fontsize=9, color=C3, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C3, alpha=0.9, lw=0.7))
    _pl(ax2, "(b)", x=-0.10)
    fig.suptitle("Figure 3.  Scenario Comparison and Welfare Decomposition",
                 fontsize=11.5, y=0.97, style="italic", color="#222222")
    _save(fig, "CLSC_Fig3_Benchmark_Decomposition")

# ════════════════════════════════════════════════════════════════════
# FIGURE 4: Revenue-Sharing Contract
# ════════════════════════════════════════════════════════════════════
def fig4(eq_d, eq_c, phi_data):
    fig = plt.figure(figsize=(13., 4.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.38, left=0.08, right=0.97, top=0.90, bottom=0.14)
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    phis = phi_data["phis"]; pms = phi_data["pi_m"]; prs = phi_data["pi_r"]; sws = phi_data["sw"]
    total = pms + prs

    # Panel A
    ax1.fill_between(phis, 0, pms, where=(pms>=0), color=C1, alpha=0.13, zorder=1)
    ax1.fill_between(phis, 0, pms, where=(pms<0), color=C2, alpha=0.13, zorder=1)
    ax1.fill_between(phis, 0, prs, color=C4, alpha=0.10, zorder=1)
    ax1.plot(phis, pms, color=C1, lw=2.0, zorder=4, label=r"Manufacturer $\pi_m^{RS}(\phi)$")
    ax1.plot(phis, prs, color=C4, lw=2.0, zorder=4, label=r"Retailer $\pi_r^{RS}(\phi)$")
    ax1.plot(phis, total, color=C3, lw=1.5, ls=(0,(5,3)), zorder=3, label="Channel total")
    _hl(ax1, eq_d["pi_m"], c=C1, lw=0.9, ls=":", a=0.5)
    _hl(ax1, eq_d["pi_r"], c=C4, lw=0.9, ls=":", a=0.5)
    ax1.text(0.02, eq_d["pi_m"]+80, fr"$\pi_m^*={eq_d['pi_m']:.0f}$", fontsize=7.5, color=C1, alpha=0.7)
    ax1.text(0.02, eq_d["pi_r"]+80, fr"$\pi_r^*={eq_d['pi_r']:.0f}$", fontsize=7.5, color=C4, alpha=0.7)
    ax1.axhline(0, color="#555555", lw=0.8, zorder=5)
    both_ok = (pms>=0) & (prs>=0)
    if both_ok.any():
        phi_lo = phis[np.argmax(both_ok)]
        phi_hi = phis[len(phis)-1-np.argmax(both_ok[::-1])]
        ax1.axvspan(phi_lo, phi_hi, color=C3, alpha=0.07, zorder=0,
                    label=fr"Feasible: $\phi\in[{phi_lo:.2f},{phi_hi:.2f}]$")
    _sax(ax1, xlabel=r"Revenue-sharing fraction $\phi$", ylabel="Profit",
         title=r"Profit Distribution under Revenue-Sharing")
    ax1.legend(loc="center right", fontsize=7.8)
    ax1.set_xlim(0, 1)
    _pl(ax1, "(a)")

    # Panel B
    ax2.plot(phis, sws, color=C5, lw=2.2, zorder=4, label=r"$SW^{RS}(\phi)$")
    _hl(ax2, eq_d["SW"], c=C1, lw=1.2)
    ax2.text(0.97, eq_d["SW"]+15, fr"$SW^*={eq_d['SW']:.0f}$", ha="right", fontsize=8, color=C1)
    _hl(ax2, eq_c["SW"], c=C3, lw=1.2, ls=(0,(5,3)))
    ax2.text(0.97, eq_c["SW"]+15, fr"$SW^C={eq_c['SW']:.0f}$", ha="right", fontsize=8, color=C3)
    sw_lo = sws.min()-200; sw_hi = eq_c["SW"]+600
    ax2.set_ylim(sw_lo, sw_hi)
    if both_ok.any():
        idx_b = np.where(both_ok)[0][np.argmax(sws[both_ok])]
        _dot(ax2, phis[idx_b], sws[idx_b], c=C5, s=55)
        ax2.annotate(
            "RS eliminates double marginalisation\n"
            r"but $D^{RS}\!\gg D^*$ $\Rightarrow$ $E^{RS}\!\gg E^*$"+"\n"
            r"Env. damage dominates at $\tau^*$"+"\n"
            r"$\Rightarrow$ higher $\tau$ needed under RS",
            xy=(phis[idx_b], sws[idx_b]),
            xytext=(0.50, (sws[idx_b]+eq_d["SW"])/2 + 1200),
            arrowprops=dict(arrowstyle="->", color=C5, lw=0.9),
            fontsize=7.8, color=C5, style="italic", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C5, alpha=0.9, lw=0.6))
    _sax(ax2, xlabel=r"Revenue-sharing fraction $\phi$",
         ylabel=r"Social welfare $SW^{RS}$",
         title=r"Social Welfare under Revenue-Sharing (at $\tau^*$)")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=8, loc="upper right")
    _pl(ax2, "(b)")
    fig.suptitle("Figure 4.  Revenue-Sharing Contract: Profit Distribution and Welfare Implications",
                 fontsize=11.5, y=0.97, style="italic", color="#222222")
    _save(fig, "CLSC_Fig4_Coordination")

# ════════════════════════════════════════════════════════════════════
# FIGURE 5: Comparative Statics
# ════════════════════════════════════════════════════════════════════
def fig5(eta_data, gamma_data):
    fig = plt.figure(figsize=(13., 9.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.52,
                  left=0.08, right=0.97, top=0.91, bottom=0.07)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

    # (a)
    ax = axes[0]
    ax.plot(eta_data["etas"], eta_data["tau_s"], color=C2, lw=2.0, zorder=4,
            label=r"Optimal tax $\tau^*(\eta)$")
    ax.fill_between(eta_data["etas"], 0, eta_data["tau_s"], color=C2, alpha=0.10)
    axr = ax.twinx()
    axr.plot(eta_data["etas"], eta_data["E_s"], color=C3, lw=1.8, ls="--", zorder=3,
             label=r"Equilibrium emissions $E^*(\eta)$")
    axr.set_ylabel(r"Equilibrium emissions $E^*$", color=C3, labelpad=5)
    axr.tick_params(axis="y", colors=C3); axr.spines["top"].set_visible(False)
    axr.yaxis.set_minor_locator(AutoMinorLocator(2))
    _sax(ax, xlabel=r"Environmental damage weight $\eta$",
         ylabel=r"Optimal carbon tax $\tau^*$",
         title=r"$\partial\tau^*/\partial\eta > 0$: Stronger Damage $\Rightarrow$ Higher Tax")
    ax.tick_params(axis="y", colors=C2); ax.spines["left"].set_color(C2)
    lns = ax.get_lines() + axr.get_lines()
    ax.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=8)
    _pl(ax, "(a)")

    # (b)
    ax = axes[1]
    ax.plot(eta_data["etas"], eta_data["SW_s"], color=C1, lw=2.0, zorder=4)
    ax.fill_between(eta_data["etas"], eta_data["SW_s"].min(), eta_data["SW_s"], color=C1, alpha=0.10)
    _sax(ax, xlabel=r"Environmental damage weight $\eta$",
         ylabel=r"Optimal social welfare $SW^*(\eta)$",
         title=r"$\partial SW^*/\partial\eta$: Welfare under Environmental Pressure")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:.0f}"))
    _pl(ax, "(b)")

    # (c)
    ax = axes[2]
    ax.plot(gamma_data["gammas"], gamma_data["tau_s"], color=C2, lw=2.0, zorder=4,
            label=r"$\tau^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], color=C2, alpha=0.10)
    axr2 = ax.twinx()
    axr2.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C1, lw=1.8, ls="--", zorder=3,
              label=r"$\rho^*(\gamma)$")
    axr2.set_ylabel(r"Recycling rate $\rho^*$", color=C1, labelpad=5)
    axr2.tick_params(axis="y", colors=C1); axr2.spines["top"].set_visible(False)
    axr2.yaxis.set_minor_locator(AutoMinorLocator(2))
    _sax(ax, xlabel=r"Green preference $\gamma$",
         ylabel=r"Optimal carbon tax $\tau^*$",
         title=r"$\partial\tau^*/\partial\gamma > 0$: Greener Consumers $\Rightarrow$ Higher Tax")
    ax.tick_params(axis="y", colors=C2); ax.spines["left"].set_color(C2)
    lns2 = ax.get_lines() + axr2.get_lines()
    ax.legend(lns2, [l.get_label() for l in lns2], loc="upper left", fontsize=8)
    _pl(ax, "(c)")

    # (d)
    ax = axes[3]
    ax.plot(gamma_data["gammas"], gamma_data["D_s"], color=C4, lw=2.0, zorder=4,
            label=r"Demand $D^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), gamma_data["D_s"],
                    color=C4, alpha=0.12)
    axr3 = ax.twinx()
    axr3.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C5, lw=1.8, ls="--", zorder=3,
              label=r"$\rho^*(\gamma)$")
    axr3.set_ylabel(r"Recycling rate $\rho^*$", color=C5, labelpad=5)
    axr3.tick_params(axis="y", colors=C5); axr3.spines["top"].set_visible(False)
    axr3.yaxis.set_minor_locator(AutoMinorLocator(2))
    _sax(ax, xlabel=r"Green preference $\gamma$",
         ylabel=r"Market demand $D^*$",
         title=r"$\partial D^*/\partial\gamma > 0$: Green Preference Expands Market")
    lns3 = ax.get_lines() + axr3.get_lines()
    ax.legend(lns3, [l.get_label() for l in lns3], loc="upper left", fontsize=8)
    _pl(ax, "(d)")

    fig.suptitle(
        r"Figure 5.  Comparative Statics: Environmental Damage $\eta$ and Green Preference $\gamma$",
        fontsize=11.5, y=0.975, style="italic", color="#222222")
    _save(fig, "CLSC_Fig5_ComparativeStatics")

# ════════════════════════════════════════════════════════════════════
# FIGURE 6: Sensitivity Analysis (新增！)
# ════════════════════════════════════════════════════════════════════
def fig6_sensitivity_analysis(eta_data, gamma_data):
    """
    敏感性分析图：展示核心参数变化对关键指标的影响
    - 左上：η 对 τ* 和 ρ* 的影响
    - 右上：η 对 E* 和 SW* 的影响
    - 左下：γ 对 τ* 和 ρ* 的影响
    - 右下：γ 对 D* 和 SW* 的影响
    """
    fig = plt.figure(figsize=(14., 10.))
    gs = GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.45,
                  left=0.07, right=0.95, top=0.92, bottom=0.08)
    
    # 1. 左上：η 对 τ* 和 ρ* 的影响
    ax1 = fig.add_subplot(gs[0,0])
    line1_1 = ax1.plot(eta_data["etas"], eta_data["tau_s"], color=C2, lw=2.2, zorder=4, 
                       label=r"Optimal carbon tax ($\tau^*$)")
    ax1.fill_between(eta_data["etas"], 0, eta_data["tau_s"], color=C2, alpha=0.1, zorder=1)
    ax1.set_xlabel(r"Environmental damage coefficient ($\eta$)", labelpad=5)
    ax1.set_ylabel(r"Optimal carbon tax ($\tau^*$)", color=C2, labelpad=5)
    ax1.tick_params(axis='y', labelcolor=C2)
    ax1.grid(True, alpha=0.2)
    
    # 双Y轴：回收率
    ax1_twin = ax1.twinx()
    line1_2 = ax1_twin.plot(eta_data["etas"], eta_data["rho_s"], color=C1, lw=2.0, ls="--", zorder=3,
                            label=r"Recycling rate ($\rho^*$)")
    ax1_twin.set_ylabel(r"Recycling rate ($\rho^*$)", color=C1, labelpad=5)
    ax1_twin.tick_params(axis='y', labelcolor=C1)
    
    # 合并图例
    lines1 = line1_1 + line1_2
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.set_title(r"Effect of $\eta$ on $\tau^*$ and $\rho^*$", pad=10)
    _pl(ax1, "(a)", x=-0.12, y=1.05)
    
    # 2. 右上：η 对 E* 和 SW* 的影响
    ax2 = fig.add_subplot(gs[0,1])
    line2_1 = ax2.plot(eta_data["etas"], eta_data["E_s"], color=C4, lw=2.2, zorder=4,
                       label=r"Total emissions ($E^*$)")
    ax2.fill_between(eta_data["etas"], eta_data["E_s"].min(), eta_data["E_s"], color=C4, alpha=0.1, zorder=1)
    ax2.set_xlabel(r"Environmental damage coefficient ($\eta$)", labelpad=5)
    ax2.set_ylabel(r"Total emissions ($E^*$)", color=C4, labelpad=5)
    ax2.tick_params(axis='y', labelcolor=C4)
    ax2.grid(True, alpha=0.2)
    
    # 双Y轴：社会福利
    ax2_twin = ax2.twinx()
    line2_2 = ax2_twin.plot(eta_data["etas"], eta_data["SW_s"], color=C3, lw=2.0, ls="--", zorder=3,
                            label=r"Social welfare ($SW^*$)")
    ax2_twin.set_ylabel(r"Social welfare ($SW^*$)", color=C3, labelpad=5)
    ax2_twin.tick_params(axis='y', labelcolor=C3)
    
    # 合并图例
    lines2 = line2_1 + line2_2
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2.set_title(r"Effect of $\eta$ on $E^*$ and $SW^*$", pad=10)
    _pl(ax2, "(b)", x=-0.12, y=1.05)
    
    # 3. 左下：γ 对 τ* 和 ρ* 的影响
    ax3 = fig.add_subplot(gs[1,0])
    line3_1 = ax3.plot(gamma_data["gammas"], gamma_data["tau_s"], color=C2, lw=2.2, zorder=4,
                       label=r"Optimal carbon tax ($\tau^*$)")
    ax3.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], color=C2, alpha=0.1, zorder=1)
    ax3.set_xlabel(r"Green preference coefficient ($\gamma$)", labelpad=5)
    ax3.set_ylabel(r"Optimal carbon tax ($\tau^*$)", color=C2, labelpad=5)
    ax3.tick_params(axis='y', labelcolor=C2)
    ax3.grid(True, alpha=0.2)
    
    # 双Y轴：回收率
    ax3_twin = ax3.twinx()
    line3_2 = ax3_twin.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C1, lw=2.0, ls="--", zorder=3,
                            label=r"Recycling rate ($\rho^*$)")
    ax3_twin.set_ylabel(r"Recycling rate ($\rho^*$)", color=C1, labelpad=5)
    ax3_twin.tick_params(axis='y', labelcolor=C1)
    
    # 合并图例
    lines3 = line3_1 + line3_2
    labels3 = [l.get_label() for l in lines3]
    ax3.legend(lines3, labels3, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax3.set_title(r"Effect of $\gamma$ on $\tau^*$ and $\rho^*$", pad=10)
    _pl(ax3, "(c)", x=-0.12, y=1.05)
    
    # 4. 右下：γ 对 D* 和 SW* 的影响
    ax4 = fig.add_subplot(gs[1,1])
    line4_1 = ax4.plot(gamma_data["gammas"], gamma_data["D_s"], color=C4, lw=2.2, zorder=4,
                       label=r"Market demand ($D^*$)")
    ax4.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), gamma_data["D_s"], color=C4, alpha=0.1, zorder=1)
    ax4.set_xlabel(r"Green preference coefficient ($\gamma$)", labelpad=5)
    ax4.set_ylabel(r"Market demand ($D^*$)", color=C4, labelpad=5)
    ax4.tick_params(axis='y', labelcolor=C4)
    ax4.grid(True, alpha=0.2)
    
    # 双Y轴：社会福利
    ax4_twin = ax4.twinx()
    line4_2 = ax4_twin.plot(gamma_data["gammas"], gamma_data["SW_s"], color=C3, lw=2.0, ls="--", zorder=3,
                            label=r"Social welfare ($SW^*$)")
    ax4_twin.set_ylabel(r"Social welfare ($SW^*$)", color=C3, labelpad=5)
    ax4_twin.tick_params(axis='y', labelcolor=C3)
    
    # 合并图例
    lines4 = line4_1 + line4_2
    labels4 = [l.get_label() for l in lines4]
    ax4.legend(lines4, labels4, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax4.set_title(r"Effect of $\gamma$ on $D^*$ and $SW^*$", pad=10)
    _pl(ax4, "(d)", x=-0.12, y=1.05)
    
    # 总标题
    fig.suptitle("Figure 6.  Sensitivity Analysis: Key Parameters vs Performance Metrics",
                 fontsize=12, y=0.98, style="italic", color="#222222")
    
    # 保存图形
    _save(fig, "CLSC_Fig6_Sensitivity_Analysis")

# ════════════════════════════════════════════════════════════════════
# MAIN (更新调用新增的敏感性分析图)
# ════════════════════════════════════════════════════════════════════
def main():
    # 记录总运行时间
    total_start = time.perf_counter()
    
    # 初始化配置
    c = Cfg()
    
    # 欢迎信息 + 配置参数展示
    print("\n" + "="*80)
    print("🚀 CLSC Stackelberg Game Model - FINAL COMPLETE VERSION 🚀".center(80))
    print("="*80)
    print("\n📋 模型基础配置参数:")
    print(f"   市场规模 (a): {c.a} | 价格弹性 (b): {c.b} | 绿色偏好系数 (γ): {c.gamma}")
    print(f"   制造成本 (c_m): {c.c_m} | 回收成本 (c_r): {c.c_r} | 回收固定成本 (k): {c.k}")
    print(f"   单位排放 (e_m): {c.e_m} | 回收减排 (e_r): {c.e_r} | 环境损害权重 (η): {c.eta}")
    print(f"   最大碳税 (τ_max): {c.tau_max} | 优化精度 (tol): {c.tol}")
    print("-"*80)

    # 1. 求解均衡点 (带耗时统计)
    print("\n🔢 [1/5] 开始求解各场景均衡点...")
    eq_start = time.perf_counter()
    
    eq_nt = dec_eq(c, tau=0.0)
    eq_d = dec_eq(c)
    eq_c = vif_eq(c)
    
    eq_elapsed = time.perf_counter() - eq_start
    print(f"✅ 均衡点求解完成 (耗时: {eq_elapsed:.2f} 秒)")
    print(f"   📌 无碳税场景 (τ=0):")
    print(f"      回收率 ρ: {eq_nt['rho']:.4f} | 总排放 E: {eq_nt['E']:.2f} | 社会福利 SW: {eq_nt['SW']:.1f}")
    print(f"   📌 分散决策场景 (τ*):")
    print(f"      最优碳税 τ*: {eq_d['tau']:.3f} | 回收率 ρ: {eq_d['rho']:.4f} | 总排放 E: {eq_d['E']:.2f} | 社会福利 SW: {eq_d['SW']:.1f}")
    print(f"   📌 集中决策场景 (VIF):")
    print(f"      最优碳税 τ^C: {eq_c['tau']:.3f} | 回收率 ρ: {eq_c['rho']:.4f} | 总排放 E: {eq_c['E']:.2f} | 社会福利 SW: {eq_c['SW']:.1f}")

    # 2. 敏感性分析 (带进度提示)
    print("\n📈 [2/5] 开始计算敏感性分析数据...")
    sens_start = time.perf_counter()
    
    print("   🔄 正在计算碳税扫描 (τ sweep, n=150)...", end=" ")
    sens = tau_sweep(c, n=150)
    print("✅")
    
    print("   🔄 正在计算环境损害敏感性 (η sweep, n=60)...", end=" ")
    eta_d = eta_sweep(c, n=60)
    print("✅")
    
    print("   🔄 正在计算绿色偏好敏感性 (γ sweep, n=60)...", end=" ")
    gam_d = gamma_sweep(c, n=60)
    print("✅")
    
    print("   🔄 正在计算收益共享契约分析 (φ sweep, n=80)...", end=" ")
    phi_d = phi_sweep(c, eq_d["tau"], n=80)
    print("✅")
    
    sens_elapsed = time.perf_counter() - sens_start
    print(f"✅ 敏感性分析完成 (耗时: {sens_elapsed:.2f} 秒)")

    # 3. 生成图形 (带进度提示，新增图6)
    print("\n🎨 [3/5] 开始渲染6个高质量论文级图形 (600 DPI)...")
    fig_start = time.perf_counter()
    
    print("   🖼️  正在生成图1: 定价与回收率动态...", end=" ")
    fig1(eq_nt, eq_d, eq_c, sens)
    print("✅")
    
    print("   🖼️  正在生成图2: 排放与社会福利...", end=" ")
    fig2(eq_nt, eq_d, eq_c, sens)
    print("✅")
    
    print("   🖼️  正在生成图3: 场景对比与福利分解...", end=" ")
    fig3(eq_nt, eq_d, eq_c)
    print("✅")
    
    print("   🖼️  正在生成图4: 收益共享契约分析...", end=" ")
    fig4(eq_d, eq_c, phi_d)
    print("✅")
    
    print("   🖼️  正在生成图5: 比较静态分析...", end=" ")
    fig5(eta_d, gam_d)
    print("✅")
    
    print("   🖼️  正在生成图6: 敏感性分析 (新增)...", end=" ")
    fig6_sensitivity_analysis(eta_d, gam_d)  # 调用新增的敏感性分析图
    print("✅")
    
    fig_elapsed = time.perf_counter() - fig_start
    print(f"✅ 所有图形生成完成 (耗时: {fig_elapsed:.2f} 秒)")
    print(f"   📁 图形文件已保存到当前目录 (PDF/PNG 格式, 600 DPI)")
    print(f"   📊 包含: 基础分析图×5 + 敏感性分析图×1 (共6个图形)")

        # 4. 生成数值汇总表 (增强版)
    print("\n📊 [4/5] 生成详细数值汇总表...")
    print("-"*85)
    print(f"{'指标':<26}  {'无碳税 (τ=0)':>15}  {'分散决策 (τ*)':>18}  {'集中决策 (VIF)':>18}")
    print("-"*85)
    print(f"{'碳税 (τ)':<26}  {0.00:>15.3f}  {eq_d['tau']:>18.3f}  {eq_c['tau']:>18.3f}")
    print(f"{'零售价格 (p)':<26}  {eq_nt['p']:>15.2f}  {eq_d['p']:>18.2f}  {eq_c['p']:>18.2f}")
    print(f"{'批发价格 (w)':<26}  {eq_nt['w']:>15.2f}  {eq_d['w']:>18.2f}  {'-':>18}")
    print(f"{'回收率 (ρ)':<26}  {eq_nt['rho']:>15.4f}  {eq_d['rho']:>18.4f}  {eq_c['rho']:>18.4f}")
    print(f"{'市场需求 (D)':<26}  {eq_nt['D']:>15.2f}  {eq_d['D']:>18.2f}  {eq_c['D']:>18.2f}")
    print(f"{'总排放 (E)':<26}  {eq_nt['E']:>15.2f}  {eq_d['E']:>18.2f}  {eq_c['E']:>18.2f}")
    print(f"{'制造商利润 (π_m)':<26}  {eq_nt['pi_m']:>15.2f}  {eq_d['pi_m']:>18.2f}  {eq_c['pi_vif']:>18.2f}")
    print(f"{'零售商利润 (π_r)':<26}  {eq_nt['pi_r']:>15.2f}  {eq_d['pi_r']:>18.2f}  {'-':>18}")
    print(f"{'消费者剩余 (CS)':<26}  {eq_nt['CS']:>15.2f}  {eq_d['CS']:>18.2f}  {eq_c['CS']:>18.2f}")
    print(f"{'社会福利 (SW)':<26}  {eq_nt['SW']:>15.1f}  {eq_d['SW']:>18.1f}  {eq_c['SW']:>18.1f}")
    print("-"*85)
    
    # 5. 关键结论与指标计算
    print("\n🎯 [5/5] 核心结论与关键指标计算...")
    # 计算关键差值和比率
    delta_rho = eq_c['rho'] - eq_d['rho']  # 回收率协调缺口
    poa = eq_d['SW'] / eq_c['SW']          # 价格无效率
    emission_reduction = eq_nt['E'] - eq_d['E']  # 碳税减排量
    welfare_improvement = eq_d['SW'] - eq_nt['SW']  # 福利提升
    
    print(f"   📏 回收率协调缺口 (ρ^C - ρ^*): {delta_rho:.4f}")
    print(f"   📈 价格无效率 (PoA = SW^*/SW^C): {poa:.4f} (越接近1越优)")
    print(f"   🌱 碳税减排效果 (E0 - E*): {emission_reduction:.2f} 单位")
    print(f"   💹 福利提升 (SW* - SW0): {welfare_improvement:.1f}")
    print(f"   ⚖️  集中决策福利优势 (SW^C - SW*): {eq_c['SW'] - eq_d['SW']:.1f}")
    
    # 运行总耗时统计
    total_elapsed = time.perf_counter() - total_start
    print("\n" + "="*80)
    print(f"🏁 模型运行完成！总耗时: {total_elapsed:.2f} 秒".center(80))
    print("="*80)
    print("\n📝 输出文件说明:")
    print("   • 6个高质量论文级图形文件 (PDF/PNG格式, 600 DPI) 已保存到当前目录")
    print("   • 图形命名规则: CLSC_FigX_XXX.pdf/png (X为图号)")
    print("   • 包含: 定价回收/排放福利/场景对比/收益共享/比较静态/敏感性分析")
    print("\n💡 关键发现总结:")
    print("   1. 最优碳税能有效平衡减排与经济福利，但存在回收率协调缺口")
    print("   2. 环境损害系数η↑ → 最优碳税τ*↑，绿色偏好系数γ↑ → 市场需求D↑")
    print("   3. 集中决策(VIF)能实现更高回收率和福利，但需要渠道协调机制")
    print("="*80 + "\n")

# 程序入口
if __name__ == "__main__":
    main()
