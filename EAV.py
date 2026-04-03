"""
CLSC Stackelberg Game Model — CORRECTED VERSION
=====================================================
闭环供应链碳税政策 Stackelberg 博弈模型（修正版）

修正说明:
    • 修正回收率一阶条件（原论文忽略了 γ 项）
    • 使用数值优化直接求解最优ρ，避免解析近似误差
    • 保持与论文其他部分一致

作者：飞翔的蚂蚱 · 严谨专业版
版本：Corrected v3.0
日期：2026-04-03
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
import warnings
import logging
import time
import sys
from datetime import datetime

# ── 抑制警告 ────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# ── 自动检测最佳 serif 字体 ────────────────────────────────────
def _best_serif() -> str:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Latin Modern Roman", "Lora", "Caladea", "Liberation Serif", 
                 "DejaVu Serif", "FreeSerif", "serif"]:
        if font in available:
            return font
    return "serif"

_SERIF = _best_serif()

# ── 期刊级图形样式设置 ────────────────────────────────────────
plt.rcParams.update({
    "font.family": _SERIF,
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlepad": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "patch.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.handlelength": 2.2,
    "legend.handleheight": 0.9,
    "legend.borderpad": 0.6,
    "legend.labelspacing": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── 配色方案 ────────────────────────────────────────────────────
C1 = "#1B3A6B"
C2 = "#C94B2D"
C3 = "#2E7D5E"
C4 = "#B07D2A"
C5 = "#5C4B8A"
CG = "#7A7A7A"

# ── 辅助绘图函数 ──────────────────────────────────────────────
def _sax(ax, xlabel="", ylabel="", title="", grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6, fontweight="normal")
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=7, fontweight="normal")
    if title:
        ax.set_title(title, pad=10, fontsize=10.5)
    if grid:
        ax.grid(True, which="major", zorder=0, alpha=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.tick_params(which="major", length=4.5, width=0.9)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.9)
        ax.spines[sp].set_color("#333333")
    return ax

def _pl(ax, label, x=-0.15, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, 
            fontsize=13, fontweight="bold", 
            va="top", ha="left", color="#1a1a1a")

def _vl(ax, x, c=C2, lw=1.4, ls=":", alpha=0.8):
    ax.axvline(x, color=c, lw=lw, ls=ls, alpha=alpha, zorder=3)

def _hl(ax, y, c=CG, lw=1.1, ls="--", alpha=0.7):
    ax.axhline(y, color=c, lw=lw, ls=ls, alpha=alpha, zorder=3)

def _dot(ax, x, y, c=C2, s=60, z=10):
    ax.scatter([x], [y], color=c, s=s, zorder=z, 
               edgecolors="white", linewidths=1.2)

def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(f"./{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ {name}.{{pdf,png}}")

# ── 模型配置 ──────────────────────────────────────────────────
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
    def dc(self):
        return self.c_m - self.c_r
    
    @property
    def de(self):
        return round(self.e_m - self.e_r, 10)

# ── 核心函数 ──────────────────────────────────────────────────
def Df(c, p, rho):
    return max(c.a - c.b * p + c.gamma * rho, 0.0)

def Ef(c, d, rho):
    return max((c.e_m - (c.e_m - c.e_r) * rho) * d, 0.0)

def CSf(c, p, rho):
    d = Df(c, p, rho)
    return d**2 / (2 * c.b)

def prf(c, w, rho):
    return (c.a + c.gamma * rho + c.b * w) / (2 * c.b)

def pim(c, w, rho, tau):
    p = prf(c, w, rho)
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    return (w - c.c_m) * d + c.dc * rho * d - tau * em - c.k * rho**2

def pivif(c, p, rho, tau):
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    return (p - c.c_m) * d + c.dc * rho * d - c.k * rho**2 - tau * em

def SWf(c, cs, pi_m, pi_r, tau, em):
    tax_revenue = tau * em
    environmental_damage = c.eta * em**2
    return cs + pi_m + pi_r + tax_revenue - environmental_damage

# ═══════════════════════════════════════════════════════════════
# 【关键修正】回收率反应函数
# ═══════════════════════════════════════════════════════════════

def rs_original(c, d, tau):
    """原论文公式（有缺陷）"""
    Delta_c = c.dc + tau * c.de
    return float(np.clip(Delta_c * d / (2 * c.k), 0, 1))

def rs_optimal(c, w, tau):
    """修正版 - 数值优化"""
    def neg_pi_rho(rho):
        p = prf(c, w, rho)
        d = Df(c, p, rho)
        em = Ef(c, d, rho)
        profit = (w - c.c_m) * d + c.dc * rho * d - tau * em - c.k * rho**2
        return -profit
    
    res = minimize_scalar(neg_pi_rho, bounds=(0, 1), method="bounded", 
                         options={"xatol": 1e-12})
    return float(np.clip(res.x, 0, 1))

# ═══════════════════════════════════════════════════════════════
# 均衡求解器
# ═══════════════════════════════════════════════════════════════

def dec_solve(c, tau, use_corrected=True):
    def solve_rho_for_w(w, rho_init=0.15):
        if use_corrected:
            return rs_optimal(c, w, tau)
        else:
            rho = rho_init
            for _ in range(100):
                p = prf(c, w, rho)
                d = Df(c, p, rho)
                rho_new = rs_original(c, d, tau)
                if abs(rho_new - rho) < 1e-10:
                    break
                rho = 0.7 * rho + 0.3 * rho_new
            return rho
    
    def neg_pi_m(w):
        rho = solve_rho_for_w(w)
        return -pim(c, w, rho, tau)
    
    res = minimize_scalar(neg_pi_m, 
                         bounds=(c.c_m + 0.01, c.a / c.b - 0.01), 
                         method="bounded", 
                         options={"xatol": c.tol})
    w = float(res.x)
    rho = solve_rho_for_w(w, rho_init=0.15)
    p = prf(c, w, rho)
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    
    pm = pim(c, w, rho, tau)
    pr = (p - w) * d
    cs = CSf(c, p, rho)
    sw = SWf(c, cs, pm, pr, tau, em)
    
    return dict(tau=tau, w=w, p=p, rho=rho, D=d, E=em, 
                pi_m=pm, pi_r=pr, CS=cs, 
                env_dam=c.eta * em**2, tax_rev=tau * em, SW=sw)

def dec_eq(c, tau=None, use_corrected=True):
    if tau is not None:
        return dec_solve(c, tau, use_corrected=use_corrected)
    
    res = minimize_scalar(lambda t: -dec_solve(c, t, use_corrected=use_corrected)["SW"], 
                         bounds=(0, c.tau_max), 
                         method="bounded", 
                         options={"xatol": c.tol})
    return dec_solve(c, float(res.x), use_corrected=use_corrected)

def vif_br(c, tau):
    def neg_pi_vif(x):
        return -pivif(c, x[0], x[1], tau)
    
    res = minimize(neg_pi_vif, 
                  x0=[350, 0.3], 
                  bounds=[(c.c_m + 0.01, c.a / c.b - 0.01), (0, 1)], 
                  method="L-BFGS-B", 
                  options={"ftol": 1e-15, "gtol": 1e-12})
    return float(res.x[0]), float(np.clip(res.x[1], 0, 1))

def vif_sw_val(c, tau):
    p, rho = vif_br(c, tau)
    profit_vif = pivif(c, p, rho, tau)
    if profit_vif < 0:
        return -1e10
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    cs = CSf(c, p, rho)
    return SWf(c, cs, profit_vif, 0, tau, em)

def vif_eq(c):
    tau_grid = np.linspace(0, c.tau_max, 200)
    sw_values = [vif_sw_val(c, t) for t in tau_grid]
    best_idx = np.argmax(sw_values)
    tau_init = tau_grid[best_idx]
    
    res = minimize_scalar(lambda t: -vif_sw_val(c, t), 
                         bounds=(max(0, tau_init - 10), 
                                min(c.tau_max, tau_init + 10)), 
                         method="bounded", 
                         options={"xatol": c.tol})
    tau_opt = float(res.x)
    p_opt, rho_opt = vif_br(c, tau_opt)
    d = Df(c, p_opt, rho_opt)
    em = Ef(c, d, rho_opt)
    profit_vif = pivif(c, p_opt, rho_opt, tau_opt)
    cs = CSf(c, p_opt, rho_opt)
    sw = SWf(c, cs, profit_vif, 0, tau_opt, em)
    
    return dict(tau=tau_opt, p=p_opt, rho=rho_opt, D=d, E=em, 
                pi_vif=profit_vif, CS=cs, 
                env_dam=c.eta * em**2, tax_rev=tau_opt * em, SW=sw)

# ── 敏感性分析 ────────────────────────────────────────────────
def tau_sweep(c, n=150, use_corrected=True):
    taus = np.linspace(0, c.tau_max, n)
    keys = ["w", "p", "rho", "D", "E", "pi_m", "pi_r", "CS", "SW", 
            "env_dam", "tax_rev"]
    out = {k: np.empty(n) for k in keys}
    out["taus"] = taus
    for i, t in enumerate(taus):
        eq = dec_solve(c, t, use_corrected=use_corrected)
        for k in keys:
            out[k][i] = eq[k]
    return out

def eta_sweep(c, n=60, use_corrected=True):
    etas = np.linspace(0.5, 8.0, n)
    ts, Es, SWs, rhos = [], [], [], []
    orig = c.eta
    for ev in etas:
        c.eta = ev
        eq = dec_eq(c, use_corrected=use_corrected)
        ts.append(eq["tau"])
        Es.append(eq["E"])
        SWs.append(eq["SW"])
        rhos.append(eq["rho"])
    c.eta = orig
    return dict(etas=etas, tau_s=np.array(ts), E_s=np.array(Es), 
                SW_s=np.array(SWs), rho_s=np.array(rhos))

def gamma_sweep(c, n=60, use_corrected=True):
    gammas = np.linspace(2, 28, n)
    ts, rhos, Ds, SWs = [], [], [], []
    orig = c.gamma
    for gv in gammas:
        c.gamma = gv
        eq = dec_eq(c, use_corrected=use_corrected)
        ts.append(eq["tau"])
        rhos.append(eq["rho"])
        Ds.append(eq["D"])
        SWs.append(eq["SW"])
    c.gamma = orig
    return dict(gammas=gammas, tau_s=np.array(ts), rho_s=np.array(rhos), 
                D_s=np.array(Ds), SW_s=np.array(SWs))

def phi_sweep(c, tau, n=80, use_corrected=True):
    phis = np.linspace(0.01, 0.99, n)
    pms, prs, sws = [], [], []
    for phi in phis:
        w = c.c_m
        rho = 0.3
        for _ in range(50):
            p = (c.a + c.gamma * rho) / (2 * c.b) + w / (2 * (1 - phi))
            d = Df(c, p, rho)
            if use_corrected:
                rho_new = rs_optimal(c, w, tau)
            else:
                rho_new = rs_original(c, d, tau)
            if abs(rho_new - rho) < 1e-8:
                break
            rho = 0.7 * rho + 0.3 * rho_new
        d = Df(c, p, rho)
        em = Ef(c, d, rho)
        pm = phi * p * d + c.dc * rho * d - c.k * rho**2 - tau * em
        pr = (1 - phi) * p * d - w * d
        cs = CSf(c, p, rho)
        sv = SWf(c, cs, pm, pr, tau, em)
        pms.append(pm)
        prs.append(pr)
        sws.append(sv)
    return dict(phis=phis, pi_m=np.array(pms), pi_r=np.array(prs), sw=np.array(sws))

# ═══════════════════════════════════════════════════════════════
# 图形函数（简化版，节省空间）
# ═══════════════════════════════════════════════════════════════

def fig1(eq_nt, eq_d, eq_c, sens):
    fig = plt.figure(figsize=(14.0, 5.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.96, top=0.88, bottom=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    taus = sens["taus"]
    
    ax1.fill_between(taus, sens["w"], sens["p"], color=C1, alpha=0.10, zorder=1, label="Retailer margin")
    ax1.plot(taus, sens["p"], color=C1, lw=2.2, zorder=4, label=r"Retail price $p^*(\tau)$")
    ax1.plot(taus, sens["w"], color=C1, lw=1.8, ls="--", zorder=4, label=r"Wholesale price $w^*(\tau)$")
    _hl(ax1, eq_c["p"], c=C3, lw=1.3, ls=(0, (5, 3)), alpha=0.8)
    ax1.text(taus[-1] * 0.97, eq_c["p"] + 2.5, f"$p^C={eq_c['p']:.1f}$ (VIF)", ha="right", va="bottom", fontsize=9, color=C3, style="italic")
    _vl(ax1, eq_d["tau"], c=C2, lw=1.5)
    _dot(ax1, eq_d["tau"], eq_d["p"], c=C2)
    _dot(ax1, eq_d["tau"], eq_d["w"], c=C2)
    margin = eq_d["p"] - eq_d["w"]
    ax1.annotate("", xy=(eq_d["tau"] + 3, eq_d["p"]), xytext=(eq_d["tau"] + 3, eq_d["w"]), arrowprops=dict(arrowstyle="<->", color=C2, lw=1.3))
    ax1.text(eq_d["tau"] + 6, (eq_d["p"] + eq_d["w"]) / 2, f"$\\Delta={margin:.1f}$", fontsize=9, color=C2, va="center")
    ax1.text(eq_d["tau"], sens["p"].max() - 5, f"$\\tau^*={eq_d['tau']:.1f}$", fontsize=9.5, color=C2, fontweight="bold", ha="center", va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C2, alpha=0.9, lw=0.7))
    _sax(ax1, xlabel=r"Carbon tax $\tau$", ylabel="Price", title="Pricing Dynamics")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax1.set_xlim(0, taus[-1])
    ax1.set_ylim(280, 380)
    _pl(ax1, "(a)")
    
    ax2.axvspan(0, eq_d["tau"], color=C4, alpha=0.06, zorder=0)
    ax2.plot(taus, sens["rho"], color=C1, lw=2.2, zorder=4, label=r"$\rho^*(\tau)$")
    _hl(ax2, eq_c["rho"], c=C3, lw=1.5, ls=(0, (5, 3)), alpha=0.9)
    ax2.text(taus[-1] * 0.97, eq_c["rho"] + 0.008, f"$\\rho^C={eq_c['rho']:.3f}$ (VIF)", ha="right", va="bottom", fontsize=9, color=C3, style="italic")
    _vl(ax2, eq_d["tau"], c=C2, lw=1.5)
    gap = eq_c["rho"] - eq_d["rho"]
    ax2.annotate("", xy=(eq_d["tau"] + 3, eq_c["rho"]), xytext=(eq_d["tau"] + 3, eq_d["rho"]), arrowprops=dict(arrowstyle="<->", color=C2, lw=1.3))
    ax2.text(eq_d["tau"] + 6, (eq_d["rho"] + eq_c["rho"]) / 2, f"$\\Delta\\rho={gap:.3f}$", fontsize=9, color=C2, va="center", fontweight="bold")
    _dot(ax2, eq_d["tau"], eq_d["rho"], c=C2)
    _sax(ax2, xlabel=r"Carbon tax $\tau$", ylabel=r"Recycling rate $\rho$", title="Recycling Rate")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax2.set_xlim(0, taus[-1])
    ax2.set_ylim(0, 0.55)
    _pl(ax2, "(b)")
    
    fig.suptitle("Figure 1. Supply Chain Equilibrium", fontsize=12, y=0.96, style="italic")
    _save(fig, "CLSC_Fig1_Pricing_Recycling")

def fig2(eq_nt, eq_d, eq_c, sens):
    fig = plt.figure(figsize=(14.0, 5.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.96, top=0.88, bottom=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    taus = sens["taus"]
    
    ax1.plot(taus, sens["E"], color=C4, lw=2.2, zorder=4, label=r"$E^*(\tau)$")
    _hl(ax1, eq_nt["E"], c=CG, lw=1.3)
    ax1.text(4, eq_nt["E"] + 0.6, f"$E_0={eq_nt['E']:.1f}$", fontsize=9, color=CG)
    _vl(ax1, eq_d["tau"], c=C2, lw=1.5)
    _dot(ax1, eq_d["tau"], eq_d["E"], c=C2)
    _sax(ax1, xlabel=r"Carbon tax $\tau$", ylabel=r"Emissions $E$", title="Environmental Impact")
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax1.set_xlim(0, taus[-1])
    _pl(ax1, "(a)")
    
    ax2.plot(taus, sens["SW"], color=C1, lw=2.2, zorder=4, label=r"$SW(\tau)$")
    _hl(ax2, eq_nt["SW"], c=CG, lw=1.3)
    _hl(ax2, eq_c["SW"], c=C3, lw=1.3, ls=(0, (5, 3)), alpha=0.9)
    idx = int(np.argmax(sens["SW"]))
    _vl(ax2, sens["taus"][idx], c=C2, lw=1.5)
    _dot(ax2, sens["taus"][idx], sens["SW"][idx], c=C2, s=70)
    _sax(ax2, xlabel=r"Carbon tax $\tau$", ylabel=r"Social welfare $SW$", title="Social Welfare")
    ax2.set_xlim(0, taus[-1])
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)
    _pl(ax2, "(b)")
    
    fig.suptitle("Figure 2. Environmental Impact and Social Welfare", fontsize=12, y=0.96, style="italic")
    _save(fig, "CLSC_Fig2_Emission_Welfare")

def fig3(eq_nt, eq_d, eq_c):
    fig = plt.figure(figsize=(14.0, 5.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.40, left=0.07, right=0.96, top=0.86, bottom=0.14)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    scenarios = ["No-Tax", "Decentralised", "VIF"]
    colors3 = [CG, C1, C3]
    rhos = [eq_nt["rho"], eq_d["rho"], eq_c["rho"]]
    sws = [eq_nt["SW"], eq_d["SW"], eq_c["SW"]]
    x = np.arange(3)
    
    bars = ax1.bar(x, rhos, 0.45, color=colors3, alpha=0.85, edgecolor="white", lw=0.9, zorder=3)
    for i, (bar, v) in enumerate(zip(bars, rhos)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=colors3[i])
    ax1.set_ylabel(r"Recycling rate $\rho$", labelpad=7)
    ax1.set_ylim(0, max(rhos) * 1.45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1r = ax1.twinx()
    ax1r.plot(x, sws, "D--", color=C4, lw=1.8, ms=8, markerfacecolor=C4, markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax1r.set_ylabel(r"Social welfare $SW$", color=C4, labelpad=7)
    ax1r.tick_params(axis="y", colors=C4)
    poa = eq_d["SW"] / eq_c["SW"]
    gap = eq_c["rho"] - eq_d["rho"]
    ax1.set_title(f"$\\Delta\\rho={gap:.3f}$, PoA$={poa:.4f}$", fontsize=10, pad=8)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    _pl(ax1, "(a)", x=-0.12)
    
    dCS = eq_d["CS"] - eq_nt["CS"]
    dPim = eq_d["pi_m"] - eq_nt["pi_m"]
    dPir = eq_d["pi_r"] - eq_nt["pi_r"]
    dTax = eq_d["tax_rev"]
    dEnv = eq_nt["env_dam"] - eq_d["env_dam"]
    dSW = eq_d["SW"] - eq_nt["SW"]
    vals = [dCS, dPim, dPir, dTax, dEnv, dSW]
    labs = [r"$\Delta CS$", r"$\Delta\pi_m$", r"$\Delta\pi_r$", r"$\tau E$", r"$-\Delta(\eta E^2)$", r"Net $\Delta SW$"]
    bar_bottoms = []
    run = 0
    for v in vals[:-1]:
        bar_bottoms.append(run if v >= 0 else run + v)
        run += v
    bar_bottoms.append(0)
    bar_c = [C3 if v >= 0 else C2 for v in vals]
    bar_c[-1] = C3
    ax2.bar(np.arange(6), np.abs(vals), 0.58, bottom=bar_bottoms, color=bar_c, alpha=0.82, edgecolor="white", lw=0.8, zorder=3)
    ax2.axhline(0, color="#333333", lw=1.0, zorder=4)
    ax2.set_xticks(np.arange(6))
    ax2.set_xticklabels(labs, fontsize=10)
    ax2.set_ylabel(r"Welfare change $(\Delta)$", labelpad=7)
    ax2.set_title(r"Welfare Decomposition", fontsize=10, pad=8)
    ax2.grid(axis="y", alpha=0.25, zorder=0)
    _pl(ax2, "(b)", x=-0.12)
    
    fig.suptitle("Figure 3. Scenario Comparison and Welfare Decomposition", fontsize=12, y=0.95, style="italic")
    _save(fig, "CLSC_Fig3_Benchmark_Decomposition")

def fig4(eq_d, eq_c, phi_data):
    fig = plt.figure(figsize=(14.0, 5.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.96, top=0.88, bottom=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    phis = phi_data["phis"]
    pms = phi_data["pi_m"]
    prs = phi_data["pi_r"]
    sws = phi_data["sw"]
    
    ax1.plot(phis, pms, color=C1, lw=2.2, zorder=4, label=r"$\pi_m^{RS}(\phi)$")
    ax1.plot(phis, prs, color=C4, lw=2.2, zorder=4, label=r"$\pi_r^{RS}(\phi)$")
    ax1.axhline(0, color="#555555", lw=0.9, zorder=5)
    _sax(ax1, xlabel=r"Revenue-sharing $\phi$", ylabel="Profit", title="Profit Distribution")
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.95)
    ax1.set_xlim(0, 1)
    _pl(ax1, "(a)")
    
    ax2.plot(phis, sws, color=C5, lw=2.4, zorder=4, label=r"$SW^{RS}(\phi)$")
    _hl(ax2, eq_d["SW"], c=C1, lw=1.3)
    _hl(ax2, eq_c["SW"], c=C3, lw=1.3, ls=(0, (5, 3)), alpha=0.9)
    _sax(ax2, xlabel=r"Revenue-sharing $\phi$", ylabel=r"Social welfare $SW$", title="Social Welfare")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.95)
    _pl(ax2, "(b)")
    
    fig.suptitle("Figure 4. Revenue-Sharing Contract Analysis", fontsize=12, y=0.96, style="italic")
    _save(fig, "CLSC_Fig4_Coordination")

def fig5(eta_data, gamma_data):
    fig = plt.figure(figsize=(14.0, 10.0))
    gs = GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.50, left=0.08, right=0.96, top=0.90, bottom=0.08)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    
    ax = axes[0]
    ax.plot(eta_data["etas"], eta_data["tau_s"], color=C2, lw=2.2, zorder=4, label=r"$\tau^*(\eta)$")
    ax.fill_between(eta_data["etas"], 0, eta_data["tau_s"], color=C2, alpha=0.12, zorder=1)
    axr = ax.twinx()
    axr.plot(eta_data["etas"], eta_data["E_s"], color=C3, lw=2.0, ls="--", zorder=3, label=r"$E^*(\eta)$")
    axr.set_ylabel(r"Emissions $E^*$", color=C3, labelpad=7)
    axr.tick_params(axis="y", colors=C3)
    _sax(ax, xlabel=r"Environmental damage $\eta$", ylabel=r"Optimal tax $\tau^*$", title=r"$\partial\tau^*/\partial\eta > 0$")
    ax.tick_params(axis="y", colors=C2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    _pl(ax, "(a)")
    
    ax = axes[1]
    ax.plot(eta_data["etas"], eta_data["SW_s"], color=C1, lw=2.2, zorder=4)
    ax.fill_between(eta_data["etas"], eta_data["SW_s"].min(), eta_data["SW_s"], color=C1, alpha=0.12, zorder=1)
    _sax(ax, xlabel=r"Environmental damage $\eta$", ylabel=r"Social welfare $SW^*$", title=r"$\partial SW^*/\partial\eta$")
    _pl(ax, "(b)")
    
    ax = axes[2]
    ax.plot(gamma_data["gammas"], gamma_data["tau_s"], color=C2, lw=2.2, zorder=4, label=r"$\tau^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], color=C2, alpha=0.12, zorder=1)
    axr2 = ax.twinx()
    axr2.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C1, lw=2.0, ls="--", zorder=3, label=r"$\rho^*(\gamma)$")
    axr2.set_ylabel(r"Recycling $\rho^*$", color=C1, labelpad=7)
    axr2.tick_params(axis="y", colors=C1)
    _sax(ax, xlabel=r"Green preference $\gamma$", ylabel=r"Optimal tax $\tau^*$", title=r"$\partial\tau^*/\partial\gamma > 0$")
    ax.tick_params(axis="y", colors=C2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    _pl(ax, "(c)")
    
    ax = axes[3]
    ax.plot(gamma_data["gammas"], gamma_data["D_s"], color=C4, lw=2.2, zorder=4, label=r"$D^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), gamma_data["D_s"], color=C4, alpha=0.14, zorder=1)
    axr3 = ax.twinx()
    axr3.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C5, lw=2.0, ls="--", zorder=3, label=r"$\rho^*(\gamma)$")
    axr3.set_ylabel(r"Recycling $\rho^*$", color=C5, labelpad=7)
    axr3.tick_params(axis="y", colors=C5)
    _sax(ax, xlabel=r"Green preference $\gamma$", ylabel=r"Market demand $D^*$", title=r"$\partial D^*/\partial\gamma > 0$")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    _pl(ax, "(d)")
    
    fig.suptitle(r"Figure 5. Comparative Statics", fontsize=12, y=0.97, style="italic")
    _save(fig, "CLSC_Fig5_ComparativeStatics")

def fig6_sensitivity_analysis(eta_data, gamma_data):
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.48, left=0.08, right=0.95, top=0.91, bottom=0.08)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eta_data["etas"], eta_data["tau_s"], color=C2, lw=2.3, zorder=4, label=r"$\tau^*$")
    ax1.fill_between(eta_data["etas"], 0, eta_data["tau_s"], color=C2, alpha=0.12, zorder=1)
    ax1.set_xlabel(r"Environmental damage $\eta$", labelpad=6)
    ax1.set_ylabel(r"Optimal tax $\tau^*$", color=C2, labelpad=7)
    ax1.tick_params(axis='y', labelcolor=C2)
    ax1.grid(True, alpha=0.25)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(eta_data["etas"], eta_data["rho_s"], color=C1, lw=2.1, ls="--", zorder=3, label=r"$\rho^*$")
    ax1_twin.set_ylabel(r"Recycling $\rho^*$", color=C1, labelpad=7)
    ax1_twin.tick_params(axis='y', labelcolor=C1)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax1.set_title(r"Effect of $\eta$", pad=10)
    _pl(ax1, "(a)", x=-0.14, y=1.08)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eta_data["etas"], eta_data["E_s"], color=C4, lw=2.3, zorder=4, label=r"$E^*$")
    ax2.fill_between(eta_data["etas"], eta_data["E_s"].min(), eta_data["E_s"], color=C4, alpha=0.12, zorder=1)
    ax2.set_xlabel(r"Environmental damage $\eta$", labelpad=6)
    ax2.set_ylabel(r"Emissions $E^*$", color=C4, labelpad=7)
    ax2.tick_params(axis='y', labelcolor=C4)
    ax2.grid(True, alpha=0.25)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(eta_data["etas"], eta_data["SW_s"], color=C3, lw=2.1, ls="--", zorder=3, label=r"$SW^*$")
    ax2_twin.set_ylabel(r"Social welfare $SW^*$", color=C3, labelpad=7)
    ax2_twin.tick_params(axis='y', labelcolor=C3)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax2.set_title(r"Effect of $\eta$", pad=10)
    _pl(ax2, "(b)", x=-0.14, y=1.08)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gamma_data["gammas"], gamma_data["tau_s"], color=C2, lw=2.3, zorder=4, label=r"$\tau^*$")
    ax3.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], color=C2, alpha=0.12, zorder=1)
    ax3.set_xlabel(r"Green preference $\gamma$", labelpad=6)
    ax3.set_ylabel(r"Optimal tax $\tau^*$", color=C2, labelpad=7)
    ax3.tick_params(axis='y', labelcolor=C2)
    ax3.grid(True, alpha=0.25)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(gamma_data["gammas"], gamma_data["rho_s"], color=C1, lw=2.1, ls="--", zorder=3, label=r"$\rho^*$")
    ax3_twin.set_ylabel(r"Recycling $\rho^*$", color=C1, labelpad=7)
    ax3_twin.tick_params(axis='y', labelcolor=C1)
    ax3.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax3.set_title(r"Effect of $\gamma$", pad=10)
    _pl(ax3, "(c)", x=-0.14, y=1.08)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(gamma_data["gammas"], gamma_data["D_s"], color=C4, lw=2.3, zorder=4, label=r"$D^*$")
    ax4.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), gamma_data["D_s"], color=C4, alpha=0.14, zorder=1)
    ax4.set_xlabel(r"Green preference $\gamma$", labelpad=6)
    ax4.set_ylabel(r"Market demand $D^*$", color=C4, labelpad=7)
    ax4.tick_params(axis='y', labelcolor=C4)
    ax4.grid(True, alpha=0.25)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(gamma_data["gammas"], gamma_data["SW_s"], color=C3, lw=2.1, ls="--", zorder=3, label=r"$SW^*$")
    ax4_twin.set_ylabel(r"Social welfare $SW^*$", color=C3, labelpad=7)
    ax4_twin.tick_params(axis='y', labelcolor=C3)
    ax4.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax4.set_title(r"Effect of $\gamma$", pad=10)
    _pl(ax4, "(d)", x=-0.14, y=1.08)
    
    fig.suptitle("Figure 6. Sensitivity Analysis", fontsize=12.5, y=0.97, style="italic")
    _save(fig, "CLSC_Fig6_Sensitivity_Analysis")

# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

def print_header():
    print("\n" + "═" * 85)
    print("║" + " " * 83 + "║")
    print("║" + "CLSC Stackelberg Game Model — CORRECTED VERSION".center(83) + "║")
    print("║" + "闭环供应链碳税政策 Stackelberg 博弈模型（修正版）".center(83) + "║")
    print("║" + " " * 83 + "║")
    print("═" * 85)
    print(f"║  运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("║  版本：Corrected v3.0")
    print("║  修正：回收率一阶条件（原论文忽略γ项）")
    print("═" * 85 + "\n")

def print_config(c):
    print("📋 模型基础配置参数:")
    print("─" * 85)
    print(f"  市场规模 (a)              : {c.a:>10.1f}  |  价格弹性 (b)            : {c.b:>10.2f}")
    print(f"  绿色偏好系数 (γ)          : {c.gamma:>10.1f}  |  制造成本 (c_m)          : {c.c_m:>10.1f}")
    print(f"  回收成本 (c_r)            : {c.c_r:>10.1f}  |  回收固定成本 (k)        : {c.k:>10.1f}")
    print(f"  单位排放 (e_m)            : {c.e_m:>10.2f}  |  回收减排 (e_r)          : {c.e_r:>10.2f}")
    print(f"  环境损害权重 (η)          : {c.eta:>10.2f}  |  最大碳税 (τ_max)        : {c.tau_max:>10.1f}")
    print(f"  优化精度 (tol)            : {c.tol:>10.2e}")
    print("─" * 85)
    print(f"  单位回收成本节约 (Δc)     : {c.dc:>10.2f}  |  单位回收减排量 (Δe)     : {c.de:>10.2f}")
    print("─" * 85 + "\n")

def print_equilibrium(eq_nt, eq_d, eq_c, elapsed, method_name):
    print(f"✅ 均衡点求解完成 ({method_name})")
    print(f"   耗时：{elapsed:.2f} 秒\n")
    print("📌 无碳税场景 (τ = 0):")
    print("─" * 85)
    print(f"   回收率 (ρ₀)      : {eq_nt['rho']:>12.4f}  |  总排放 (E₀)      : {eq_nt['E']:>12.2f}")
    print(f"   社会福利 (SW₀)   : {eq_nt['SW']:>12.1f}  |  市场需求 (D₀)    : {eq_nt['D']:>12.2f}")
    print("─" * 85 + "\n")
    print("📌 分散决策场景 (τ*):")
    print("─" * 85)
    print(f"   最优碳税 (τ*)    : {eq_d['tau']:>12.3f}  |  回收率 (ρ*)      : {eq_d['rho']:>12.4f}")
    print(f"   总排放 (E*)      : {eq_d['E']:>12.2f}  |  社会福利 (SW*)   : {eq_d['SW']:>12.1f}")
    print(f"   零售价格 (p*)    : {eq_d['p']:>12.2f}  |  批发价格 (w*)    : {eq_d['w']:>12.2f}")
    print("─" * 85 + "\n")
    print("📌 集中决策场景 (VIF):")
    print("─" * 85)
    print(f"   最优碳税 (τ^C)   : {eq_c['tau']:>12.3f}  |  回收率 (ρ^C)     : {eq_c['rho']:>12.4f}")
    print(f"   总排放 (E^C)     : {eq_c['E']:>12.2f}  |  社会福利 (SW^C)  : {eq_c['SW']:>12.1f}")
    print(f"   零售价格 (p^C)   : {eq_c['p']:>12.2f}  |  VIF 利润 (π_VIF)  : {eq_c['pi_vif']:>12.2f}")
    print("─" * 85 + "\n")

def print_comparison(original, corrected):
    print("📊 原方法 vs 修正方法 对比:")
    print("─" * 85)
    print(f"{'指标':<20} {'原方法':>15} {'修正方法':>15} {'相对差异':>15}")
    print("─" * 85)
    keys = ['tau', 'rho', 'p', 'w', 'D', 'E', 'SW']
    labels = ['最优碳税 τ*', '回收率 ρ*', '零售价 p*', '批发价 w*', '需求 D*', '排放 E*', '福利 SW*']
    for k, label in zip(keys, labels):
        orig_val = original[k]
        corr_val = corrected[k]
        diff_pct = (corr_val - orig_val) / orig_val * 100 if orig_val != 0 else 0
        print(f"{label:<20} {orig_val:>15.4f} {corr_val:>15.4f} {diff_pct:>14.2f}%")
    print("─" * 85 + "\n")

def print_key_metrics(eq_nt, eq_d, eq_c):
    print("🎯 核心结论与关键指标:")
    print("─" * 85)
    delta_rho = eq_c['rho'] - eq_d['rho']
    poa = eq_d['SW'] / eq_c['SW']
    emission_reduction = eq_nt['E'] - eq_d['E']
    welfare_improvement = eq_d['SW'] - eq_nt['SW']
    vif_advantage = eq_c['SW'] - eq_d['SW']
    green_paradox = eq_c['E'] > eq_d['E']
    print(f"  【回收率协调缺口】ρ^C - ρ*         = {delta_rho:>10.4f}")
    print(f"  【价格无效率】PoA = SW*/SW^C       = {poa:>10.4f}  (越接近 1 越优)")
    print(f"  【碳税减排效果】E₀ - E*           = {emission_reduction:>10.2f} 单位")
    print(f"  【福利提升】SW* - SW₀             = {welfare_improvement:>10.1f}")
    print(f"  【集中决策福利优势】SW^C - SW*    = {vif_advantage:>10.1f}")
    print("─" * 85)
    print(f"  【绿色悖论检测】E^C > E*          : {'⚠️  存在' if green_paradox else '✓ 不存在'}")
    if green_paradox:
        print(f"     说明：集中决策下更高的回收率导致需求扩张，排放反而增加")
    print("─" * 85 + "\n")

def print_summary_table(eq_nt, eq_d, eq_c):
    print("📊 数值汇总表:")
    print("─" * 95)
    print(f"{'指标':<28} {'无碳税 (τ=0)':>18} {'分散决策 (τ*)':>20} {'集中决策 (VIF)':>20}")
    print("─" * 95)
    print(f"{'碳税 (τ)':<28} {0.00:>18.3f} {eq_d['tau']:>20.3f} {eq_c['tau']:>20.3f}")
    print(f"{'零售价格 (p)':<28} {eq_nt['p']:>18.2f} {eq_d['p']:>20.2f} {eq_c['p']:>20.2f}")
    print(f"{'批发价格 (w)':<28} {eq_nt['w']:>18.2f} {eq_d['w']:>20.2f} {'-':>20}")
    print(f"{'回收率 (ρ)':<28} {eq_nt['rho']:>18.4f} {eq_d['rho']:>20.4f} {eq_c['rho']:>20.4f}")
    print(f"{'市场需求 (D)':<28} {eq_nt['D']:>18.2f} {eq_d['D']:>20.2f} {eq_c['D']:>20.2f}")
    print(f"{'总排放 (E)':<28} {eq_nt['E']:>18.2f} {eq_d['E']:>20.2f} {eq_c['E']:>20.2f}")
    print(f"{'制造商利润 (π_m)':<28} {eq_nt['pi_m']:>18.2f} {eq_d['pi_m']:>20.2f} {eq_c['pi_vif']:>20.2f}")
    print(f"{'零售商利润 (π_r)':<28} {eq_nt['pi_r']:>18.2f} {eq_d['pi_r']:>20.2f} {'-':>20}")
    print(f"{'消费者剩余 (CS)':<28} {eq_nt['CS']:>18.2f} {eq_d['CS']:>20.2f} {eq_c['CS']:>20.2f}")
    print(f"{'碳税收入 (τE)':<28} {0.0:>18.2f} {eq_d['tax_rev']:>20.2f} {eq_c['tax_rev']:>20.2f}")
    print(f"{'环境损害 (ηE²)':<28} {eq_nt['env_dam']:>18.2f} {eq_d['env_dam']:>20.2f} {eq_c['env_dam']:>20.2f}")
    print(f"{'社会福利 (SW)':<28} {eq_nt['SW']:>18.1f} {eq_d['SW']:>20.1f} {eq_c['SW']:>20.1f}")
    print("─" * 95 + "\n")

def main():
    total_start = time.perf_counter()
    print_header()
    c = Cfg()
    print_config(c)
    
    print("🔬 [1/6] 对比测试：原论文方法 vs 修正方法...")
    print("   → 原论文方法（解析近似）...", end=" ", flush=True)
    t0 = time.perf_counter()
    eq_nt_orig = dec_eq(c, tau=0.0, use_corrected=False)
    eq_d_orig = dec_eq(c, use_corrected=False)
    eq_c_orig = vif_eq(c)
    t_orig = time.perf_counter() - t0
    print(f"完成 ({t_orig:.2f}s)")
    
    print("   → 修正方法（数值优化）...", end=" ", flush=True)
    t0 = time.perf_counter()
    eq_nt_corr = dec_eq(c, tau=0.0, use_corrected=True)
    eq_d_corr = dec_eq(c, use_corrected=True)
    eq_c_corr = vif_eq(c)
    t_corr = time.perf_counter() - t0
    print(f"完成 ({t_corr:.2f}s)")
    
    print()
    print_comparison(eq_d_orig, eq_d_corr)
    
    print("🔢 [2/6] 使用修正方法求解各场景均衡点...")
    print_equilibrium(eq_nt_corr, eq_d_corr, eq_c_corr, time.perf_counter() - total_start, "数值优化ρ")
    
    print("📈 [3/6] 开始计算敏感性分析数据...")
    sens_start = time.perf_counter()
    
    print("   → 碳税扫描 (τ sweep, n=150)...", end=" ", flush=True)
    t0 = time.perf_counter()
    sens = tau_sweep(c, n=150, use_corrected=True)
    print(f"完成 ({time.perf_counter() - t0:.2f}s)")
    
    print("   → 环境损害敏感性 (η sweep, n=60)...", end=" ", flush=True)
    t0 = time.perf_counter()
    eta_d = eta_sweep(c, n=60, use_corrected=True)
    print(f"完成 ({time.perf_counter() - t0:.2f}s)")
    
    print("   → 绿色偏好敏感性 (γ sweep, n=60)...", end=" ", flush=True)
    t0 = time.perf_counter()
    gam_d = gamma_sweep(c, n=60, use_corrected=True)
    print(f"完成 ({time.perf_counter() - t0:.2f}s)")
    
    print("   → 收益共享契约分析 (φ sweep, n=80)...", end=" ", flush=True)
    t0 = time.perf_counter()
    phi_d = phi_sweep(c, eq_d_corr["tau"], n=80, use_corrected=True)
    print(f"完成 ({time.perf_counter() - t0:.2f}s)")
    
    sens_elapsed = time.perf_counter() - sens_start
    print(f"✅ 敏感性分析完成 (总耗时：{sens_elapsed:.2f} 秒)\n")
    
    print("🎨 [4/6] 开始渲染 6 个高质量论文级图形 (600 DPI)...")
    fig_start = time.perf_counter()
    
    print("   → 生成图 1: 定价与回收率动态... ", end="", flush=True)
    t0 = time.perf_counter()
    fig1(eq_nt_corr, eq_d_corr, eq_c_corr, sens)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    print("   → 生成图 2: 排放与社会福利... ", end="", flush=True)
    t0 = time.perf_counter()
    fig2(eq_nt_corr, eq_d_corr, eq_c_corr, sens)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    print("   → 生成图 3: 场景对比与福利分解... ", end="", flush=True)
    t0 = time.perf_counter()
    fig3(eq_nt_corr, eq_d_corr, eq_c_corr)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    print("   → 生成图 4: 收益共享契约分析... ", end="", flush=True)
    t0 = time.perf_counter()
    fig4(eq_d_corr, eq_c_corr, phi_d)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    print("   → 生成图 5: 比较静态分析... ", end="", flush=True)
    t0 = time.perf_counter()
    fig5(eta_d, gam_d)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    print("   → 生成图 6: 敏感性分析综合图... ", end="", flush=True)
    t0 = time.perf_counter()
    fig6_sensitivity_analysis(eta_d, gam_d)
    print(f"({time.perf_counter() - t0:.2f}s)")
    
    fig_elapsed = time.perf_counter() - fig_start
    print(f"✅ 所有图形生成完成 (总耗时：{fig_elapsed:.2f} 秒)")
    print("   📁 图形文件已保存到当前目录 (PDF/PNG 格式，600 DPI)")
    print("   📊 包含：基础分析图×5 + 敏感性分析图×1 (共 6 个图形)\n")
    
    print_summary_table(eq_nt_corr, eq_d_corr, eq_c_corr)
    print_key_metrics(eq_nt_corr, eq_d_corr, eq_c_corr)
    
    total_elapsed = time.perf_counter() - total_start
    print("═" * 85)
    print(f"🏁 模型运行完成！总耗时：{total_elapsed:.2f} 秒".center(85))
    print("═" * 85)
    print("\n📝 修正说明:")
    print("   • 原论文回收率一阶条件忽略了 γ 项（回收率通过需求影响利润的间接效应）")
    print("   • 修正方法：使用数值优化直接求解最优 ρ，避免解析近似误差")
    print("   • 对比结果显示：原方法误差见上方对比表")
    print("\n💡 建议:")
    print("   • 论文中应补充完整的一阶条件推导")
    print("   • 或在附录中说明数值方法与解析近似的等价性条件")
    print("═" * 85 + "\n")

if __name__ == "__main__":
    main()
