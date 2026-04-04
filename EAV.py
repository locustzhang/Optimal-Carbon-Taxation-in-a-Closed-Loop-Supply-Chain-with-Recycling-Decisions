""" CLSC Stackelberg Game Model — CORRECTED VERSION
=====================================================
闭环供应链碳税政策 Stackelberg 博弈模型（修正版）

修正说明:
    • 修正回收率一阶条件（原论文忽略了 γ 项）
    • 使用数值优化直接求解最优ρ，避免解析近似误差
    • 保持与论文其他部分一致

图形增强（Management Science 期刊标准）:
    • 严格遵循 MS/OR 顶刊图形规范
    • 双栏 (3.5") / 单栏 (7.0") 宽度控制
    • Elsevier/APA 标准字号层级
    • 高对比度、色盲友好配色（色盲模拟验证）
    • 精细 tick、spines、axes 设置
    • 完整图注 (caption-ready)
    • 600 DPI PDF/PNG 双格式输出

作者：飞翔的蚂蚱 · 严谨专业版
版本：MS-Enhanced v4.0
日期：2026-04-03
"""

import numpy as np

# ── 在 matplotlib/fontTools 载入前静默 timestamp 警告 ────────
import logging
logging.getLogger("fontTools").setLevel(logging.CRITICAL)
logging.getLogger("fonttools").setLevel(logging.CRITICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

# ══════════════════════════════════════════════════════════════════
# ██ MANAGEMENT SCIENCE 期刊图形规范设置
# ══════════════════════════════════════════════════════════════════

def _best_serif() -> str:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["Latin Modern Roman", "Lora", "Caladea", "Liberation Serif", 
                 "DejaVu Serif", "FreeSerif", "serif"]:
        if font in available:
            return font
    return "serif"

_SERIF = _best_serif()

# ── MS 期刊精确尺寸规范 ────────────────────────────────────────
# Management Science: 双栏 3.46", 单栏 7.09", 最大高度 9.0"
MS_COL2 = 3.46    # 双栏宽度 (inches)
MS_COL1 = 7.09    # 单栏宽度 (inches)
MS_H_SM = 2.60    # 标准小图高度
MS_H_MD = 3.20    # 标准中图高度
MS_H_LG = 6.50    # 大图/2×2

# ── 色盲友好配色方案（Wong 2011 Nature Methods）────────────────
# 验证：Deuteranopia / Protanopia 均可区分
C1 = "#0072B2"    # 深蓝 (主线—制造商/批发)
C2 = "#D55E00"    # 橙红 (最优点/标注)
C3 = "#009E73"    # 深绿 (集中决策 VIF)
C4 = "#E69F00"    # 金黄 (需求/收益)
C5 = "#CC79A7"    # 粉紫 (收益共享)
CG = "#56B4E9"    # 浅蓝 (参考线/零碳税基线)
CB = "#404040"    # 深灰 (axes / spines)
CL = "#F5F5F5"    # 浅灰背景带

# ── 线型集合 ───────────────────────────────────────────────────
LS_SOLID = "-"
LS_DASH = "--"
LS_DOT = (0, (4, 1.5))
LS_DASD = (0, (6, 2, 1, 2))
LS_DDOT = (0, (1, 1.5))

# ── MS 标准字号层级 (pt) ───────────────────────────────────────
FS_PANEL = 9.0    # 子图标签 (a)(b)
FS_TITLE = 8.5    # 子图标题
FS_LABEL = 8.0    # 轴标签
FS_TICK = 7.5     # 刻度标签
FS_LEGEND = 7.5   # 图例
FS_ANNOT = 7.0    # 标注文字
FS_SUPER = 9.5    # 图总标题

plt.rcParams.update({
    # 字体
    "font.family" : _SERIF,
    "mathtext.fontset" : "cm",
    "font.size" : FS_LABEL,
    "axes.labelsize" : FS_LABEL,
    "axes.titlesize" : FS_TITLE,
    "axes.titleweight" : "normal",
    "axes.titlepad" : 5,
    "axes.labelpad" : 4,
    
    # 图例
    "legend.fontsize" : FS_LEGEND,
    "legend.title_fontsize" : FS_LEGEND,
    
    # 刻度
    "xtick.labelsize" : FS_TICK,
    "ytick.labelsize" : FS_TICK,
    
    # Axes 线宽 & 样式
    "axes.linewidth" : 0.75,
    "lines.linewidth" : 1.6,
    "lines.markersize" : 4.5,
    "patch.linewidth" : 0.6,
    
    # Tick 方向与尺寸
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.major.width" : 0.75,
    "ytick.major.width" : 0.75,
    "xtick.minor.width" : 0.5,
    "ytick.minor.width" : 0.5,
    "xtick.major.size" : 3.5,
    "ytick.major.size" : 3.5,
    "xtick.minor.size" : 2.0,
    "ytick.minor.size" : 2.0,
    "xtick.major.pad" : 3,
    "ytick.major.pad" : 3,
    "xtick.top" : True,          # MS 标准：四边 tick
    "ytick.right" : True,
    
    # Spines
    "axes.spines.top" : True,
    "axes.spines.right" : True,
    
    # Grid
    "grid.alpha" : 0.20,
    "grid.linestyle" : "-",
    "grid.linewidth" : 0.4,
    "grid.color" : "#999999",
    
    # 图例框
    "legend.frameon" : True,
    "legend.framealpha" : 1.0,
    "legend.edgecolor" : "#CCCCCC",
    "legend.handlelength" : 2.0,
    "legend.handleheight" : 0.8,
    "legend.borderpad" : 0.5,
    "legend.labelspacing" : 0.35,
    "legend.handletextpad" : 0.5,
    "legend.columnspacing" : 1.2,
    
    # 图面
    "figure.facecolor" : "white",
    "axes.facecolor" : "white",
    
    # 输出
    "savefig.dpi" : 600,
    "savefig.bbox" : "tight",
    "savefig.pad_inches" : 0.04,
    "pdf.fonttype" : 42,         # TrueType 嵌入，Elsevier 要求
    "ps.fonttype" : 42,
})

# ══════════════════════════════════════════════════════════════════
# ██ 辅助绘图工具函数（MS 规范强化版）
# ══════════════════════════════════════════════════════════════════

def _style_ax(ax, xlabel="", ylabel="", title="", grid=True, 
              ylabel_color=None, xlabel_color=None):
    """统一应用 MS 标准样式到一个 Axes"""
    if xlabel:
        kw = dict(labelpad=4, fontsize=FS_LABEL)
        if xlabel_color: kw["color"] = xlabel_color
        ax.set_xlabel(xlabel, **kw)
    if ylabel:
        kw = dict(labelpad=5, fontsize=FS_LABEL)
        if ylabel_color: kw["color"] = ylabel_color
        ax.set_ylabel(ylabel, **kw)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, pad=4, fontstyle="italic")
    if grid:
        ax.grid(True, which="major", zorder=0, alpha=0.20, lw=0.4)
    
    # 副刻度
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Spines 统一颜色
    for sp in ax.spines.values():
        sp.set_linewidth(0.75)
        sp.set_color(CB)
    
    # Tick 样式
    ax.tick_params(which="major", length=3.5, width=0.75, 
                   top=True, right=True, direction="in")
    ax.tick_params(which="minor", length=2.0, width=0.5, 
                   top=True, right=True, direction="in")
    return ax

def _panel_label(ax, label, x=-0.16, y=1.06):
    """子图面板标签 (a)(b)(c)…"""
    ax.text(x, y, label, transform=ax.transAxes, 
            fontsize=FS_PANEL, fontweight="bold", 
            va="top", ha="left", color="#111111")

def _vline(ax, x, color=C2, lw=1.0, ls=LS_DOT, alpha=0.85, zorder=4):
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

def _hline(ax, y, color=CG, lw=0.9, ls=LS_DASH, alpha=0.75, zorder=4):
    ax.axhline(y, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

def _dot(ax, x, y, color=C2, s=28, z=10, marker="o"):
    ax.scatter([x], [y], color=color, s=s, zorder=z, 
               edgecolors="white", linewidths=0.8, marker=marker)

def _bracket_arrow(ax, x, y0, y1, color=C2, dx=3, lw=0.9):
    """双向箭头标注两个水平量之间的差距"""
    ax.annotate("", xy=(x + dx, y1), xytext=(x + dx, y0),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, 
                               mutation_scale=7))

def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(f"./{name}.{ext}", dpi=600, bbox_inches="tight", 
                   facecolor="white")
    plt.close(fig)
    print(f" ✓ {name}.{{pdf,png}}")

def _twin_style(ax_twin, ylabel="", color=C4):
    """右侧双轴样式"""
    ax_twin.set_ylabel(ylabel, labelpad=5, fontsize=FS_LABEL, color=color)
    ax_twin.tick_params(axis="y", which="both", colors=color, 
                       labelsize=FS_TICK, direction="in", 
                       length=3.5, width=0.75)
    ax_twin.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_twin.spines["right"].set_color(color)
    ax_twin.spines["right"].set_linewidth(0.75)
    for sp in ["top", "left", "bottom"]:
        ax_twin.spines[sp].set_visible(False)
    return ax_twin

# ══════════════════════════════════════════════════════════════════
# ██ 模型配置
# ══════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════
# ██ 核心模型函数
# ══════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════
# ██ 回收率反应函数（修正版 - 数值优化）
# ══════════════════════════════════════════════════════════════════

def rs_optimal(c, w, tau):
    """修正版 - 数值优化求解最优回收率"""
    def neg_pi_rho(rho):
        p = prf(c, w, rho)
        d = Df(c, p, rho)
        em = Ef(c, d, rho)
        profit = (w - c.c_m) * d + c.dc * rho * d - tau * em - c.k * rho**2
        return -profit
    
    res = minimize_scalar(neg_pi_rho, bounds=(0, 1), method="bounded", 
                         options={"xatol": 1e-12})
    return float(np.clip(res.x, 0, 1))

# ══════════════════════════════════════════════════════════════════
# ██ 均衡求解器
# ══════════════════════════════════════════════════════════════════

def dec_solve(c, tau):
    """求解分散式均衡（给定碳税τ）"""
    def solve_rho_for_w(w, rho_init=0.15):
        return rs_optimal(c, w, tau)
    
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

def dec_eq(c, tau=None):
    """求解分散式均衡（优化τ）"""
    if tau is not None:
        return dec_solve(c, tau)
    
    res = minimize_scalar(
        lambda t: -dec_solve(c, t)["SW"],
        bounds=(0, c.tau_max), 
        method="bounded", 
        options={"xatol": c.tol}
    )
    return dec_solve(c, float(res.x))

def vif_br(c, tau):
    """VIF 最优反应（联合优化 p 和ρ）"""
    def neg_pi_vif(x):
        return -pivif(c, x[0], x[1], tau)
    
    res = minimize(neg_pi_vif, 
                  x0=[350, 0.3], 
                  bounds=[(c.c_m + 0.01, c.a / c.b - 0.01), (0, 1)], 
                  method="L-BFGS-B", 
                  options={"ftol": 1e-15, "gtol": 1e-12})
    return float(res.x[0]), float(np.clip(res.x[1], 0, 1))

def vif_sw_val(c, tau):
    """VIF 社会福利计算（加入参与约束）"""
    p, rho = vif_br(c, tau)
    profit_vif = pivif(c, p, rho, tau)
    
    if profit_vif < 0:
        return -1e10
    
    d = Df(c, p, rho)
    em = Ef(c, d, rho)
    cs = CSf(c, p, rho)
    return SWf(c, cs, profit_vif, 0, tau, em)

def vif_eq(c):
    """求解 VIF 均衡（优化τ）"""
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

# ══════════════════════════════════════════════════════════════════
# ██ 敏感性分析
# ══════════════════════════════════════════════════════════════════

def tau_sweep(c, n=150):
    """碳税扫描"""
    taus = np.linspace(0, c.tau_max, n)
    keys = ["w", "p", "rho", "D", "E", "pi_m", "pi_r", "CS", "SW", 
            "env_dam", "tax_rev"]
    out = {k: np.empty(n) for k in keys}
    out["taus"] = taus
    
    for i, t in enumerate(taus):
        eq = dec_solve(c, t)
        for k in keys:
            out[k][i] = eq[k]
    return out

def eta_sweep(c, n=60):
    """环境损害系数η敏感性分析"""
    etas = np.linspace(0.5, 8.0, n)
    ts, Es, SWs, rhos = [], [], [], []
    orig = c.eta
    
    for ev in etas:
        c.eta = ev
        eq = dec_eq(c)
        ts.append(eq["tau"])
        Es.append(eq["E"])
        SWs.append(eq["SW"])
        rhos.append(eq["rho"])
    
    c.eta = orig
    return dict(etas=etas, tau_s=np.array(ts), E_s=np.array(Es), 
                SW_s=np.array(SWs), rho_s=np.array(rhos))

def gamma_sweep(c, n=60):
    """绿色偏好系数γ敏感性分析"""
    gammas = np.linspace(2, 28, n)
    ts, rhos, Ds, SWs = [], [], [], []
    orig = c.gamma
    
    for gv in gammas:
        c.gamma = gv
        eq = dec_eq(c)
        ts.append(eq["tau"])
        rhos.append(eq["rho"])
        Ds.append(eq["D"])
        SWs.append(eq["SW"])
    
    c.gamma = orig
    return dict(gammas=gammas, tau_s=np.array(ts), rho_s=np.array(rhos), 
                D_s=np.array(Ds), SW_s=np.array(SWs))

def phi_sweep(c, tau, n=80):
    """收益共享契约分析"""
    phis = np.linspace(0.01, 0.99, n)
    pms, prs, sws = [], [], []
    
    for phi in phis:
        w = c.c_m
        rho = 0.3
        for _ in range(50):
            p = (c.a + c.gamma * rho) / (2 * c.b) + w / (2 * (1 - phi))
            d = Df(c, p, rho)
            rho_new = rs_optimal(c, w, tau)
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
    
    return dict(phis=phis, pi_m=np.array(pms), pi_r=np.array(prs), 
                sw=np.array(sws))

# ══════════════════════════════════════════════════════════════════
# ██ Figure 1 — Pricing Dynamics & Recycling Rate
# ══════════════════════════════════════════════════════════════════

def fig1(eq_nt, eq_d, eq_c, sens):
    """
    Figure 1. Equilibrium Pricing and Recycling Rate as Functions of Carbon Tax.
    Panel (a): wholesale price w*(τ) and retail price p*(τ) with margin shading.
    Panel (b): manufacturer's recycling rate ρ*(τ) with VIF benchmark.
    """
    W = MS_COL1
    H = MS_H_MD + 0.1
    fig, axes = plt.subplots(1, 2, figsize=(W, H), 
                            gridspec_kw={"wspace": 0.42, "left": 0.09, 
                                        "right": 0.96, "top": 0.87, 
                                        "bottom": 0.16})
    ax1, ax2 = axes
    taus = sens["taus"]
    τ_star = eq_d["tau"]
    
    # ── Panel (a): 价格 ────────────────────────────────────────
    # 填充零售商利润区间
    ax1.fill_between(taus, sens["w"], sens["p"], color=C1, alpha=0.10, 
                    zorder=1, label="Retailer margin")
    
    # 主线
    ax1.plot(taus, sens["p"], color=C1, lw=1.8, zorder=4, 
            label=r"Retail price $p^*(\tau)$")
    ax1.plot(taus, sens["w"], color=C1, lw=1.5, ls=LS_DASH, zorder=4, 
            label=r"Wholesale price $w^*(\tau)$")
    
    # VIF 参考水平线
    _hline(ax1, eq_c["p"], color=C3, lw=1.0, ls=LS_DASD, alpha=0.9)
    ax1.text(taus[-1] * 0.98, eq_c["p"] + 1.8, 
            rf"$p^{{C}}={eq_c['p']:.1f}$" + "\n(VIF)", 
            ha="right", va="bottom", fontsize=FS_ANNOT, color=C3, 
            style="italic", linespacing=1.3)
    
    # 最优碳税垂线
    _vline(ax1, τ_star, color=C2, lw=0.9, ls=LS_DOT)
    
    # 点标注
    _dot(ax1, τ_star, eq_d["p"], color=C2, s=22)
    _dot(ax1, τ_star, eq_d["w"], color=C2, s=22)
    
    # 双向箭头标注边际利润
    margin = eq_d["p"] - eq_d["w"]
    _bracket_arrow(ax1, τ_star, eq_d["w"], eq_d["p"], color=C2, dx=4, lw=0.9)
    ax1.text(τ_star + 7, (eq_d["p"] + eq_d["w"]) / 2, 
            rf"$\Delta={margin:.1f}$", fontsize=FS_ANNOT, 
            color=C2, va="center")
    
    # τ* 标签框
    ax1.text(τ_star, sens["p"].max() - 4, 
            rf"$\tau^*={τ_star:.1f}$", 
            fontsize=FS_ANNOT + 0.3, color=C2, fontweight="bold", 
            ha="center", va="top", 
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C2, 
                     alpha=0.95, lw=0.65))
    
    _style_ax(ax1, xlabel=r"Carbon tax $\tau$", ylabel="Price", 
             title="(a) Pricing Dynamics")
    leg1 = ax1.legend(loc="lower right", fontsize=FS_LEGEND, 
                     framealpha=1.0, edgecolor="#CCCCCC", handlelength=1.8)
    ax1.set_xlim(0, taus[-1])
    ax1.set_ylim(275, 385)
    
    # ── Panel (b): 回收率 ──────────────────────────────────────
    # 背景带：τ < τ* 区域（低于最优碳税区）
    ax2.axvspan(0, τ_star, color="#F8F0E3", alpha=0.55, zorder=0, lw=0)
    ax2.axvspan(τ_star, taus[-1], color="#EEF4F8", alpha=0.45, zorder=0, lw=0)
    
    # 主曲线
    ax2.plot(taus, sens["rho"], color=C1, lw=1.8, zorder=4, 
            label=r"$\rho^*(\tau)$: Decentralized")
    
    # VIF 参考线
    _hline(ax2, eq_c["rho"], color=C3, lw=1.1, ls=LS_DASD, alpha=0.95)
    ax2.text(taus[-1] * 0.98, eq_c["rho"] + 0.010, 
            rf"$\rho^{{C}}={eq_c['rho']:.3f}$" + "\n(VIF centralized)", 
            ha="right", va="bottom", fontsize=FS_ANNOT, color=C3, 
            style="italic", linespacing=1.3)
    
    # τ* 垂线与标注
    _vline(ax2, τ_star, color=C2, lw=0.9, ls=LS_DOT)
    _dot(ax2, τ_star, eq_d["rho"], color=C2, s=22)
    
    gap = eq_c["rho"] - eq_d["rho"]
    _bracket_arrow(ax2, τ_star, eq_d["rho"], eq_c["rho"], 
                  color=C2, dx=4, lw=0.9)
    ax2.text(τ_star + 7, (eq_d["rho"] + eq_c["rho"]) / 2, 
            rf"$\Delta\rho={gap:.3f}$", fontsize=FS_ANNOT, 
            color=C2, va="center", fontweight="bold")
    
    # 区域标签
    ax2.text(τ_star / 2, 0.02, r"$\tau < \tau^*$", 
            ha="center", va="bottom", fontsize=FS_ANNOT, 
            color="#A07040", fontstyle="italic")
    ax2.text((τ_star + taus[-1]) / 2, 0.02, r"$\tau > \tau^*$", 
            ha="center", va="bottom", fontsize=FS_ANNOT, 
            color="#3060A0", fontstyle="italic")
    
    _style_ax(ax2, xlabel=r"Carbon tax $\tau$", 
             ylabel=r"Recycling rate $\rho^*$", 
             title="(b) Recycling Rate Response")
    ax2.legend(loc="lower right", fontsize=FS_LEGEND, 
              framealpha=1.0, edgecolor="#CCCCCC")
    ax2.set_xlim(0, taus[-1])
    ax2.set_ylim(0, 0.60)
    
    # 总标题（MS 期刊格式：斜体）
    fig.suptitle(
        "Figure 1. Supply Chain Equilibrium under Carbon Tax Policy", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=0.975)
    
    _save(fig, "CLSC_Fig1_Pricing_Recycling")

# ══════════════════════════════════════════════════════════════════
# ██ Figure 2 — Environmental Impact & Social Welfare
# ══════════════════════════════════════════════════════════════════

def fig2(eq_nt, eq_d, eq_c, sens):
    """
    Figure 2. Emissions and Social Welfare under Decentralized Carbon Taxation.
    Panel (a): E*(τ) with no-tax baseline.
    Panel (b): SW(τ) with welfare decomposition reference lines.
    """
    W = MS_COL1
    H = MS_H_MD + 0.1
    fig, axes = plt.subplots(1, 2, figsize=(W, H), 
                            gridspec_kw={"wspace": 0.42, "left": 0.09, 
                                        "right": 0.96, "top": 0.87, 
                                        "bottom": 0.16})
    ax1, ax2 = axes
    taus = sens["taus"]
    τ_star = eq_d["tau"]
    
    # ── Panel (a): 排放量 ──────────────────────────────────────
    ax1.fill_between(taus, eq_nt["E"], sens["E"], 
                    where=sens["E"] <= eq_nt["E"], 
                    color=C3, alpha=0.13, zorder=1, 
                    label="Emission reduction")
    ax1.plot(taus, sens["E"], color=C4, lw=1.8, zorder=4, 
            label=r"$E^*(\tau)$: Total emissions")
    
    # 基准线（无碳税）
    _hline(ax1, eq_nt["E"], color=CG, lw=1.0, ls=LS_DASH, alpha=0.85)
    ax1.text(4, eq_nt["E"] + 0.7, 
            rf"$E_0={eq_nt['E']:.1f}$ (no tax)", 
            fontsize=FS_ANNOT, color="#3060A0", style="italic")
    
    # VIF 参考
    _hline(ax1, eq_c["E"], color=C3, lw=0.9, ls=LS_DASD, alpha=0.80)
    ax1.text(4, eq_c["E"] - 1.8, 
            rf"$E^C={eq_c['E']:.1f}$ (VIF)", 
            fontsize=FS_ANNOT, color=C3, style="italic")
    
    # 最优碳税点
    _vline(ax1, τ_star, color=C2, lw=0.9, ls=LS_DOT)
    _dot(ax1, τ_star, eq_d["E"], color=C2, s=22)
    ax1.text(τ_star + 3, eq_d["E"] + 0.5, 
            rf"$\tau^*={τ_star:.1f}$", 
            fontsize=FS_ANNOT, color=C2, style="italic")
    
    _style_ax(ax1, xlabel=r"Carbon tax $\tau$", 
             ylabel=r"Total emissions $E$", 
             title="(a) Environmental Impact")
    ax1.legend(loc="lower left", fontsize=FS_LEGEND, 
              framealpha=1.0, edgecolor="#CCCCCC")
    ax1.set_xlim(0, taus[-1])
    
    # ── Panel (b): 社会福利 ────────────────────────────────────
    sw_max = sens["SW"].max()
    idx_max = int(np.argmax(sens["SW"]))
    τ_sw_max = sens["taus"][idx_max]
    
    ax2.fill_between(taus, eq_nt["SW"], sens["SW"], 
                    where=sens["SW"] >= eq_nt["SW"], 
                    color=C1, alpha=0.10, zorder=1, 
                    label="Welfare gain from tax")
    ax2.plot(taus, sens["SW"], color=C1, lw=1.8, zorder=4, 
            label=r"$SW(\tau)$: Social welfare")
    
    # 参考线
    _hline(ax2, eq_nt["SW"], color=CG, lw=0.9, ls=LS_DASH, alpha=0.85)
    ax2.text(4, eq_nt["SW"] - 800, 
            rf"$SW_0={eq_nt['SW']:.0f}$ (no tax)", 
            fontsize=FS_ANNOT, color="#3060A0", style="italic")
    
    _hline(ax2, eq_c["SW"], color=C3, lw=0.9, ls=LS_DASD, alpha=0.85)
    ax2.text(taus[-1] * 0.60, eq_c["SW"] + 400, 
            rf"$SW^C={eq_c['SW']:.0f}$ (VIF)", 
            fontsize=FS_ANNOT, color=C3, style="italic")
    
    # 最优点
    _vline(ax2, τ_sw_max, color=C2, lw=0.9, ls=LS_DOT)
    _dot(ax2, τ_sw_max, sw_max, color=C2, s=30, marker="*")
    ax2.text(τ_sw_max + 3, sw_max - 800, 
            rf"$\tau^*_{{SW}}={τ_sw_max:.1f}$", 
            fontsize=FS_ANNOT, color=C2, style="italic")
    
    _style_ax(ax2, xlabel=r"Carbon tax $\tau$", 
             ylabel=r"Social welfare $SW$", 
             title="(b) Social Welfare")
    ax2.legend(loc="upper right", fontsize=FS_LEGEND, 
              framealpha=1.0, edgecolor="#CCCCCC")
    ax2.set_xlim(0, taus[-1])
    
    fig.suptitle(
        "Figure 2. Environmental Impact and Social Welfare under Carbon Tax", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=0.975)
    
    _save(fig, "CLSC_Fig2_Emission_Welfare")

# ══════════════════════════════════════════════════════════════════
# ██ Figure 3 — Scenario Comparison & Welfare Decomposition
# ══════════════════════════════════════════════════════════════════

def fig3(eq_nt, eq_d, eq_c):
    """
    Figure 3. Scenario Comparison and Welfare Decomposition.
    Panel (a): bar chart of ρ across scenarios with SW overlay.
    Panel (b): waterfall decomposition of ΔSW (tax vs. no-tax).
    """
    W = MS_COL1
    H = MS_H_MD + 0.3
    fig, axes = plt.subplots(1, 2, figsize=(W, H), 
                            gridspec_kw={"wspace": 0.50, "left": 0.09, 
                                        "right": 0.96, "top": 0.87, 
                                        "bottom": 0.17})
    ax1, ax2 = axes
    
    # ── Panel (a): 场景柱形图 ──────────────────────────────────
    scenarios = ["No-Tax\n" + r"$(\tau=0)$", 
                 "Decentralized\n" + r"$(\tau^*)$", 
                 "Centralized\n(VIF)"]
    colors3 = [CG, C1, C3]
    rhos = [eq_nt["rho"], eq_d["rho"], eq_c["rho"]]
    sws = [eq_nt["SW"], eq_d["SW"], eq_c["SW"]]
    x = np.arange(3)
    w_bar = 0.45
    
    bars = ax1.bar(x, rhos, w_bar, color=colors3, alpha=0.85, 
                  edgecolor="white", lw=0.6, zorder=3)
    
    # 数值标注
    for i, (bar, v) in enumerate(zip(bars, rhos)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.006, 
                f"{v:.3f}", ha="center", va="bottom", 
                fontsize=FS_ANNOT + 0.5, fontweight="bold", 
                color=colors3[i])
    
    ax1.set_ylabel(r"Recycling rate $\rho$", labelpad=5, fontsize=FS_LABEL)
    ax1.set_ylim(0, max(rhos) * 1.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=FS_TICK + 0.2, linespacing=1.3)
    ax1.tick_params(which="major", direction="in", top=False, right=False, 
                   length=3.5, width=0.75)
    ax1.tick_params(which="minor", bottom=False, top=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.20, lw=0.4, zorder=0)
    
    # 右轴：SW 折线
    ax1r = ax1.twinx()
    ax1r.plot(x, sws, "D--", color=C4, lw=1.4, ms=5.5, 
             markerfacecolor=C4, markeredgecolor="white", 
             markeredgewidth=0.7, zorder=5, 
             label="Social welfare $SW$")
    ax1r.set_ylabel(r"Social welfare $SW$", color=C4, 
                   labelpad=5, fontsize=FS_LABEL)
    ax1r.tick_params(axis="y", colors=C4, labelsize=FS_TICK, 
                    direction="in", length=3.5, width=0.75)
    ax1r.spines["right"].set_color(C4)
    ax1r.spines["right"].set_linewidth(0.75)
    ax1r.spines["top"].set_visible(False)
    ax1r.spines["left"].set_visible(False)
    ax1r.spines["bottom"].set_visible(False)
    
    poa = eq_d["SW"] / eq_c["SW"]
    gap = eq_c["rho"] - eq_d["rho"]
    ax1.set_title(
        f"(a) Scenario Comparison\n"
        rf"$\Delta\rho={gap:.3f}$; PoA $= {poa:.4f}$", 
        fontsize=FS_TITLE, pad=4)
    
    # ── Panel (b): 福利分解瀑布图 ──────────────────────────────
    dCS = eq_d["CS"] - eq_nt["CS"]
    dPim = eq_d["pi_m"] - eq_nt["pi_m"]
    dPir = eq_d["pi_r"] - eq_nt["pi_r"]
    dTax = eq_d["tax_rev"]
    dEnv = eq_nt["env_dam"] - eq_d["env_dam"]
    dSW = eq_d["SW"] - eq_nt["SW"]
    
    vals = [dCS, dPim, dPir, dTax, dEnv, dSW]
    labs = [r"$\Delta CS$", r"$\Delta\pi_M$", r"$\Delta\pi_R$", 
            r"$\tau E$", r"$-\Delta D$", r"$\Delta SW$"]
    n_bars = len(vals)
    
    # 计算瀑布底部
    bottoms = []
    run = 0.0
    for v in vals[:-1]:
        bottoms.append(run if v >= 0 else run + v)
        run += v
    bottoms.append(0)  # Net ΔSW 从 0 开始
    
    bar_colors = [C3 if v >= 0 else C2 for v in vals]
    bar_colors[-1] = C3 if vals[-1] >= 0 else C2
    
    bars2 = ax2.bar(np.arange(n_bars), np.abs(vals), 0.55, 
                   bottom=bottoms, color=bar_colors, alpha=0.82, 
                   edgecolor="white", lw=0.6, zorder=3)
    ax2.axhline(0, color=CB, lw=0.75, zorder=5)
    
    # 数值标注（正负分别放置）
    for i, (bar, v) in enumerate(zip(bars2, vals)):
        ypos = (bar.get_y() + bar.get_height() + 
               (120 if v >= 0 else -120))
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos, 
                f"{v:+.0f}", ha="center", 
                va="bottom" if v >= 0 else "top", 
                fontsize=FS_ANNOT, color=bar_colors[i], 
                fontweight="bold")
    
    # 累计连接线（灰色虚线）
    run = 0.0
    for i in range(n_bars - 2):
        run += vals[i]
        ax2.plot([i + 0.275, i + 0.725], [run, run], 
                color="#888888", lw=0.6, ls="--", zorder=4)
    
    ax2.set_xticks(np.arange(n_bars))
    ax2.set_xticklabels(labs, fontsize=FS_TICK + 0.2)
    ax2.set_ylabel(r"Welfare change ($\Delta$)", 
                  labelpad=5, fontsize=FS_LABEL)
    ax2.set_title("(b) Welfare Decomposition\n"
                  r"(Decentralized $\tau^*$ vs. No-Tax)", fontsize=FS_TITLE, pad=4)
    ax2.tick_params(which="major", direction="in", top=False, right=False, 
                   length=3.5, width=0.75)
    ax2.tick_params(which="minor", bottom=False, top=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.20, lw=0.4, zorder=0)
    
    # 图例 patch
    p_pos = mpatches.Patch(color=C3, alpha=0.82, label="Gain (+)")
    p_neg = mpatches.Patch(color=C2, alpha=0.82, label="Loss (−)")
    ax2.legend(handles=[p_pos, p_neg], loc="lower right", 
              fontsize=FS_LEGEND, framealpha=1.0, 
              edgecolor="#CCCCCC")
    
    fig.suptitle(
        "Figure 3. Scenario Comparison and Welfare Decomposition", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=1.000)
    
    _save(fig, "CLSC_Fig3_Benchmark_Decomposition")

# ══════════════════════════════════════════════════════════════════
# ██ Figure 4 — Revenue-Sharing Contract Analysis
# ══════════════════════════════════════════════════════════════════

def fig4(eq_d, eq_c, phi_data):
    """
    Figure 4. Revenue-Sharing Contract: Profit Distribution and Social Welfare.
    Panel (a): Manufacturer and retailer profits vs. revenue-sharing ratio φ.
    Panel (b): Social welfare SW^{RS}(φ) with decentralized and VIF benchmarks.
    """
    W = MS_COL1
    H = MS_H_MD + 0.1
    fig, axes = plt.subplots(1, 2, figsize=(W, H), 
                            gridspec_kw={"wspace": 0.42, "left": 0.09, 
                                        "right": 0.96, "top": 0.87, 
                                        "bottom": 0.16})
    ax1, ax2 = axes
    phis = phi_data["phis"]
    pms = phi_data["pi_m"]
    prs = phi_data["pi_r"]
    sws = phi_data["sw"]
    
    # ── Panel (a): 利润分布 ────────────────────────────────────
    ax1.plot(phis, pms, color=C1, lw=1.8, zorder=4, 
            label=r"Manufacturer $\pi_m^{RS}(\phi)$")
    ax1.plot(phis, prs, color=C4, lw=1.8, ls=LS_DASH, zorder=4, 
            label=r"Retailer $\pi_r^{RS}(\phi)$")
    ax1.fill_between(phis, 0, pms, where=pms >= 0, 
                    color=C1, alpha=0.08, zorder=1)
    ax1.fill_between(phis, 0, prs, where=prs >= 0, 
                    color=C4, alpha=0.08, zorder=1)
    
    # 零轴
    ax1.axhline(0, color=CB, lw=0.75, zorder=5)
    
    # 找 retailer 为 0 的边界 φ
    if np.any(prs < 0):
        idx_zero = np.where(np.diff(np.sign(prs)))[0]
        if len(idx_zero) > 0:
            phi_zero = phis[idx_zero[0]]
            _vline(ax1, phi_zero, color=C4, lw=0.8, ls=LS_DOT)
            ax1.text(phi_zero + 0.02, prs.min() * 0.5, 
                    rf"$\phi={phi_zero:.2f}$", 
                    fontsize=FS_ANNOT, color=C4)
    
    _style_ax(ax1, xlabel=r"Revenue-sharing ratio $\phi$", 
             ylabel="Profit", title="(a) Profit Distribution")
    ax1.legend(loc="center right", fontsize=FS_LEGEND, 
              framealpha=1.0, edgecolor="#CCCCCC")
    ax1.set_xlim(0, 1)
    
    # ── Panel (b): 社会福利 ────────────────────────────────────
    ax2.plot(phis, sws, color=C5, lw=1.8, zorder=4, 
            label=r"$SW^{RS}(\phi)$: Revenue sharing")
    
    # 参考线
    _hline(ax2, eq_d["SW"], color=C1, lw=0.9, ls=LS_DASH, alpha=0.85)
    ax2.text(0.02, eq_d["SW"] + 300, 
            rf"$SW^*={eq_d['SW']:.0f}$" + "\n(Decentralized)", 
            fontsize=FS_ANNOT, color=C1, style="italic", linespacing=1.3)
    
    _hline(ax2, eq_c["SW"], color=C3, lw=0.9, ls=LS_DASD, alpha=0.90)
    ax2.text(0.02, eq_c["SW"] + 300, 
            rf"$SW^C={eq_c['SW']:.0f}$" + "\n(VIF)", 
            fontsize=FS_ANNOT, color=C3, style="italic", linespacing=1.3)
    
    # 最优 φ
    idx_max = int(np.argmax(sws))
    phi_opt = phis[idx_max]
    sw_opt = sws[idx_max]
    _dot(ax2, phi_opt, sw_opt, color=C5, s=30, marker="*")
    _vline(ax2, phi_opt, color=C5, lw=0.8, ls=LS_DOT)
    ax2.text(phi_opt + 0.02, sw_opt - 1200, 
            rf"$\phi^*={phi_opt:.2f}$", 
            fontsize=FS_ANNOT, color=C5, style="italic")
    
    _style_ax(ax2, xlabel=r"Revenue-sharing ratio $\phi$", 
             ylabel=r"Social welfare $SW^{RS}$", 
             title="(b) Social Welfare")
    ax2.legend(loc="upper right", fontsize=FS_LEGEND, 
              framealpha=1.0, edgecolor="#CCCCCC")
    ax2.set_xlim(0, 1)
    
    fig.suptitle(
        "Figure 4. Revenue-Sharing Contract Analysis", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=0.975)
    
    _save(fig, "CLSC_Fig4_Coordination")

# ══════════════════════════════════════════════════════════════════
# ██ Figure 5 — Comparative Statics (η and γ)
# ══════════════════════════════════════════════════════════════════

def fig5(eta_data, gamma_data):
    """
    Figure 5. Comparative Statics.
    Panels (a)(b): Effects of environmental damage parameter η.
    Panels (c)(d): Effects of green preference parameter γ.
    """
    W = MS_COL1
    H = MS_H_LG
    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(2, 2, figure=fig, wspace=0.48, hspace=0.58, 
                 left=0.09, right=0.95, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    
    # ── (a): η → τ* and E* ───────────────────────────────────
    ax = axes[0]
    ax.plot(eta_data["etas"], eta_data["tau_s"], 
           color=C2, lw=1.8, zorder=4, label=r"$\tau^*(\eta)$")
    ax.fill_between(eta_data["etas"], 0, eta_data["tau_s"], 
                   color=C2, alpha=0.10, zorder=1)
    axr = ax.twinx()
    axr.plot(eta_data["etas"], eta_data["E_s"], 
            color=C3, lw=1.6, ls=LS_DASH, zorder=3, 
            label=r"$E^*(\eta)$")
    _twin_style(axr, ylabel=r"Optimal emissions $E^*$", color=C3)
    _style_ax(ax, xlabel=r"Environmental damage $\eta$", 
             ylabel=r"Optimal tax $\tau^*$", 
             title=r"(a) Effect of $\eta$ on $\tau^*$ and $E^*$")
    ax.tick_params(axis="y", colors=C2, labelsize=FS_TICK)
    ax.spines["left"].set_color(C2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", 
             fontsize=FS_LEGEND, framealpha=1.0, 
             edgecolor="#CCCCCC")
    
    # ── (b): η → SW* ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(eta_data["etas"], eta_data["SW_s"], 
           color=C1, lw=1.8, zorder=4, label=r"$SW^*(\eta)$")
    ax.fill_between(eta_data["etas"], eta_data["SW_s"].min(), 
                   eta_data["SW_s"], color=C1, alpha=0.10, zorder=1)
    
    # 标注斜率方向
    mid = len(eta_data["etas"]) // 2
    slope = np.gradient(eta_data["SW_s"], eta_data["etas"])[mid]
    sign_str = (r"$\partial SW^*/\partial\eta < 0$" if slope < 0 
                else r"$\partial SW^*/\partial\eta > 0$")
    ax.text(0.97, 0.95, sign_str, transform=ax.transAxes, 
           ha="right", va="top", fontsize=FS_ANNOT, color=C1, 
           style="italic", 
           bbox=dict(boxstyle="round,pad=0.25", fc="white", 
                    ec="#CCCCCC", alpha=0.95, lw=0.6))
    
    _style_ax(ax, xlabel=r"Environmental damage $\eta$", 
             ylabel=r"Social welfare $SW^*$", 
             title=r"(b) Effect of $\eta$ on $SW^*$")
    ax.legend(loc="upper right", fontsize=FS_LEGEND, 
             framealpha=1.0, edgecolor="#CCCCCC")
    
    # ── (c): γ → τ* and ρ* ───────────────────────────────────
    ax = axes[2]
    ax.plot(gamma_data["gammas"], gamma_data["tau_s"], 
           color=C2, lw=1.8, zorder=4, label=r"$\tau^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], 
                   color=C2, alpha=0.10, zorder=1)
    axr2 = ax.twinx()
    axr2.plot(gamma_data["gammas"], gamma_data["rho_s"], 
             color=C1, lw=1.6, ls=LS_DASH, zorder=3, 
             label=r"$\rho^*(\gamma)$")
    _twin_style(axr2, ylabel=r"Recycling rate $\rho^*$", color=C1)
    _style_ax(ax, xlabel=r"Green preference $\gamma$", 
             ylabel=r"Optimal tax $\tau^*$", 
             title=r"(c) Effect of $\gamma$ on $\tau^*$ and $\rho^*$")
    ax.tick_params(axis="y", colors=C2, labelsize=FS_TICK)
    ax.spines["left"].set_color(C2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", 
             fontsize=FS_LEGEND, framealpha=1.0, 
             edgecolor="#CCCCCC")
    
    # ── (d): γ → D* and ρ* ───────────────────────────────────
    ax = axes[3]
    ax.plot(gamma_data["gammas"], gamma_data["D_s"], 
           color=C4, lw=1.8, zorder=4, label=r"$D^*(\gamma)$")
    ax.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), 
                   gamma_data["D_s"], color=C4, alpha=0.12, zorder=1)
    axr3 = ax.twinx()
    axr3.plot(gamma_data["gammas"], gamma_data["rho_s"], 
             color=C5, lw=1.6, ls=LS_DASH, zorder=3, 
             label=r"$\rho^*(\gamma)$")
    _twin_style(axr3, ylabel=r"Recycling rate $\rho^*$", color=C5)
    _style_ax(ax, xlabel=r"Green preference $\gamma$", 
             ylabel=r"Market demand $D^*$", 
             title=r"(d) Effect of $\gamma$ on $D^*$ and $\rho^*$")
    ax.tick_params(axis="y", colors=C4, labelsize=FS_TICK)
    ax.spines["left"].set_color(C4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", 
             fontsize=FS_LEGEND, framealpha=1.0, 
             edgecolor="#CCCCCC")
    
    fig.suptitle(
        r"Figure 5. Comparative Statics: Effects of $\eta$ and $\gamma$", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=0.975)
    
    _save(fig, "CLSC_Fig5_ComparativeStatics")

# ══════════════════════════════════════════════════════════════════
# ██ Figure 6 — Comprehensive Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════

def fig6_sensitivity_analysis(eta_data, gamma_data):
    """
    Figure 6. Sensitivity Analysis.
    Panels (a)(b): η effects on τ*, ρ*, E*, SW*.
    Panels (c)(d): γ effects on τ*, ρ*, D*, SW*.
    """
    W = MS_COL1 + 0.2
    H = MS_H_LG + 0.3
    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(2, 2, figure=fig, wspace=0.55, hspace=0.60, 
                 left=0.09, right=0.94, top=0.93, bottom=0.07)
    
    # ── (a): η → τ* / ρ* ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eta_data["etas"], eta_data["tau_s"], 
            color=C2, lw=1.8, zorder=4, label=r"$\tau^*$")
    ax1.fill_between(eta_data["etas"], 0, eta_data["tau_s"], 
                    color=C2, alpha=0.10, zorder=1)
    ax1.tick_params(axis="y", colors=C2, labelsize=FS_TICK)
    ax1.spines["left"].set_color(C2)
    ax1_t = ax1.twinx()
    ax1_t.plot(eta_data["etas"], eta_data["rho_s"], 
              color=C1, lw=1.6, ls=LS_DASH, zorder=3, 
              label=r"$\rho^*$")
    _twin_style(ax1_t, ylabel=r"Recycling rate $\rho^*$", color=C1)
    _style_ax(ax1, xlabel=r"Environmental damage $\eta$", 
             ylabel=r"Optimal tax $\tau^*$", 
             title=r"(a) $\eta$: Optimal Tax and Recycling")
    ax1.tick_params(axis="y", colors=C2)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_t.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", 
              fontsize=FS_LEGEND, framealpha=1.0, 
              edgecolor="#CCCCCC")
    
    # ── (b): η → E* / SW* ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(eta_data["etas"], eta_data["E_s"], 
            color=C4, lw=1.8, zorder=4, label=r"$E^*$")
    ax2.fill_between(eta_data["etas"], eta_data["E_s"].min(), 
                    eta_data["E_s"], color=C4, alpha=0.10, zorder=1)
    ax2.tick_params(axis="y", colors=C4, labelsize=FS_TICK)
    ax2.spines["left"].set_color(C4)
    ax2_t = ax2.twinx()
    ax2_t.plot(eta_data["etas"], eta_data["SW_s"], 
              color=C3, lw=1.6, ls=LS_DASH, zorder=3, 
              label=r"$SW^*$")
    _twin_style(ax2_t, ylabel=r"Social welfare $SW^*$", color=C3)
    _style_ax(ax2, xlabel=r"Environmental damage $\eta$", 
             ylabel=r"Total emissions $E^*$", 
             title=r"(b) $\eta$: Emissions and Social Welfare")
    ax2.tick_params(axis="y", colors=C4)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", 
              fontsize=FS_LEGEND, framealpha=1.0, 
              edgecolor="#CCCCCC")
    
    # ── (c): γ → τ* / ρ* ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(gamma_data["gammas"], gamma_data["tau_s"], 
            color=C2, lw=1.8, zorder=4, label=r"$\tau^*$")
    ax3.fill_between(gamma_data["gammas"], 0, gamma_data["tau_s"], 
                    color=C2, alpha=0.10, zorder=1)
    ax3.tick_params(axis="y", colors=C2, labelsize=FS_TICK)
    ax3.spines["left"].set_color(C2)
    ax3_t = ax3.twinx()
    ax3_t.plot(gamma_data["gammas"], gamma_data["rho_s"], 
              color=C1, lw=1.6, ls=LS_DASH, zorder=3, 
              label=r"$\rho^*$")
    _twin_style(ax3_t, ylabel=r"Recycling rate $\rho^*$", color=C1)
    _style_ax(ax3, xlabel=r"Green preference $\gamma$", 
             ylabel=r"Optimal tax $\tau^*$", 
             title=r"(c) $\gamma$: Optimal Tax and Recycling")
    ax3.tick_params(axis="y", colors=C2)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_t.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="lower right", 
              fontsize=FS_LEGEND, framealpha=1.0, 
              edgecolor="#CCCCCC")
    
    # ── (d): γ → D* / SW* ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(gamma_data["gammas"], gamma_data["D_s"], 
            color=C4, lw=1.8, zorder=4, label=r"$D^*$")
    ax4.fill_between(gamma_data["gammas"], gamma_data["D_s"].min(), 
                    gamma_data["D_s"], color=C4, alpha=0.12, zorder=1)
    ax4.tick_params(axis="y", colors=C4, labelsize=FS_TICK)
    ax4.spines["left"].set_color(C4)
    ax4_t = ax4.twinx()
    ax4_t.plot(gamma_data["gammas"], gamma_data["SW_s"], 
              color=C3, lw=1.6, ls=LS_DASH, zorder=3, 
              label=r"$SW^*$")
    _twin_style(ax4_t, ylabel=r"Social welfare $SW^*$", color=C3)
    _style_ax(ax4, xlabel=r"Green preference $\gamma$", 
             ylabel=r"Market demand $D^*$", 
             title=r"(d) $\gamma$: Market Demand and Social Welfare")
    ax4.tick_params(axis="y", colors=C4)
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4_t.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc="upper left", 
              fontsize=FS_LEGEND, framealpha=1.0, 
              edgecolor="#CCCCCC")
    
    fig.suptitle(
        r"Figure 6. Sensitivity Analysis: Effects of $\eta$ and $\gamma$", 
        fontsize=FS_SUPER, fontstyle="italic", x=0.52, y=0.975)
    
    _save(fig, "CLSC_Fig6_Sensitivity_Analysis")

# ══════════════════════════════════════════════════════════════════
# ██ 控制台输出函数
# ══════════════════════════════════════════════════════════════════

def print_header():
    print("\n" + "═" * 85)
    print("║" + " " * 83 + "║")
    print("║" + "CLSC Stackelberg Game Model — CORRECTED VERSION".center(83) + "║")
    print("║" + "闭环供应链碳税政策 Stackelberg 博弈模型（修正版）".center(83) + "║")
    print("║" + " " * 83 + "║")
    print("═" * 85)
    print(f"║  运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("║  版本：Corrected v3.0 | 图形：MS-Enhanced v4.0")
    print("║  修正：回收率一阶条件（原论文忽略γ项）")
    print("║  图形规范：Management Science / Operations Research 期刊标准")
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
    print(f"  【价格无效率】PoA = SW*/SW^C       = {poa:>10.4f} (越接近 1 越优)")
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

# ══════════════════════════════════════════════════════════════════
# ██ 主程序
# ══════════════════════════════════════════════════════════════════

def _timed(label, fn):
    """执行 fn() 并在同一行打印标签与耗时"""
    print(f" → {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"完成 ({elapsed:.2f}s)")
    return result

def main():
    total_start = time.perf_counter()
    
    print_header()
    c = Cfg()
    print_config(c)
    
    print("🔬 [1/6] 求解均衡点...")
    eq_nt = _timed("无碳税均衡", lambda: dec_eq(c, tau=0.0))
    eq_d = _timed("分散决策均衡", lambda: dec_eq(c))
    eq_c = _timed("VIF 集中式均衡", lambda: vif_eq(c))
    print()
    
    print("🔢 [2/6] 均衡结果汇总...")
    print_equilibrium(eq_nt, eq_d, eq_c, time.perf_counter() - total_start, "数值优化ρ")
    
    print("📈 [3/6] 开始计算敏感性分析数据...")
    sens_start = time.perf_counter()
    sens = _timed("碳税扫描 (τ sweep, n=150)", lambda: tau_sweep(c, n=150))
    eta_d = _timed("环境损害敏感性 (η sweep, n=60)", lambda: eta_sweep(c, n=60))
    gam_d = _timed("绿色偏好敏感性 (γ sweep, n=60)", lambda: gamma_sweep(c, n=60))
    phi_d = _timed("收益共享契约分析 (φ sweep, n=80)", 
                   lambda: phi_sweep(c, eq_d["tau"], n=80))
    sens_elapsed = time.perf_counter() - sens_start
    print(f"✅ 敏感性分析完成 (总耗时：{sens_elapsed:.2f} 秒)\n")
    
    print("🎨 [4/6] 开始渲染 6 个 Management Science 期刊级图形 (600 DPI)...")
    fig_start = time.perf_counter()
    _timed("生成图 1: 定价与回收率动态", 
           lambda: fig1(eq_nt, eq_d, eq_c, sens))
    _timed("生成图 2: 排放与社会福利", 
           lambda: fig2(eq_nt, eq_d, eq_c, sens))
    _timed("生成图 3: 场景对比与福利分解", 
           lambda: fig3(eq_nt, eq_d, eq_c))
    _timed("生成图 4: 收益共享契约分析", 
           lambda: fig4(eq_d, eq_c, phi_d))
    _timed("生成图 5: 比较静态分析", 
           lambda: fig5(eta_d, gam_d))
    _timed("生成图 6: 敏感性分析综合图", 
           lambda: fig6_sensitivity_analysis(eta_d, gam_d))
    fig_elapsed = time.perf_counter() - fig_start
    print(f"✅ 所有图形生成完成 (总耗时：{fig_elapsed:.2f} 秒)")
    print("   📁 图形文件已保存到当前目录 (PDF/PNG 格式，600 DPI)")
    print("   📊 图形规范：Management Science 期刊标准")
    print("   • 双栏宽度 3.46\"/单栏宽度 7.09\"  • 字体 TrueType 嵌入 (PDF type 42)")
    print("   • 色盲友好配色 (Wong 2011)        • 四边 inward tick")
    print("   • 精细 minor tick / spines        • 无右/上 spines（柱图），四边（折线图）\n")
    
    print_summary_table(eq_nt, eq_d, eq_c)
    print_key_metrics(eq_nt, eq_d, eq_c)
    
    total_elapsed = time.perf_counter() - total_start
    print("═" * 85)
    print(f"🏁 模型运行完成！总耗时：{total_elapsed:.2f} 秒".center(85))
    print("═" * 85)
    print("\n📝 修正说明:")
    print("   • 使用数值优化直接求解最优 ρ，避免解析近似误差")
    print("   • 回收率一阶条件已修正（包含 γ 项的间接效应）")
    print("\n💡 图形增强说明（MS-Enhanced v4.0）:")
    print("   • 严格符合 Management Science 期刊图形规范（尺寸、字号、DPI）")
    print("   • 色盲友好配色（Wong 2011, Nature Methods）")
    print("   • 四边 inward tick（MS/OR 标准）")
    print("   • PDF type 42 字体嵌入（Elsevier/APA 投稿要求）")
    print("   • 瀑布图替代原简单柱图（Fig. 3b 福利分解更直观）")
    print("   • 背景区域着色标注最优碳税左右区间（Fig. 1b）")
    print("   • 所有图形包含完整子图说明供图注使用")
    print("═" * 85 + "\n")

if __name__ == "__main__":
    main()
