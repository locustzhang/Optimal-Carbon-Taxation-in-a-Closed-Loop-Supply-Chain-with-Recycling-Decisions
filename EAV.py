import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
import matplotlib.patches as mpatches

# ==============================================================
# Global Academic Plotting Style Configuration (NPG Nature Style)
# ==============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",  # Computer Modern Math
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.5,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.constrained_layout.use": True,
    "savefig.dpi": 600,  # 600 DPI for publication quality
    "savefig.bbox": "tight"
})

# Nature Publishing Group (NPG) Color Palette
C_RED = "#E64B35"  # NPG Red
C_BLUE = "#4DBBD5"  # NPG Light Blue
C_GREEN = "#00A087"  # NPG Green
C_DARK = "#3C5488"  # NPG Dark Blue
C_ORANG = "#F39B7F"  # NPG Orange
C_GREY = "#7E6148"  # NPG Brown/Grey


# ==============================================================
# Model Configuration
# ==============================================================
@dataclass
class ModelConfig:
    a: float = 500
    b: float = 1.3
    gamma: float = 10
    c_m: float = 200
    c_r: float = 170
    k: float = 12000
    e_m: float = 0.8
    e_r: float = 0.2
    eta: float = 3.0
    g: float = 0.5
    tau_max: float = 150
    tol: float = 1e-8


# ==============================================================
# Core Functions
# ==============================================================
def demand_fn(cfg: ModelConfig, p: float) -> float:
    return max(cfg.a - cfg.b * p + cfg.gamma * cfg.g, 0.0)


def emission_fn(cfg: ModelConfig, D: float, rho: float) -> float:
    return cfg.e_m * (1.0 - rho) * D + cfg.e_r * rho * D


def retailer_best_response(cfg: ModelConfig, w: float) -> float:
    return (cfg.a + cfg.gamma * cfg.g + cfg.b * w) / (2.0 * cfg.b)


def manufacturer_profit(cfg: ModelConfig, w: float, rho: float, tau: float) -> float:
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    return (w - cfg.c_m) * D + (cfg.c_m - cfg.c_r) * rho * D - tau * E - cfg.k * rho ** 2


def optimal_recycling_rate(cfg: ModelConfig, w: float, tau: float) -> float:
    res = minimize_scalar(lambda r: -manufacturer_profit(cfg, w, r, tau), bounds=(0.0, 1.0), method="bounded",
                          options={'xatol': cfg.tol})
    return res.x


def optimal_wholesale_price(cfg: ModelConfig, tau: float) -> tuple:
    def neg_profit(w):
        return -manufacturer_profit(cfg, w, optimal_recycling_rate(cfg, w, tau), tau)

    res = minimize_scalar(neg_profit, bounds=(cfg.c_m, cfg.a / cfg.b), method="bounded", options={'xatol': cfg.tol})
    return res.x, optimal_recycling_rate(cfg, res.x, tau)


def decentralized_equilibrium(cfg: ModelConfig, tau=None) -> dict:
    if tau is None:
        def neg_welfare(t):
            w, r = optimal_wholesale_price(cfg, t)
            p = retailer_best_response(cfg, w)
            D = demand_fn(cfg, p)
            if D <= 0: return 1e10
            E = emission_fn(cfg, D, r)
            return -((cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2 * cfg.b) + manufacturer_profit(cfg, w, r, t) + (
                        p - w) * D + t * E - cfg.eta * E ** 2)

        tau = minimize_scalar(neg_welfare, bounds=(0.0, cfg.tau_max), method="bounded", options={'xatol': cfg.tol}).x

    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    E = emission_fn(cfg, D, rho)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    env_dam = cfg.eta * E ** 2
    tax_rev = tau * E
    SW = CS + pi_m + pi_r + tax_rev - env_dam
    return dict(tau=tau, w=w, p=p, rho=rho, D=D, E=E, pi_m=pi_m, pi_r=pi_r, CS=CS, env_dam=env_dam, tax_rev=tax_rev,
                SW=SW)


def centralized_equilibrium(cfg: ModelConfig) -> dict:
    def vif_profit(p, rho, t):
        D = demand_fn(cfg, p)
        return (p - cfg.c_m) * D + (cfg.c_m - cfg.c_r) * rho * D - cfg.k * rho ** 2 - t * emission_fn(cfg, D, rho)

    def vif_opt(t):
        def neg_vif(p):
            r = minimize_scalar(lambda r: -vif_profit(p, r, t), bounds=(0.0, 1.0), method="bounded",
                                options={'xatol': cfg.tol}).x
            D = demand_fn(cfg, p)
            if D <= 0: return 1e10
            E = emission_fn(cfg, D, r)
            CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2 * cfg.b)
            return -(CS + vif_profit(p, r, t) + t * E - cfg.eta * E ** 2)

        res_p = minimize_scalar(neg_vif, bounds=(cfg.c_m, cfg.a / cfg.b), method="bounded",
                                options={'xatol': cfg.tol}).x
        res_r = minimize_scalar(lambda r: -vif_profit(res_p, r, t), bounds=(0.0, 1.0), method="bounded",
                                options={'xatol': cfg.tol}).x
        return res_p, res_r

    def neg_sw(t):
        p, r = vif_opt(t)
        D = demand_fn(cfg, p)
        if D <= 0: return 1e10
        E = emission_fn(cfg, D, r)
        CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2 * cfg.b)
        return -(CS + vif_profit(p, r, t) + t * E - cfg.eta * E ** 2)

    tau_c = minimize_scalar(neg_sw, bounds=(0.0, cfg.tau_max), method="bounded", options={'xatol': cfg.tol}).x
    p, rho = vif_opt(tau_c)
    D = demand_fn(cfg, p)
    E = emission_fn(cfg, D, rho)
    pi_vif = vif_profit(p, rho, tau_c)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    env_dam = cfg.eta * E ** 2
    SW = CS + pi_vif + tau_c * E - env_dam
    return dict(tau=tau_c, p=p, rho=rho, D=D, E=E, pi_vif=pi_vif, CS=CS, env_dam=env_dam, tax_rev=tau_c * E, SW=SW)


# ==============================================================
# Analytical Engines
# ==============================================================
def standard_sensitivity(cfg: ModelConfig) -> dict:
    taus = np.linspace(0, cfg.tau_max, 100)
    res = {'taus': taus, 'rho': [], 'E': [], 'SW': [], 'p': [], 'w': []}
    for t in taus:
        eq = decentralized_equilibrium(cfg, t)
        res['w'].append(eq['w']);
        res['p'].append(eq['p']);
        res['rho'].append(eq['rho'])
        res['E'].append(eq['E']);
        res['SW'].append(eq['SW'])
    return {k: np.array(v) for k, v in res.items()}


def strategic_sensitivity(cfg: ModelConfig) -> dict:
    gammas = np.linspace(5, 25, 40)
    res = {'gammas': gammas, 'tau_stars': [], 'rho_stars': []}
    orig_gamma = cfg.gamma
    for g in gammas:
        cfg.gamma = g
        eq = decentralized_equilibrium(cfg)
        res['tau_stars'].append(eq['tau'])
        res['rho_stars'].append(eq['rho'])
    cfg.gamma = orig_gamma
    return {k: np.array(v) for k, v in res.items()}


# ==============================================================
# Advanced Visualization (Nature-Level)
# ==============================================================
def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_set_1(eq_nt, eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1A: Pricing Dynamics & Margin Squeeze
    ax = axes[0]
    ax.plot(sens['taus'], sens['p'], color=C_DARK, lw=2.5, label='Retail Price ($p$)')
    ax.plot(sens['taus'], sens['w'], color=C_BLUE, lw=2.5, ls='--', label='Wholesale Price ($w$)')
    # Shade the double marginalization gap
    ax.fill_between(sens['taus'], sens['w'], sens['p'], color=C_BLUE, alpha=0.1, label='Retailer Margin')
    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=2)
    ax.scatter([eq_d['tau'], eq_d['tau']], [eq_d['p'], eq_d['w']], color=C_RED, s=70, zorder=5)
    ax.annotate('Optimal Tax $\\tau^*$', xy=(eq_d['tau'], eq_d['w'] - 5), xytext=(eq_d['tau'] + 15, eq_d['w'] - 15),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=11, fontweight='bold')
    ax.set(xlabel='Carbon Tax Level ($\\tau$)', ylabel='Price', title='A. Supply Chain Pricing Dynamics')
    ax.legend(frameon=False);
    clean_ax(ax)

    # 1B: Recycling Rate & Policy Gap
    ax = axes[1]
    ax.plot(sens['taus'], sens['rho'], color=C_DARK, lw=2.5, label='Decentralized Response ($\\rho^*$)')
    ax.axhline(eq_c['rho'], color=C_GREEN, ls='-.', lw=2.5,
               label=f"Centralized Benchmark ($\\rho^C={eq_c['rho']:.3f}$)")

    # Highlight Gap at optimal tax
    ax.plot([eq_d['tau'], eq_d['tau']], [eq_d['rho'], eq_c['rho']], color=C_RED, lw=2.5, ls='-')
    ax.scatter([eq_d['tau']], [eq_d['rho']], color=C_RED, s=70, zorder=5, label='Decentralized Equilibrium')
    ax.text(eq_d['tau'] + 3, (eq_d['rho'] + eq_c['rho']) / 2,
            f'Coordination Gap\n$\\Delta\\rho = {eq_c["rho"] - eq_d["rho"]:.3f}$',
            color=C_RED, fontweight='bold', va='center')
    ax.set(xlabel='Carbon Tax Level ($\\tau$)', ylabel='Recycling Rate ($\\rho$)',
           title='B. Recycling Rate & Coordination Gap')
    ax.legend(frameon=False, loc='lower right');
    clean_ax(ax)

    plt.savefig('CLSC_Set1_Pricing_Recycling.png')
    plt.savefig('CLSC_Set1_Pricing_Recycling.pdf')
    plt.show()


def plot_set_2(eq_nt, eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 2A: Emission Trajectory & Reduction Area
    ax = axes[0]
    ax.plot(sens['taus'], sens['E'], color=C_ORANG, lw=2.5, label='Actual Emissions ($E$)')
    ax.axhline(eq_nt['E'], color=C_GREY, ls='--', lw=1.5, label='No-Tax Baseline Emissions')

    # Shade emission reduction area
    ax.fill_between(sens['taus'], sens['E'], eq_nt['E'], where=(sens['E'] < eq_nt['E']),
                    color=C_GREEN, alpha=0.15, label='Environmental Gain (Emission Reduction)')

    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=2)
    ax.scatter([eq_d['tau']], [eq_d['E']], color=C_RED, s=70, zorder=5)
    ax.set(xlabel='Carbon Tax Level ($\\tau$)', ylabel='Total Carbon Emissions ($E$)',
           title='A. Environmental Impact Control')
    ax.legend(frameon=False);
    clean_ax(ax)

    # 2B: Welfare Maximization & Policy Feasibility Range
    ax = axes[1]
    ax.plot(sens['taus'], sens['SW'], color=C_DARK, lw=2.5, label='Social Welfare ($SW$)')
    ax.axhline(eq_nt['SW'], color=C_GREY, ls='--', lw=1.5, label='No-Tax Baseline Welfare')

    # Highlight policy feasibility zone (Where SW > Base SW)
    feasible_mask = sens['SW'] > eq_nt['SW']
    ax.fill_between(sens['taus'], 0, 1, where=feasible_mask, color=C_BLUE, alpha=0.1,
                    transform=ax.get_xaxis_transform(), label='Effective Policy Zone ($SW > SW_0$)')

    idx = np.nanargmax(sens['SW'])
    ax.scatter([sens['taus'][idx]], [sens['SW'][idx]], color=C_RED, s=90, edgecolor='white', lw=1.5, zorder=5)
    ax.annotate('Welfare Maximum', xy=(sens['taus'][idx], sens['SW'][idx]),
                xytext=(sens['taus'][idx] + 15, sens['SW'][idx]),
                arrowprops=dict(facecolor=C_RED, arrowstyle='->'), fontsize=11, color=C_RED, fontweight='bold')

    ax.set(xlabel='Carbon Tax Level ($\\tau$)', ylabel='Social Welfare ($SW$)', title='B. Social Welfare Optimization')
    ax.set_ylim(bottom=min(sens['SW']) * 0.99, top=max(sens['SW']) * 1.005)
    ax.legend(frameon=False, loc='lower center');
    clean_ax(ax)

    plt.savefig('CLSC_Set2_Environment_Welfare.png')
    plt.savefig('CLSC_Set2_Environment_Welfare.pdf')
    plt.show()


def plot_set_3(cfg, eq_nt, eq_d, eq_c, strat_sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 3A: Strategic Benchmarking (Academic Hatching Bar Chart)
    ax = axes[0]
    labels = ['No Tax ($\\tau=0$)', 'Decentralized ($\\tau^*$)', 'Centralized ($\\tau^C$)']
    rho_vals = [eq_nt['rho'], eq_d['rho'], eq_c['rho']]
    sw_vals = [eq_nt['SW'], eq_d['SW'], eq_c['SW']]

    x = np.arange(3)
    width = 0.35
    # Add hatching (///) for academic black & white print compatibility
    bars1 = ax.bar(x - width / 2, rho_vals, width, label='Recycling Rate ($\\rho$)',
                   color=C_BLUE, edgecolor='black', hatch='//', alpha=0.85)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, sw_vals, width, label='Social Welfare ($SW$)',
                    color=C_GREEN, edgecolor='black', hatch='\\\\', alpha=0.85)

    for bar in bars1: ax.text(bar.get_x() + width / 2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center',
                              fontweight='bold', fontsize=10)
    for bar in bars2: ax2.text(bar.get_x() + width / 2, bar.get_height() + (max(sw_vals) * 0.01),
                               f'{bar.get_height():.0f}', ha='center', fontsize=10)

    ax.set_xticks(x);
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Recycling Rate ($\\rho$)', color=C_DARK, fontweight='bold')
    ax2.set_ylabel('Social Welfare ($SW$)', color=C_GREEN, fontweight='bold')
    ax.set_title('A. System Efficiency & Coordination Gap', loc='left')
    ax.spines['top'].set_visible(False);
    ax2.spines['top'].set_visible(False)

    # Unified legend
    lines, labels = ax.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)

    # 3B: Strategic Sensitivity (Consumer Green Preference)
    ax = axes[1]
    gammas = strat_sens['gammas']
    l1 = ax.plot(gammas, strat_sens['tau_stars'], color=C_RED, lw=2.5, marker='o', markevery=5,
                 label='Optimal Tax ($\\tau^*$)')
    ax.set_xlabel('Consumer Green Preference Factor ($\\gamma$)');
    ax.set_ylabel('Optimal Carbon Tax ($\\tau^*$)', color=C_RED, fontweight='bold')

    ax2 = ax.twinx()
    l2 = ax2.plot(gammas, strat_sens['rho_stars'], color=C_DARK, lw=2.5, ls='--', marker='s', markevery=5,
                  label='Optimal Recycling ($\\rho^*$)')
    ax2.set_ylabel('Equilibrium Recycling Rate ($\\rho^*$)', color=C_DARK, fontweight='bold')

    ax.set_title('B. Strategic Impact of Consumer Green Preference', loc='left')
    ax.spines['top'].set_visible(False);
    ax2.spines['top'].set_visible(False)

    lines, labels = ax.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.savefig('CLSC_Set3_Strategy.png')
    plt.savefig('CLSC_Set3_Strategy.pdf')
    plt.show()


# ==============================================================
# Terminal Report Generator (Paper Writing Assistant)
# ==============================================================
def print_paper_assistant_report(eq_nt, eq_d, eq_c):
    print("\n" + "═" * 85)
    print(" 📜 RESEARCH PAPER ASSISTANT: DATA & INSIGHTS REPORT ".center(85))
    print("═" * 85)

    # [Section 1] Metrics Comparison
    print("\n>>> SECTION 1: MACRO SYSTEM EQUILIBRIUM COMPARISON")
    print("-" * 85)
    headers = ["Metric", "No Tax (Base)", "Decentralized (Opt)", "Centralized (VIF)"]
    print(f"{headers[0]:<20} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<18}")
    print("-" * 85)

    metrics = [
        ("Carbon Tax (tau)", f"{0.0:.2f}", f"{eq_d['tau']:.2f}", f"{eq_c['tau']:.2f}"),
        ("Retail Price (p)", f"{eq_nt['p']:.2f}", f"{eq_d['p']:.2f}", f"{eq_c['p']:.2f}"),
        ("Market Demand (D)", f"{eq_nt['D']:.2f}", f"{eq_d['D']:.2f}", f"{eq_c['D']:.2f}"),
        ("Recycling Rate (rho)", f"{eq_nt['rho']:.4f}", f"{eq_d['rho']:.4f}", f"{eq_c['rho']:.4f}"),
        ("Total Emissions (E)", f"{eq_nt['E']:.2f}", f"{eq_d['E']:.2f}", f"{eq_c['E']:.2f}"),
        ("Social Welfare (SW)", f"{eq_nt['SW']:.2f}", f"{eq_d['SW']:.2f}", f"{eq_c['SW']:.2f}")
    ]
    for row in metrics:
        print(f"{row[0]:<20} | {row[1]:<18} | {row[2]:<18} | {row[3]:<18}")

    # [Section 2] Paper Writing Insights
    print("\n>>> SECTION 2: KEY FINDINGS FOR 'RESULTS & DISCUSSION' CHAPTER")
    print("-" * 85)

    # Calculations for insights
    E_reduction = (eq_nt['E'] - eq_d['E']) / eq_nt['E'] * 100
    SW_gain = (eq_d['SW'] - eq_nt['SW']) / eq_nt['SW'] * 100
    gap = eq_c['rho'] - eq_d['rho']

    total_prof_base = eq_nt['pi_m'] + eq_nt['pi_r']
    total_prof_tax = eq_d['pi_m'] + eq_d['pi_r']
    prof_drop = (total_prof_base - total_prof_tax) / total_prof_base * 100

    print(f"🔹 [Environmental Effectiveness]")
    print(f"   The implementation of the optimal carbon tax (tau* = {eq_d['tau']:.2f}) reduces total carbon")
    print(f"   emissions by {E_reduction:.2f}% compared to the baseline (from {eq_nt['E']:.1f} to {eq_d['E']:.1f}).")

    print(f"\n🔹 [Welfare vs. Profit Trade-off]")
    print(f"   While the tax decreases total supply chain profit by {prof_drop:.2f}%, the overall social")
    print(f"   welfare still increases by {SW_gain:.2f}%. This indicates that the environmental damage")
    print(f"   mitigation and tax revenue redistribution overcompensate the economic losses.")

    print(f"\n🔹 [Double Marginalization Effect]")
    print(
        f"   In the decentralized setting, the retailer's markup is {eq_d['p'] - eq_d['w']:.2f}, whereas the centralized")
    print(
        f"   VIF completely eliminates this double margin, leading to a retail price drop of {eq_d['p'] - eq_c['p']:.2f}.")
    print(f"   This leads to a significantly higher recycling rate in VIF (Coordination Gap = {gap:.4f}).")

    print(f"\n🔹 [Profit Distribution (Decentralized)]")
    m_share = eq_d['pi_m'] / total_prof_tax * 100
    r_share = eq_d['pi_r'] / total_prof_tax * 100
    print(f"   Under the optimal tax, the Manufacturer claims {m_share:.1f}% of the total channel profit,")
    print(f"   leaving {r_share:.1f}% to the Retailer, reflecting the Stackelberg leader's advantage.")

    print("═" * 85 + "\n")


# ==============================================================
# Main Execution
# ==============================================================
def main():
    cfg = ModelConfig()

    # Base computations
    eq_nt = decentralized_equilibrium(cfg, tau=0.0)
    eq_d = decentralized_equilibrium(cfg)
    eq_c = centralized_equilibrium(cfg)

    # 1. Print Paper Assistant Report
    print_paper_assistant_report(eq_nt, eq_d, eq_c)

    # 2. Run Engines
    print("[Processing...] Running high-resolution data engines for visualization...")
    sens = standard_sensitivity(cfg)
    strat_sens = strategic_sensitivity(cfg)

    # 3. Generate NPG-Style Figures
    print("[Processing...] Rendering 600 DPI publication-quality figures...")
    plot_set_1(eq_nt, eq_d, eq_c, sens)
    plot_set_2(eq_nt, eq_d, eq_c, sens)
    plot_set_3(cfg, eq_nt, eq_d, eq_c, strat_sens)

    print("\n✅ SUCCESS: All visual sets have been saved to your directory (600 DPI).")


if __name__ == "__main__":
    main()
