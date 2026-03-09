import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

# ==============================================================
# Global Academic Plotting Style Configuration
# ==============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",  # Computer Modern for pure academic math font
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.constrained_layout.use": True
})


# ==============================================================
# Model Configuration (Core Unchanged)
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
        rho = optimal_recycling_rate(cfg, w, tau)
        return -manufacturer_profit(cfg, w, rho, tau)

    res = minimize_scalar(neg_profit, bounds=(cfg.c_m, cfg.a / cfg.b), method="bounded", options={'xatol': cfg.tol})
    w_star = res.x
    return w_star, optimal_recycling_rate(cfg, w_star, tau)


def decentralized_welfare(cfg: ModelConfig, tau: float) -> float:
    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    return CS + pi_m + pi_r + tau * E - cfg.eta * E ** 2


def optimal_carbon_tax(cfg: ModelConfig) -> float:
    res = minimize_scalar(lambda t: -decentralized_welfare(cfg, t), bounds=(0.0, cfg.tau_max), method="bounded",
                          options={'xatol': cfg.tol})
    return res.x


def decentralized_equilibrium(cfg: ModelConfig) -> dict:
    tau = optimal_carbon_tax(cfg)
    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    E = emission_fn(cfg, D, rho)
    return dict(tau=tau, w=w, p=p, rho=rho, D=D, E=E, SW=decentralized_welfare(cfg, tau))


# --- Centralized Benchmark (VIF) ---
def vif_profit(cfg: ModelConfig, p: float, rho: float, tau: float) -> float:
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    return (p - cfg.c_m) * D + (cfg.c_m - cfg.c_r) * rho * D - cfg.k * rho ** 2 - tau * emission_fn(cfg, D, rho)


def vif_optimal_recycling(cfg: ModelConfig, p: float, tau: float) -> float:
    res = minimize_scalar(lambda r: -vif_profit(cfg, p, r, tau), bounds=(0.0, 1.0), method="bounded",
                          options={'xatol': cfg.tol})
    return res.x


def vif_optimal_price(cfg: ModelConfig, tau: float) -> tuple:
    def neg_vif(p): return -vif_profit(cfg, p, vif_optimal_recycling(cfg, p, tau), tau)

    res = minimize_scalar(neg_vif, bounds=(cfg.c_m + 1.0, cfg.a / cfg.b), method="bounded", options={'xatol': cfg.tol})
    return res.x, vif_optimal_recycling(cfg, res.x, tau)


def vif_welfare(cfg: ModelConfig, tau: float) -> float:
    p, rho = vif_optimal_price(cfg, tau)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    return CS + vif_profit(cfg, p, rho, tau) + tau * E - cfg.eta * E ** 2


def centralized_equilibrium(cfg: ModelConfig) -> dict:
    res = minimize_scalar(lambda t: -vif_welfare(cfg, t), bounds=(0.0, cfg.tau_max), method="bounded",
                          options={'xatol': cfg.tol})
    p, rho = vif_optimal_price(cfg, res.x)
    return dict(tau=res.x, p=p, rho=rho, D=demand_fn(cfg, p), E=emission_fn(cfg, demand_fn(cfg, p), rho),
                SW=vif_welfare(cfg, res.x))


# ==============================================================
# Analytical Engines
# ==============================================================
def standard_sensitivity(cfg: ModelConfig) -> dict:
    taus = np.linspace(0, cfg.tau_max, 100)
    res = {'taus': taus, 'rho': [], 'E': [], 'SW': [], 'p': [], 'w': []}
    for t in taus:
        w, rho = optimal_wholesale_price(cfg, t)
        p = retailer_best_response(cfg, w)
        D = demand_fn(cfg, p)  # [Fix Applied] Correctly calculate Demand first
        res['w'].append(w);
        res['p'].append(p);
        res['rho'].append(rho)
        res['E'].append(emission_fn(cfg, D, rho))  # [Fix Applied] Passed correct cfg parameter
        res['SW'].append(decentralized_welfare(cfg, t))
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
# Advanced Visualization (3 Sets of 1x2 Plots)
# ==============================================================
C_MAIN, C_CEN, C_RED, C_GRN, C_YEL = "#00429d", "#73a2c6", "#d73027", "#1a9850", "#fee090"


def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.5)


# --- SET 1: Operational Responses ---
def plot_set_1(eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Set 1: Operational Responses to Carbon Taxation", fontsize=16, y=1.05)

    # 1A: Pricing Dynamics
    ax = axes[0]
    ax.plot(sens['taus'], sens['p'], color=C_MAIN, lw=2.5, label='Retail Price ($p$)')
    ax.plot(sens['taus'], sens['w'], color=C_CEN, lw=2.5, ls='-.', label='Wholesale Price ($w$)')
    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=1.5, alpha=0.8)
    ax.scatter([eq_d['tau'], eq_d['tau']], [eq_d['p'], eq_d['w']], color=C_RED, zorder=5)
    ax.set(xlabel='Carbon Tax ($\\tau$)', ylabel='Price Level', title='A. Price Transmission Mechanism')
    ax.legend(frameon=False);
    clean_ax(ax)

    # 1B: Recycling Rate & Gap
    ax = axes[1]
    ax.plot(sens['taus'], sens['rho'], color=C_MAIN, lw=2.5, label='Decentralized ($\\rho^*$)')
    ax.axhline(eq_c['rho'], color=C_GRN, ls='--', lw=2, label=f"Centralized ($\\rho^C={eq_c['rho']:.3f}$)")
    mask = (sens['taus'] >= eq_d['tau'] * 0.6) & (sens['taus'] <= eq_d['tau'] * 1.4)
    ax.fill_between(sens['taus'], sens['rho'], eq_c['rho'], where=mask, color=C_YEL, alpha=0.3,
                    label='Coordination Gap')
    ax.annotate('', xy=(eq_d['tau'] * 1.1, eq_c['rho']), xytext=(eq_d['tau'] * 1.1, eq_d['rho']),
                arrowprops=dict(arrowstyle='<|-|>', color=C_RED))
    ax.text(eq_d['tau'] * 1.15, (eq_d['rho'] + eq_c['rho']) / 2, f'$\\Delta\\rho$={eq_c["rho"] - eq_d["rho"]:.3f}',
            color=C_RED, fontweight='bold')
    ax.set(xlabel='Carbon Tax ($\\tau$)', ylabel='Recycling Rate ($\\rho$)', title='B. Recycling Optimization & Gap')
    ax.legend(frameon=False, loc='lower right');
    clean_ax(ax)

    plt.savefig('CLSC_Set1_Operational.png', dpi=300, bbox_inches='tight')
    plt.show()


# --- SET 2: Macro-Policy & Environment ---
def plot_set_2(eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Set 2: Macro-Policy & Environmental Impact", fontsize=16, y=1.05)

    # 2A: Emission Trajectory
    ax = axes[0]
    ax.plot(sens['taus'], sens['E'], color='#f46d43', lw=2.5, label='Total Emissions ($E$)')
    ax.axhline(eq_c['E'], color=C_GRN, ls='--', lw=2, label=f"Centralized Benchmark ($E^C$)")
    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=1.5, label=f"Optimum $\\tau^*={eq_d['tau']:.1f}$")
    ax.scatter([eq_d['tau']], [eq_d['E']], color=C_RED, s=60, zorder=5)
    ax.set(xlabel='Carbon Tax ($\\tau$)', ylabel='Carbon Emissions ($E$)', title='A. Environmental Impact Control')
    ax.legend(frameon=False);
    clean_ax(ax)

    # 2B: Welfare Maximization
    ax = axes[1]
    ax.plot(sens['taus'], sens['SW'], color='#8c510a', lw=2.5, label='Social Welfare ($SW$)')
    ax.axhline(eq_c['SW'], color=C_GRN, ls='--', lw=2, label=f"Centralized Bound ($SW^C$)")
    idx = np.nanargmax(sens['SW'])
    ax.scatter([sens['taus'][idx]], [sens['SW'][idx]], color=C_RED, s=80, edgecolor='black', zorder=5)
    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=1.5)
    ax.set(xlabel='Carbon Tax ($\\tau$)', ylabel='Social Welfare ($SW$)', title='B. Social Welfare Maximization')
    ax.legend(frameon=False, loc='lower right');
    clean_ax(ax)

    plt.savefig('CLSC_Set2_Macro.png', dpi=300, bbox_inches='tight')
    plt.show()


# --- SET 3: Strategic Insights ---
def plot_set_3(cfg, eq_d, eq_c, strat_sens):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Set 3: Strategic Benchmarking & Consumer Insights", fontsize=16, y=1.05)

    # 3A: Benchmark Bar Chart (Dual Axis)
    ax = axes[0]
    w_nt, rho_nt = optimal_wholesale_price(cfg, 0.0)
    SW_nt = decentralized_welfare(cfg, 0.0)
    labels = ['No Tax ($\\tau=0$)', 'Decentralized ($\\tau^*$)', 'Centralized ($\\tau^C$)']
    rho_vals, sw_vals = [rho_nt, eq_d['rho'], eq_c['rho']], [SW_nt, eq_d['SW'], eq_c['SW']]

    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, rho_vals, width, label='Recycling Rate ($\\rho$)', color=[C_CEN, C_MAIN, C_GRN],
           edgecolor='black', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, sw_vals, width, label='Social Welfare ($SW$)', color=['#e0f3f8', '#abd9e9', '#d9ef8b'],
            edgecolor='black', alpha=0.9)

    for i, v in enumerate(rho_vals): ax.text(x[i] - width / 2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    for i, v in enumerate(sw_vals): ax2.text(x[i] + width / 2, v + (max(sw_vals) * 0.01), f'{v:.0f}', ha='center')

    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_ylabel('Recycling Rate ($\\rho$)', color=C_MAIN);
    ax2.set_ylabel('Social Welfare ($SW$)', color='#555555')
    ax.set_title('A. System Efficiency Benchmarks', loc='left')
    ax.spines['top'].set_visible(False);
    ax2.spines['top'].set_visible(False)
    lines, labels = ax.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)

    # 3B: Sensitivity to Green Preference (Dual Axis)
    ax = axes[1]
    gammas = strat_sens['gammas']
    ax.plot(gammas, strat_sens['tau_stars'], color='#5e4fa2', lw=2.5, label='Optimal Tax ($\\tau^*$)')
    ax.set_xlabel('Consumer Green Preference ($\\gamma$)');
    ax.set_ylabel('Optimal Carbon Tax ($\\tau^*$)', color='#5e4fa2')

    ax2 = ax.twinx()
    ax2.plot(gammas, strat_sens['rho_stars'], color=C_GRN, lw=2.5, ls='--', label='Optimal Recycling ($\\rho^*$)')
    ax2.set_ylabel('Equilibrium Recycling Rate ($\\rho^*$)', color=C_GRN)

    ax.set_title('B. Sensitivity to Green Preference', loc='left')
    ax.spines['top'].set_visible(False);
    ax2.spines['top'].set_visible(False)
    lines, labels = ax.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.savefig('CLSC_Set3_Strategic.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================
# Main Execution
# ==============================================================
def main():
    print("═" * 80)
    print(" CLOSED-LOOP SUPPLY CHAIN STACKELBERG OPTIMIZATION (SCIENCE EDITION) ".center(80))
    print("═" * 80)

    cfg = ModelConfig()

    # Solve scenarios
    w_nt, rho_nt = optimal_wholesale_price(cfg, 0.0)
    SW_nt = decentralized_welfare(cfg, 0.0)
    eq_d = decentralized_equilibrium(cfg)
    eq_c = centralized_equilibrium(cfg)

    # Console Report
    rows = [
        ("No Tax Baseline", 0.0, rho_nt, SW_nt),
        ("Decentralized Optimum", eq_d['tau'], eq_d['rho'], eq_d['SW']),
        ("Centralized (VIF)", eq_c['tau'], eq_c['rho'], eq_c['SW'])
    ]
    df = pd.DataFrame(rows, columns=['Scenario', 'Opt. Tax', 'Recycling Rate', 'Welfare'])
    print("\n[I] EQUILIBRIUM OUTCOMES:")
    print("-" * 60)
    print(df.to_string(index=False, float_format="%.4f"))
    print("-" * 60)
    print(f" ▸ Coordination Gap : {eq_c['rho'] - eq_d['rho']:.4f}")
    print(f" ▸ Efficiency Loss  : {(1 - eq_d['SW'] / eq_c['SW']) * 100:.2f}%\n")

    print("[II] RUNNING ANALYTICS ENGINE...")
    sens = standard_sensitivity(cfg)
    strat_sens = strategic_sensitivity(cfg)

    print("[III] GENERATING PUBLICATION-READY FIGURES...")
    plot_set_1(eq_d, eq_c, sens)
    plot_set_2(eq_d, eq_c, sens)
    plot_set_3(cfg, eq_d, eq_c, strat_sens)

    print(" ✔ All visual sets successfully generated & saved to your working directory.")


if __name__ == "__main__":
    main()
