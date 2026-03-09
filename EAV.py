import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

# ==============================================================
# 全局绘图样式配置（移除LaTeX依赖）
# ==============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",  # 改用基础数学字体，避免LaTeX解析错误
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
# 模型配置
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
# 核心函数
# ==============================================================
def demand_fn(cfg: ModelConfig, p: float) -> float:
    return max(cfg.a - cfg.b * p + cfg.gamma * cfg.g, 0.0)


def emission_fn(cfg: ModelConfig, D: float, rho: float) -> float:
    return cfg.e_m * (1.0 - rho) * D + cfg.e_r * rho * D


def retailer_best_response(cfg: ModelConfig, w: float) -> float:
    return (cfg.a + cfg.gamma * cfg.g + cfg.b * w) / (2.0 * cfg.b)


# ==============================================================
# 分散决策（Stackelberg博弈）
# ==============================================================
def manufacturer_profit(cfg: ModelConfig, w: float, rho: float, tau: float) -> float:
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    return (w - cfg.c_m) * D + (cfg.c_m - cfg.c_r) * rho * D - tau * E - cfg.k * rho ** 2


def optimal_recycling_rate(cfg: ModelConfig, w: float, tau: float) -> float:
    res = minimize_scalar(
        lambda r: -manufacturer_profit(cfg, w, r, tau),
        bounds=(0.0, 1.0), method="bounded", options={'xatol': cfg.tol}
    )
    return res.x


def optimal_wholesale_price(cfg: ModelConfig, tau: float) -> tuple:
    def neg_profit(w):
        rho = optimal_recycling_rate(cfg, w, tau)
        return -manufacturer_profit(cfg, w, rho, tau)

    res = minimize_scalar(
        neg_profit, bounds=(cfg.c_m, cfg.a / cfg.b),
        method="bounded", options={'xatol': cfg.tol}
    )
    w_star = res.x
    rho_star = optimal_recycling_rate(cfg, w_star, tau)
    return w_star, rho_star


def decentralized_welfare(cfg: ModelConfig, tau: float) -> float:
    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    tax_revenue = tau * E
    env_damage = cfg.eta * E ** 2
    return CS + pi_m + pi_r + tax_revenue - env_damage


def optimal_carbon_tax(cfg: ModelConfig) -> float:
    res = minimize_scalar(
        lambda t: -decentralized_welfare(cfg, t),
        bounds=(0.0, cfg.tau_max), method="bounded", options={'xatol': cfg.tol}
    )
    return res.x


def decentralized_equilibrium(cfg: ModelConfig) -> dict:
    tau = optimal_carbon_tax(cfg)
    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    SW = decentralized_welfare(cfg, tau)
    return dict(tau=tau, w=w, p=p, rho=rho, D=D, E=E, CS=CS, pi_m=pi_m, pi_r=pi_r, SW=SW)


# ==============================================================
# 集中决策（VIF）
# ==============================================================
def vif_profit(cfg: ModelConfig, p: float, rho: float, tau: float) -> float:
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    return (p - cfg.c_m) * D + (cfg.c_m - cfg.c_r) * rho * D - cfg.k * rho ** 2 - tau * E


def vif_optimal_recycling(cfg: ModelConfig, p: float, tau: float) -> float:
    res = minimize_scalar(
        lambda r: -vif_profit(cfg, p, r, tau),
        bounds=(0.0, 1.0), method="bounded", options={'xatol': cfg.tol}
    )
    return res.x


def vif_optimal_price(cfg: ModelConfig, tau: float) -> tuple:
    def neg_vif(p):
        rho = vif_optimal_recycling(cfg, p, tau)
        return -vif_profit(cfg, p, rho, tau)

    res = minimize_scalar(
        neg_vif, bounds=(cfg.c_m + 1.0, cfg.a / cfg.b),
        method="bounded", options={'xatol': cfg.tol}
    )
    p_star = res.x
    rho_star = vif_optimal_recycling(cfg, p_star, tau)
    return p_star, rho_star


def vif_welfare(cfg: ModelConfig, tau: float) -> float:
    p, rho = vif_optimal_price(cfg, tau)
    D = demand_fn(cfg, p)
    if D <= 0: return -1e10
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    pi_vif = vif_profit(cfg, p, rho, tau)
    return CS + pi_vif + tau * E - cfg.eta * E ** 2


def centralized_equilibrium(cfg: ModelConfig) -> dict:
    res = minimize_scalar(
        lambda t: -vif_welfare(cfg, t),
        bounds=(0.0, cfg.tau_max), method="bounded", options={'xatol': cfg.tol}
    )
    tau_c = res.x
    p, rho = vif_optimal_price(cfg, tau_c)
    D = demand_fn(cfg, p)
    E = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g) ** 2 / (2.0 * cfg.b)
    pi_c = vif_profit(cfg, p, rho, tau_c)
    SW = vif_welfare(cfg, tau_c)
    return dict(tau=tau_c, p=p, rho=rho, D=D, E=E, CS=CS, pi_vif=pi_c, SW=SW)


# ==============================================================
# 情景分析 & 敏感性分析
# ==============================================================
def scenario_analysis(cfg: ModelConfig) -> pd.DataFrame:
    # 情景1：无碳税
    w_nt, rho_nt = optimal_wholesale_price(cfg, 0.0)
    p_nt = retailer_best_response(cfg, w_nt)
    D_nt = demand_fn(cfg, p_nt)
    E_nt = emission_fn(cfg, D_nt, rho_nt)
    SW_nt = decentralized_welfare(cfg, 0.0)

    # 情景2：分散决策+最优碳税
    eq_d = decentralized_equilibrium(cfg)

    # 情景3：集中决策（VIF）+最优碳税
    eq_c = centralized_equilibrium(cfg)

    gap = eq_c['rho'] - eq_d['rho']
    poa = eq_d['SW'] / eq_c['SW']

    print("\n" + "═" * 75)
    print(" ❖ CLSC MODEL EQUILIBRIUM RESULTS ❖".center(75))
    print("═" * 75)

    rows = [
        ("No Tax Base (tau=0)", 0.0, p_nt, rho_nt, D_nt, E_nt, SW_nt),
        ("Decentralized (tau*)", eq_d['tau'], eq_d['p'], eq_d['rho'], eq_d['D'], eq_d['E'], eq_d['SW']),
        ("Centralized VIF (tau^C)", eq_c['tau'], eq_c['p'], eq_c['rho'], eq_c['D'], eq_c['E'], eq_c['SW']),
    ]

    df = pd.DataFrame(rows,
                      columns=['Scenario', 'Tax (tau)', 'Price (p)', 'Recycle (rho)', 'Demand (D)', 'Emission (E)',
                               'Welfare (SW)'])
    print(df.to_string(index=False, float_format="%.4f"))
    print("-" * 75)

    print(f" ▸ Coordination Gap (rho^C - rho*) : {gap:.4f}  [{'✓ Efficient' if gap > 0 else '✗'}]")
    print(f" ▸ Efficiency Loss (Price of Anarchy): {(1 - poa) * 100:.2f}%")
    print(
        f" ▸ Double Marginalization Check    : p_dec ({eq_d['p']:.1f}) > p_VIF ({eq_c['p']:.1f})  [{'✓ Valid' if eq_d['p'] > eq_c['p'] else '✗'}]")

    int_tau_d = eq_d['tau'] < cfg.tau_max * 0.97
    int_rho_d = 0.01 < eq_d['rho'] < 0.99
    print(f"\n[Interior Solution Validation]")
    print(f"   tau* = {eq_d['tau']:7.3f} | Interior: {'[PASS]' if int_tau_d else '[WARN]'}")
    print(f"   rho* = {eq_d['rho']:7.4f} | Interior: {'[PASS]' if int_rho_d else '[WARN]'}")

    return df, eq_d, eq_c


def sensitivity_analysis(cfg: ModelConfig) -> dict:
    taus = np.linspace(0, cfg.tau_max, 100)
    rho_list, E_list, SW_list = [], [], []

    for tau in taus:
        w, rho = optimal_wholesale_price(cfg, tau)
        p = retailer_best_response(cfg, w)
        D = demand_fn(cfg, p)
        E = emission_fn(cfg, D, rho)
        SW = decentralized_welfare(cfg, tau)
        rho_list.append(rho)
        E_list.append(E)
        SW_list.append(SW)

    return dict(taus=taus, rho=np.array(rho_list), E=np.array(E_list), SW=np.array(SW_list))


# ==============================================================
# 可视化（全常规字母版）
# ==============================================================
def plot_results(cfg: ModelConfig, eq_d: dict, eq_c: dict, sens: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 移除LaTeX加粗，改用常规标题
    fig.suptitle("CLSC Optimal Carbon Taxation & Coordination Analysis", fontsize=18, y=1.02)

    tau_star, tau_c = eq_d['tau'], eq_c['tau']
    rho_star, rho_c = eq_d['rho'], eq_c['rho']
    taus = sens['taus']

    # 配色方案
    c_main = "#00429d"
    c_cen = "#73a2c6"
    c_alert = "#d73027"
    c_green = "#1a9850"
    c_fill = "#fee090"

    # 辅助函数：清理坐标轴
    def clean_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.6)

    # ── 子图1：碳税 vs 回收率 ──────────────────────────────
    ax = axes[0, 0]
    # 替换LaTeX符号为常规字母
    ax.plot(taus, sens['rho'], color=c_main, lw=2.5, label='rho*(tau) (Decentralized)')
    ax.axhline(rho_c, color=c_green, ls='--', lw=2, label=f'rho^C = {rho_c:.3f} (Centralized VIF)')
    ax.axvline(tau_star, color=c_alert, ls=':', lw=1.5, alpha=0.8)

    fill_mask = (taus >= tau_star * 0.8) & (taus <= tau_star * 1.2)
    ax.fill_between(taus, sens['rho'], rho_c, where=fill_mask, alpha=0.2, color=c_fill, label='Coordination Gap')

    ax.annotate('', xy=(tau_star * 1.05, rho_c), xytext=(tau_star * 1.05, rho_star),
                arrowprops=dict(arrowstyle='<|-|>', color=c_alert, lw=1.5))
    ax.text(tau_star * 1.08, (rho_star + rho_c) / 2, f'Delta rho={rho_c - rho_star:.3f}',
            color=c_alert, fontsize=11, va='center', fontweight='bold')
    ax.scatter([tau_star], [rho_star], color=c_alert, s=80, zorder=5, edgecolor='black')

    ax.set_xlabel('Carbon Tax (tau)')
    ax.set_ylabel('Recycling Rate (rho)')
    ax.set_title("A. Recycling Rate Dynamics", loc='left')
    ax.set_ylim(0, max(rho_c * 1.2, 1.0))
    ax.legend(frameon=False, loc='lower right')
    clean_spines(ax)

    # ── 子图2：碳税 vs 总排放量 ──────────────────────────────
    ax = axes[0, 1]
    ax.plot(taus, sens['E'], color='#f46d43', lw=2.5, label='E(tau) (Decentralized)')
    ax.axvline(tau_star, color=c_alert, ls=':', lw=1.5, label=f'tau* = {tau_star:.1f}')
    ax.axhline(eq_c['E'], color=c_green, ls='--', lw=2, label=f'E^C = {eq_c["E"]:.2f} (Centralized)')
    ax.scatter([tau_star], [eq_d['E']], color=c_alert, s=80, zorder=5, edgecolor='black')

    ax.set_xlabel('Carbon Tax (tau)')
    ax.set_ylabel('Total Carbon Emission (E)')
    ax.set_title("B. Environmental Impact", loc='left')
    ax.legend(frameon=False, loc='upper right')
    clean_spines(ax)

    # ── 子图3：碳税 vs 社会福利 ─────────────────────────────
    ax = axes[1, 0]
    ax.plot(taus, sens['SW'], color='#8c510a', lw=2.5, label='SW(tau) (Decentralized)')

    idx = np.nanargmax(sens['SW'])
    ax.scatter([taus[idx]], [sens['SW'][idx]], color=c_alert, s=100, zorder=5, edgecolor='black',
               label=f'Max Welfare (tau*={taus[idx]:.1f})')
    ax.axhline(eq_c['SW'], color=c_green, ls='--', lw=2, label=f'SW^C = {eq_c["SW"]:.0f} (Centralized)')
    ax.axvline(tau_star, color=c_alert, ls=':', lw=1.5)

    ax.set_xlabel('Carbon Tax (tau)')
    ax.set_ylabel('Social Welfare (SW)')
    ax.set_title("C. Social Welfare Maximization", loc='left')
    ax.legend(frameon=False, loc='lower right')
    clean_spines(ax)

    # ── 子图4：对比柱状图 ──────────────────────────────
    ax = axes[1, 1]
    w_nt, rho_nt = optimal_wholesale_price(cfg, 0.0)

    scenarios = [
        'No Tax\n(tau=0)',
        'Decentralized\n(tau*)',
        'Centralized\n(tau^C)'
    ]
    rho_vals = [rho_nt, rho_star, rho_c]
    SW_vals = [decentralized_welfare(cfg, 0.0), eq_d['SW'], eq_c['SW']]

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width / 2, rho_vals, width, label='Recycling Rate (rho)',
                   color=[c_cen, c_main, c_green], edgecolor='black', linewidth=1.2, alpha=0.85)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, SW_vals, width, label='Social Welfare (SW)',
                    color=['#e0f3f8', '#abd9e9', '#d9ef8b'], edgecolor='black', linewidth=1.2, alpha=0.9)

    # 标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax.set_ylim(0, max(rho_vals) * 1.25)
    ax2.set_ylim(min(SW_vals) * 0.98, max(SW_vals) * 1.05)

    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(SW_vals) * 0.005),
                 f'{bar.get_height():.0f}',
                 ha='center', va='bottom', fontsize=10, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylabel('Recycling Rate (rho)', color=c_main, fontweight='bold')
    ax2.set_ylabel('Social Welfare (SW)', color='#555555', fontweight='bold')
    ax.set_title("D. Benchmark Comparison", loc='left')

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.savefig('CLSC_Academic_Results.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


# ==============================================================
# 主函数
# ==============================================================
def main():
    print("═" * 75)
    print(" CLOSED-LOOP SUPPLY CHAIN STACKELBERG GAME OPTIMIZATION ".center(75))
    print("═" * 75)

    cfg = ModelConfig()

    print(f" [Parameters Initialized]")
    print(f"   ▸ Demand : a={cfg.a}, b={cfg.b}, gamma={cfg.gamma}, g={cfg.g}")
    print(f"   ▸ Costs  : c_m={cfg.c_m}, c_r={cfg.c_r}, k={cfg.k}")
    print(f"   ▸ Output : e_m={cfg.e_m}, e_r={cfg.e_r} (e_m > e_r => Eco-friendly recycle)")
    print(f"   ▸ Policy : eta={cfg.eta}, tau_max={cfg.tau_max}\n")

    print("[1/3] Computing analytical solutions and equilibrium states...")
    df, eq_d, eq_c = scenario_analysis(cfg)

    print("\n[2/3] Performing high-resolution sensitivity analysis...")
    sens = sensitivity_analysis(cfg)

    print("\n[3/3] Generating publication-ready visualizations...")
    plot_results(cfg, eq_d, eq_c, sens)

    print("\n ✔ Process successfully completed. HD Chart saved as 'CLSC_Academic_Results.png'")


if __name__ == "__main__":
    main()