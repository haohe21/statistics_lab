"""抽样分布与参数估计教学实验室

运行：streamlit run sampling_lab.py
"""

from __future__ import annotations

import math
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats


# Explicitly select a CJK-capable font. Without this, Matplotlib often falls
# back to DejaVu Sans and renders Chinese titles/axis labels as squares.
from pathlib import Path
from matplotlib import font_manager

# sampling_lab.py 位于“抽样分布_区间估计”子文件夹，
# 因此需要向上返回一层，才能找到仓库根目录的 fonts 文件夹。
font_path = (
    Path(__file__).resolve().parent.parent
    / "fonts"
    / "NotoSansSC-Regular.ttf"
)

if font_path.exists():
    # 将项目中的中文字体注册给 Matplotlib
    font_manager.fontManager.addfont(str(font_path))

    # 读取字体的内部名称
    cjk_font = font_manager.FontProperties(
        fname=str(font_path)
    ).get_name()

    mpl.rcParams["font.family"] = cjk_font
    mpl.rcParams["font.sans-serif"] = [cjk_font]
else:
    # 找不到字体时的备用设置
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "DejaVu Sans",
    ]

mpl.rcParams["axes.unicode_minus"] = False


st.set_page_config(
    page_title="抽样分布与参数估计实验室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE = {
    "navy": "#15324B",
    "teal": "#1F7A8C",
    "coral": "#F25F5C",
    "gold": "#E1A928",
    "ink": "#1E2933",
    "muted": "#607080",
    "grid": "#D7E2E8",
    "bg": "#F7FAFC",
}


def draw_values(
    rng: np.random.Generator,
    distribution: str,
    size: Tuple[int, ...] | int,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Generate a population with the selected shape and requested mean/SD."""
    if distribution == "正态分布":
        z = rng.normal(0, 1, size)
    elif distribution == "右偏分布":
        log_sigma = 0.9
        raw = rng.lognormal(mean=0.0, sigma=log_sigma, size=size)
        raw_mean = math.exp(log_sigma**2 / 2)
        raw_sd = math.sqrt((math.exp(log_sigma**2) - 1) * math.exp(log_sigma**2))
        z = (raw - raw_mean) / raw_sd
    elif distribution == "厚尾分布":
        df = 3
        z = rng.standard_t(df, size=size) / math.sqrt(df / (df - 2))
    elif distribution == "T 分布":
        # 固定自由度 5，保留比正态更厚的尾部，同时具有有限方差。
        df = 5
        z = rng.standard_t(df, size=size) / math.sqrt(df / (df - 2))
    elif distribution == "泊松分布":
        # λ=10 仅用于确定离散形状；随后按理论均值和标准差变换。
        poisson_lambda = 10
        z = (rng.poisson(lam=poisson_lambda, size=size) - poisson_lambda) / math.sqrt(poisson_lambda)
    elif distribution == "均匀分布":
        # U(-√3, √3) 的理论均值为 0、标准差为 1。
        z = rng.uniform(-math.sqrt(3), math.sqrt(3), size=size)
    elif distribution == "卡方分布":
        # 固定自由度 5，理论标准化后仍保留明显的右偏形状。
        df = 5
        z = (rng.chisquare(df=df, size=size) - df) / math.sqrt(2 * df)
    else:  # 双峰分布
        component = rng.integers(0, 2, size=size)
        z = rng.normal(np.where(component == 0, -1.0, 1.0), 0.42, size=size)
        z = z / math.sqrt(1 + 0.42**2)
    return mu + sigma * z


def sample_means(
    rng: np.random.Generator,
    distribution: str,
    mu: float,
    sigma: float,
    n: int,
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = draw_values(rng, distribution, (repetitions, n), mu, sigma)
    means = values.mean(axis=1)
    first_sample = values[0]
    return first_sample, means


def estimator_property_samples(
    rng: np.random.Generator,
    distribution: str,
    mu: float,
    sigma: float,
    n: int,
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three estimators used to illustrate the three properties.

    The sample mean and the first observation are both unbiased for μ under
    iid sampling, but the sample mean is more efficient.  The shifted mean is
    deliberately biased so students can see that centering and spread are
    separate ideas.
    """
    values = draw_values(rng, distribution, (repetitions, n), mu, sigma)
    mean_estimator = values.mean(axis=1)
    first_observation = values[:, 0]
    biased_mean = mean_estimator + 0.5 * sigma
    return mean_estimator, first_observation, biased_mean


def t_critical(confidence: int, df: int) -> float:
    return float(stats.t.ppf(0.5 + confidence / 200, df))


def make_hist(ax, data, title, color, bins=35, vline=None, vline_label=None, xlim=None):
    ax.hist(data, bins=bins, color=color, alpha=0.78, edgecolor="white", linewidth=0.45)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if vline is not None:
        ax.axvline(vline, color=PALETTE["coral"], linewidth=2.2, linestyle="--")
        if vline_label:
            ax.text(vline, ax.get_ylim()[1] * 0.94, vline_label, color=PALETTE["coral"],
                    ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.55, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, colors=PALETTE["muted"])


def ci_simulation(
    rng: np.random.Generator,
    distribution: str,
    mu: float,
    sigma: float,
    n: int,
    repetitions: int,
    confidence: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = draw_values(rng, distribution, (repetitions, n), mu, sigma)
    means = values.mean(axis=1)
    sds = values.std(axis=1, ddof=1)
    crit = t_critical(confidence, n - 1)
    se = sds / math.sqrt(n)
    low = means - crit * se
    high = means + crit * se
    covered = (low <= mu) & (mu <= high)
    return means, low, high, float(covered.mean())


def explain_distribution(distribution: str, n: int) -> str:
    if distribution == "正态分布":
        return "总体为正态时，样本均值的抽样分布通常较快呈现对称形状。"
    if distribution == "右偏分布":
        return "右偏总体在小样本时会使样本均值的抽样分布偏斜；增大 n 后通常逐渐接近正态。"
    if distribution == "厚尾分布":
        return "厚尾总体更容易出现极端值，样本均值的抽样分布需要更大的 n 才稳定。"
    if distribution == "T 分布":
        return "T 分布（教学中固定自由度 df=5）比正态分布尾部更厚；自由度越小，极端值越常见。"
    if distribution == "泊松分布":
        return "泊松分布（教学中固定 λ=10）是离散分布；标准化后仍能看到整数台阶和右偏特征。"
    if distribution == "均匀分布":
        return "均匀分布在一个有限区间内等可能；单个观测并不呈钟形，但样本均值随 n 增大通常趋近正态。"
    if distribution == "卡方分布":
        return "卡方分布（教学中固定 df=5）只取非负值且右偏；样本均值随 n 增大通常逐渐对称。"
    return "双峰总体不等于样本均值也一定双峰；均值抽样分布的形状取决于 n 和总体结构。"


st.markdown(
    """
    <style>
    .main-title { font-size: 2.1rem; font-weight: 750; color: #15324B; margin-bottom: 0.1rem; }
    .subtitle { color: #607080; font-size: 1.02rem; margin-bottom: 1rem; }
    .hint { background: #EEF5F8; border-left: 4px solid #1F7A8C; padding: .65rem .9rem; border-radius: .25rem; }
    .warning { background: #FFF4D5; border-left: 4px solid #E1A928; padding: .65rem .9rem; border-radius: .25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">抽样分布与参数估计实验室</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">拖动参数，先预测，再观察总体、一次样本、抽样分布和置信区间如何变化。</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("实验参数")
    distribution = st.selectbox(
        "总体分布",
        [
            "正态分布",
            "右偏分布",
            "厚尾分布",
            "双峰分布",
            "T 分布",
            "泊松分布",
            "均匀分布",
            "卡方分布",
        ],
    )
    mu = st.slider("总体均值 μ", -20.0, 100.0, 50.0, 1.0)
    sigma = st.slider("总体标准差 σ", 0.5, 30.0, 10.0, 0.5)
    n = st.slider("每个样本的样本量 n", 2, 500, 30, 1)
    repetitions = st.slider("重复抽样次数 B", 20, 10000, 1000, 20)
    confidence = st.select_slider("置信水平", options=[90, 95, 99], value=95, format_func=lambda x: f"{x}%")
    seed = st.number_input("随机种子（改变它可重新抽样）", min_value=0, max_value=999999, value=2026, step=1)
    st.caption("n 决定每个样本的信息量；B 决定我们把抽样分布看得多清楚。")

rng = np.random.default_rng(int(seed))
population_preview = draw_values(rng, distribution, 5000, mu, sigma)
first_sample, means = sample_means(rng, distribution, mu, sigma, n, repetitions)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 抽样分布实验",
    "② 置信区间覆盖",
    "③ 数据库视角",
    "④ 课堂任务",
    "⑤ 估计量三性质",
])

with tab1:
    st.markdown('<div class="hint">先看三个层次：总体中的个体值 → 一次抽到的样本 → 所有重复样本的统计量。</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.25), constrained_layout=True)
    # Keep a common fixed reference scale so changing σ visibly changes the
    # spread instead of being hidden by per-panel or per-run autoscaling.
    # The default σ=10 gives a comfortable ±4.5σ viewing window.
    reference_sigma = 10.0
    x_limits = (mu - 4.5 * reference_sigma, mu + 4.5 * reference_sigma)
    make_hist(axes[0], population_preview, "总体分布", PALETTE["teal"], vline=mu, vline_label="μ", xlim=x_limits)
    make_hist(
        axes[1],
        first_sample,
        f"一次样本（n={n}）",
        PALETTE["coral"],
        vline=float(first_sample.mean()),
        vline_label=r"$\bar{x}$",
        xlim=x_limits,
    )
    make_hist(axes[2], means, f"样本均值的抽样分布（B={repetitions}）", PALETTE["gold"], vline=mu, vline_label="μ", xlim=x_limits)
    axes[0].set_xlabel("个体观测值")
    axes[1].set_xlabel("样本中的观测值")
    axes[2].set_xlabel("样本均值")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("三幅图共用固定横轴（以 σ=10 为参考），因此改变总体标准差时，分布宽度会直接显示出来；极端值可能落在显示范围之外。")

    theoretical_se = sigma / math.sqrt(n)
    simulated_se = float(means.std(ddof=1))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总体均值 μ", f"{mu:.2f}")
    c2.metric("一次样本均值 x̄", f"{first_sample.mean():.2f}")
    c3.metric("理论标准误 σ/√n", f"{theoretical_se:.3f}")
    c4.metric("模拟标准误 SD(x̄)", f"{simulated_se:.3f}", f"差异 {simulated_se - theoretical_se:+.3f}")
    st.write(explain_distribution(distribution, n))
    st.markdown(
        f"**当前解释：** 每个 `Xᵢ` 的方差仍由总体的 `σ²={sigma**2:.2f}` 决定；但是平均 `n={n}` 个独立观测后，"
        f"样本均值的方差变为 `σ²/n={sigma**2/n:.3f}`，标准误为 `σ/√n={theoretical_se:.3f}`。"
    )

with tab2:
    st.markdown('<div class="hint">绿色区间覆盖真实 μ，红色区间没有覆盖。覆盖率是长期性质，不保证每 100 个区间恰好有 95 个覆盖。</div>', unsafe_allow_html=True)
    ci_reps = min(repetitions, 220)
    ci_rng = np.random.default_rng(int(seed) + 991)
    ci_means, lows, highs, coverage = ci_simulation(ci_rng, distribution, mu, sigma, n, ci_reps, confidence)
    fig, (ax, ax_ref) = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
        gridspec_kw={"width_ratios": [2.45, 1]},
        constrained_layout=True,
    )
    order = np.arange(ci_reps)
    contains = (lows <= mu) & (mu <= highs)
    for i, (lo, hi, ok) in enumerate(zip(lows, highs, contains)):
        ax.plot([lo, hi], [i, i], color=PALETTE["teal"] if ok else PALETTE["coral"], linewidth=1.8, alpha=0.82)
        ax.scatter([lo, hi], [i, i], color=PALETTE["teal"] if ok else PALETTE["coral"], s=10)
    ax.axvline(mu, color=PALETTE["navy"], linestyle="--", linewidth=2, label="真实 μ")
    ax.set_title(f"{confidence}% 置信区间：前 {ci_reps} 次重复抽样", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlabel("总体均值的区间估计")
    ax.set_ylabel("重复抽样编号")
    ax.grid(axis="x", color=PALETTE["grid"], alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    reference_levels = [90, 95, 99]
    reference_crit = np.array([t_critical(level, n - 1) for level in reference_levels])
    reference_half_width = reference_crit * sigma / math.sqrt(n)
    reference_colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["gold"]]
    bars = ax_ref.bar(
        [f"{level}%" for level in reference_levels],
        reference_half_width,
        color=reference_colors,
        width=0.58,
    )
    ax_ref.set_title("置信水平越高，区间越宽", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax_ref.set_ylabel("理论区间半宽")
    ax_ref.grid(axis="y", color=PALETTE["grid"], alpha=0.55)
    ax_ref.spines[["top", "right"]].set_visible(False)
    ax_ref.tick_params(labelsize=8, colors=PALETTE["muted"])
    ax_ref.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    selected_idx = reference_levels.index(confidence)
    ax_ref.text(
        selected_idx,
        reference_half_width[selected_idx] * 1.08,
        f"当前：{confidence}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("实际覆盖率", f"{coverage:.1%}")
    d2.metric("目标覆盖率", f"{confidence}%")
    d3.metric("未覆盖区间数", f"{int((~contains).sum())} / {ci_reps}")
    d4.metric("当前理论半宽", f"{reference_half_width[selected_idx]:.3f}")
    st.markdown(
        "**严格表述：** 如果按照同样的方法不断抽样，长期来看约有目标比例的区间包含固定的 μ。"
        "本次已经得到的某一个区间，要么包含 μ，要么不包含 μ。"
    )

with tab3:
    st.markdown('<div class="hint">数据库有多少行，不等于有多少个独立观测。先选择数据结构，再比较名义样本量和有效信息量。</div>', unsafe_allow_html=True)
    design = st.radio("数据库中的一行代表什么？", ["独立个体", "重复测量：每人多次", "聚类数据：每组多人"], horizontal=True)
    units = st.slider("独立单位数", 10, 500, 50, 10)
    if design == "独立个体":
        # 在此模式中，每一行就是一个独立单位，其他结构参数不适用。
        rows_per_unit = 1
        rho = 0.0
        st.caption("独立个体模式：每个单位只有 1 行，记录之间视为不相关（m=1，ρ=0）。")
    else:
        rows_per_unit = st.slider("每个独立单位的记录数", 2, 20, 5, 1)
        rho = st.slider("同一单位内的相关程度 ρ", 0.0, 0.9, 0.4, 0.05)
        if design == "重复测量：每人多次":
            st.caption("重复测量模式：同一人的多次记录通常相关；m 越大或 ρ 越高，新增记录带来的信息越少。")
        else:
            st.caption("聚类模式：同一组内的多人记录通常相关；例如同一班级、家庭或医院中的个体。")
    total_rows = units * rows_per_unit
    design_effect = 1 + (rows_per_unit - 1) * rho
    effective_n = total_rows / design_effect
    naive_se = sigma / math.sqrt(total_rows)
    cluster_se = sigma / math.sqrt(effective_n)
    left, right = st.columns([1.1, 1.9])
    with left:
        st.metric("数据库行数", f"{total_rows}")
        st.metric("独立单位数", f"{units}")
        st.metric("设计效应", f"{design_effect:.2f}")
        st.metric("近似有效样本量", f"{effective_n:.1f}")
        st.metric("有效信息占比", f"{effective_n / total_rows:.1%}")
    with right:
        x = np.arange(2)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        bars = ax.bar(["把每行当独立", "考虑相关结构"], [naive_se, cluster_se], color=[PALETTE["teal"], PALETTE["coral"]], width=0.55)
        ax.set_ylabel("均值标准误")
        ax.set_title(f"{design}：相关记录会改变标准误", fontsize=13, fontweight="bold", color=PALETTE["navy"])
        ax.grid(axis="y", color=PALETTE["grid"], alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    if design == "独立个体":
        st.success("若每一行确实是一个相互独立的个体，名义行数与独立单位数一致，简单的 σ/√n 更有可能合理。")
    else:
        st.warning("同一被试/班级/家庭内的记录通常相关。若把所有行直接当成独立观测，标准误可能被低估，置信区间会过窄。")
    st.markdown(
        f"**当前解读：** 数据库共有 `{total_rows}` 行，但按当前设计约相当于 `{effective_n:.1f}` 个独立观测；"
        f"忽略相关性会把标准误从 `{cluster_se:.3f}` 低估为 `{naive_se:.3f}`。"
    )
    st.caption("这里使用的是教学用设计效应近似：DEFF = 1 + (m−1)ρ。正式分析应根据研究设计使用重复测量、多层模型或聚类稳健标准误。")

with tab4:
    st.markdown("### 课堂挑战：先预测，再拖动参数验证")
    tasks = [
        "固定 σ，分别把 n 调为 4、16、64。标准误大约如何变化？",
        "固定 n=30，把 B 从 100 增加到 10,000。什么变化了，什么没有变化？",
        "选择右偏或厚尾总体，比较 n=5 与 n=100 时抽样分布的形状。",
        "设置置信水平为 95%，观察覆盖率为什么不一定恰好等于 95%。",
        "在数据库视角中，让行数远大于独立单位数，解释为什么不能直接用 σ/√行数。",
        "比较均匀、泊松和卡方总体：哪些是离散的？哪些明显右偏？增大 n 后样本均值发生什么变化？",
    ]
    for i, task in enumerate(tasks, 1):
        st.markdown(f"**{i}.** {task}")
    st.markdown("---")
    st.markdown("### 下课前请学生写下三句话")
    st.write("1. `n` 决定每个样本的信息量；`B` 决定抽样分布显示得多清楚。")
    st.write("2. 标准差描述个体差异；标准误描述统计量在重复抽样中的差异。")
    st.write("3. 置信区间很窄，只能说明随机不确定性较小，不能自动证明样本无偏。")

with tab5:
    st.markdown(
        '<div class="hint">把同一个抽样过程重复很多次：看估计量的中心（无偏性）、宽度（有效性）和随 n 增大是否集中（一致性）。</div>',
        unsafe_allow_html=True,
    )
    property_reps = min(repetitions, 2000)
    property_rng = np.random.default_rng(int(seed) + 2027)
    mean_est, first_est, biased_est = estimator_property_samples(
        property_rng, distribution, mu, sigma, n, property_reps
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)

    # Panel 1: unbiasedness is about the center of the repeated-estimate
    # distribution. The shifted mean is intentionally biased.
    bins = np.histogram_bin_edges(np.concatenate([mean_est, biased_est]), bins=35)
    axes[0].hist(mean_est, bins=bins, alpha=0.68, color=PALETTE["teal"], label="样本均值（无偏示范）")
    axes[0].hist(biased_est, bins=bins, alpha=0.58, color=PALETTE["coral"], label="人为加偏移的均值")
    axes[0].axvline(mu, color=PALETTE["navy"], linestyle="--", linewidth=2, label="真实 μ")
    axes[0].axvline(mean_est.mean(), color=PALETTE["teal"], linewidth=2, linestyle=":")
    axes[0].axvline(biased_est.mean(), color=PALETTE["coral"], linewidth=2, linestyle=":")
    axes[0].set_title("无偏性：长期中心是否对准 μ", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[0].set_xlabel("重复抽样得到的估计值")
    axes[0].legend(frameon=False, fontsize=8)

    # Panel 2: both X1 and x-bar are unbiased, but x-bar is much narrower.
    eff_bins = np.histogram_bin_edges(np.concatenate([mean_est, first_est]), bins=35)
    axes[1].hist(first_est, bins=eff_bins, alpha=0.55, color=PALETTE["gold"], label=r"单个观测 $X_1$（无偏但低效）")
    axes[1].hist(mean_est, bins=eff_bins, alpha=0.72, color=PALETTE["teal"], label=r"样本均值 $\bar{x}$")
    axes[1].axvline(mu, color=PALETTE["navy"], linestyle="--", linewidth=2, label="真实 μ")
    axes[1].set_title("有效性：在无偏估计量中谁的方差更小", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[1].set_xlabel("重复抽样得到的估计值")
    axes[1].legend(frameon=False, fontsize=8)

    # Panel 3: consistency is illustrated by shrinking errors as n grows.
    consistency_ns = [2, 8, 32, 128]
    consistency_reps = min(repetitions, 1200)
    consistency_rng = np.random.default_rng(int(seed) + 9090)
    errors = []
    for n_level in consistency_ns:
        vals = draw_values(consistency_rng, distribution, (consistency_reps, n_level), mu, sigma)
        errors.append(vals.mean(axis=1) - mu)
    all_errors = np.concatenate(errors)
    error_limit = max(float(sigma), float(np.quantile(np.abs(all_errors), 0.995) * 1.1), 1.0)
    box = axes[2].boxplot(
        errors,
        orientation="horizontal",
        positions=np.arange(len(consistency_ns)),
        tick_labels=[f"n={n_level}" for n_level in consistency_ns],
        patch_artist=True,
        widths=0.58,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(PALETTE["gold"])
        patch.set_alpha(0.65)
    for median in box["medians"]:
        median.set_color(PALETTE["navy"])
        median.set_linewidth(1.8)
    axes[2].axvline(0, color=PALETTE["coral"], linestyle="--", linewidth=2, label="误差=0")
    axes[2].set_xlim(-error_limit, error_limit)
    axes[2].set_title("一致性：n 增大，样本均值误差收缩", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    axes[2].set_xlabel(r"估计误差  $\hat{\mu}-\mu$")
    axes[2].set_ylabel("每个样本的样本量")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color=PALETTE["grid"], alpha=0.55, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors=PALETTE["muted"])
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    tolerance = 0.1 * sigma
    c1, c2, c3 = st.columns(3)
    c1.metric("样本均值的模拟偏差", f"{mean_est.mean() - mu:+.3f}")
    c2.metric("效率对照：SD(X₁) / SD(x̄)", f"{first_est.std(ddof=1) / mean_est.std(ddof=1):.2f} 倍")
    c3.metric(f"n=128 时 |误差| < {tolerance:.1f} 的比例", f"{np.mean(np.abs(errors[-1]) < tolerance):.1%}")
    st.markdown(
        "**如何读图：** 无偏性看重复估计值的平均是否接近 μ；有效性在比较无偏估计量时看谁的分布更窄；一致性看 n 增大后估计误差是否越来越集中在 0 附近。"
    )
    st.caption(
        "这里用‘单个观测 X₁’作为无偏但低效的对照，用‘样本均值 + 0.5σ’作为故意有偏的示范。"
        "X₁ 即使在 n 增大时也只使用第一个观测，因此无偏却不一致；模拟中心的轻微偏离来自有限重复次数，并不改变定义。"
    )

st.divider()
st.caption("教学用途：本程序用于直观理解抽样分布与参数估计；模拟结果受随机种子、重复次数和模型设定影响。")
