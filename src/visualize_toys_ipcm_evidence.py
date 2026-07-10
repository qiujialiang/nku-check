import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import get_user_seqs  # noqa: E402
from model.bsarec import BSARecModel  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 11,
    }
)


def build_args(item_size, c, alpha, ipcm=False):
    return SimpleNamespace(
        data_name="Toys_and_Games",
        model_type="BSARec",
        item_size=item_size,
        num_users=19413,
        batch_size=512,
        max_seq_length=50,
        hidden_size=64,
        num_hidden_layers=2,
        hidden_act="gelu",
        num_attention_heads=1,
        attention_probs_dropout_prob=0.5,
        hidden_dropout_prob=0.5,
        initializer_range=0.02,
        c=c,
        alpha=alpha,
        ipcm=ipcm,
        ipcm_lambda=0.003,
        ipcm_tau=0.2,
        ipcm_window_size=20,
        ipcm_neg_mode="sfns",
        ipcm_sfns_candidates=5,
        ipcm_sfns_rho=0.95,
        ipcm_no_gate=False,
        ipcm_gate=True,
        ipcm_high_freq_only=False,
    )


def load_model(ckpt_path, model_args, device):
    model = BSARecModel(model_args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def pad_sequence(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def make_test_arrays(user_seq, max_len, max_users=0):
    if max_users and max_users > 0:
        user_seq = user_seq[:max_users]
    input_ids, answers, histories = [], [], []
    for seq in user_seq:
        input_ids.append(pad_sequence(seq[:-1], max_len))
        answers.append(seq[-1])
        histories.append(seq[:-1])
    return np.asarray(input_ids, dtype=np.int64), np.asarray(answers, dtype=np.int64), histories


def spectral_vec(x, mask=None, window_size=20, high_freq_only=False, cutoff=0):
    if mask is not None:
        x = x * mask
    if window_size and window_size > 0:
        x = x[:, -window_size:, :]
    freq = torch.fft.rfft(x, dim=1, norm="ortho")
    if high_freq_only and cutoff > 0:
        freq = freq[:, cutoff:, :]
    vec = torch.cat([freq.real, freq.imag], dim=-1).flatten(start_dim=1)
    return F.normalize(vec, dim=-1)


def ndcg_at_rank(rank):
    return 1.0 / np.log2(rank + 2.0)


@torch.no_grad()
def collect_evidence(
    base_model,
    ours_model,
    input_ids_np,
    answers_np,
    histories,
    device,
    batch_size=512,
    topk=20,
    window_size=20,
    sfns_candidates=5,
    sfns_rho=0.95,
    seed=42,
):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    item_size = base_model.args.item_size
    rows = []
    sim_rows = []

    for start in range(0, len(input_ids_np), batch_size):
        end = min(start + batch_size, len(input_ids_np))
        input_ids = torch.as_tensor(input_ids_np[start:end], dtype=torch.long, device=device)
        answers = torch.as_tensor(answers_np[start:end], dtype=torch.long, device=device)
        user_ids = torch.arange(start, end, dtype=torch.long, device=device)

        base_hidden = base_model.predict(input_ids, user_ids)
        ours_hidden = ours_model.predict(input_ids, user_ids)
        base_last = base_hidden[:, -1, :]
        ours_last = ours_hidden[:, -1, :]

        base_scores = torch.matmul(base_last, base_model.item_embeddings.weight.T)
        ours_scores = torch.matmul(ours_last, ours_model.item_embeddings.weight.T)
        base_scores[:, 0] = -1e9
        ours_scores[:, 0] = -1e9
        for local_i, hist in enumerate(histories[start:end]):
            if hist:
                base_scores[local_i, hist] = -1e9
                ours_scores[local_i, hist] = -1e9

        base_top = torch.topk(base_scores, k=topk, dim=1).indices.cpu().numpy()
        ours_top = torch.topk(ours_scores, k=topk, dim=1).indices.cpu().numpy()

        pos_ids = torch.cat([input_ids[:, 1:], answers[:, None]], dim=1)
        hist_mask = (input_ids > 0).float().unsqueeze(-1)
        pos_mask = (pos_ids > 0).float().unsqueeze(-1)
        hist_emb = ours_model.get_continuation_embedding(input_ids) * hist_mask
        pos_emb = ours_model.get_continuation_embedding(pos_ids) * pos_mask

        z_hist = spectral_vec(hist_emb, window_size=window_size)
        z_pos = spectral_vec(pos_emb, window_size=window_size)
        z_base_pred = spectral_vec(base_hidden * pos_mask, window_size=window_size)
        z_ours_pred = spectral_vec(ours_hidden * pos_mask, window_size=window_size)

        shift = (1.0 - torch.sum(z_hist * z_pos, dim=-1)).clamp(0.0, 2.0)
        base_align = torch.sum(z_base_pred * z_pos, dim=-1)
        ours_align = torch.sum(z_ours_pred * z_pos, dim=-1)

        random_neg = torch.randint(1, item_size, answers.shape, generator=rng, device=device)
        random_neg = torch.where(random_neg == answers, (random_neg % (item_size - 1)) + 1, random_neg)
        rand_ids = torch.cat([input_ids[:, 1:], random_neg[:, None]], dim=1)
        rand_emb = ours_model.get_continuation_embedding(rand_ids) * pos_mask
        z_rand = spectral_vec(rand_emb, window_size=window_size)
        rand_sim = torch.sum(z_pos * z_rand, dim=-1)

        candidates = torch.randint(
            1,
            item_size,
            (answers.shape[0], sfns_candidates),
            generator=rng,
            device=device,
        )
        answer_expanded = answers[:, None].expand_as(candidates)
        candidates = torch.where(candidates == answer_expanded, (candidates % (item_size - 1)) + 1, candidates)
        neg_prefix = input_ids[:, 1:].unsqueeze(1).expand(-1, sfns_candidates, -1)
        cand_ids = torch.cat([neg_prefix, candidates.unsqueeze(-1)], dim=-1)
        cand_ids = cand_ids.reshape(-1, input_ids.shape[1])
        cand_mask = pos_mask.unsqueeze(1).expand(-1, sfns_candidates, -1, -1).reshape(-1, input_ids.shape[1], 1)
        cand_emb = ours_model.get_continuation_embedding(cand_ids) * cand_mask
        z_cand = spectral_vec(cand_emb, window_size=window_size).view(answers.shape[0], sfns_candidates, -1)
        cand_sim = torch.sum(z_pos[:, None, :] * z_cand, dim=-1)
        valid = cand_sim < sfns_rho
        hard_idx = torch.argmax(cand_sim.masked_fill(~valid, -1e4), dim=1)
        fallback_idx = torch.argmin(cand_sim, dim=1)
        chosen = torch.where(valid.any(dim=1), hard_idx, fallback_idx)
        sfns_sim = cand_sim[torch.arange(answers.shape[0], device=device), chosen]

        answers_cpu = answers.cpu().numpy()
        shift_cpu = shift.cpu().numpy()
        base_align_cpu = base_align.cpu().numpy()
        ours_align_cpu = ours_align.cpu().numpy()
        rand_sim_cpu = rand_sim.cpu().numpy()
        sfns_sim_cpu = sfns_sim.cpu().numpy()

        for i, ans in enumerate(answers_cpu):
            base_hit_rank = np.where(base_top[i, :10] == ans)[0]
            ours_hit_rank = np.where(ours_top[i, :10] == ans)[0]
            base_hr10 = float(base_hit_rank.size > 0)
            ours_hr10 = float(ours_hit_rank.size > 0)
            base_ndcg10 = ndcg_at_rank(base_hit_rank[0]) if base_hit_rank.size > 0 else 0.0
            ours_ndcg10 = ndcg_at_rank(ours_hit_rank[0]) if ours_hit_rank.size > 0 else 0.0
            rows.append(
                {
                    "user_index": start + i,
                    "answer": int(ans),
                    "seq_len": int(np.count_nonzero(input_ids_np[start + i])),
                    "shift_strength": float(shift_cpu[i]),
                    "fcm_weight": float(shift_cpu[i]),
                    "base_hr10": base_hr10,
                    "ours_hr10": ours_hr10,
                    "base_ndcg10": float(base_ndcg10),
                    "ours_ndcg10": float(ours_ndcg10),
                    "base_pred_pos_sim": float(base_align_cpu[i]),
                    "ours_pred_pos_sim": float(ours_align_cpu[i]),
                    "random_neg_pos_sim": float(rand_sim_cpu[i]),
                    "sfns_neg_pos_sim": float(sfns_sim_cpu[i]),
                }
            )
            sim_rows.append({"type": "Random negative", "similarity": float(rand_sim_cpu[i])})
            sim_rows.append({"type": "SHN negative", "similarity": float(sfns_sim_cpu[i])})

    return pd.DataFrame(rows), pd.DataFrame(sim_rows)


def bootstrap_ci(values, rng, n_boot=1000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0]
    boot = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])


def make_summary(df):
    labels = ["Low shift", "Medium shift", "High shift"]
    df = df.copy()
    try:
        df["shift_group"] = pd.qcut(df["shift_strength"], q=3, labels=labels, duplicates="drop")
    except ValueError:
        df["shift_group"] = pd.cut(df["shift_strength"], bins=3, labels=labels)

    rng = np.random.default_rng(42)
    summary_rows = []
    for group, sub in df.groupby("shift_group", observed=True):
        for method, col in [("BSARec", "base_ndcg10"), ("Our", "ours_ndcg10")]:
            lo, hi = bootstrap_ci(sub[col].values, rng)
            summary_rows.append(
                {
                    "shift_group": str(group),
                    "method": method,
                    "mean_ndcg10": sub[col].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(sub),
                }
            )
    return pd.DataFrame(summary_rows)


def save_all(fig, out_prefix):
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.1, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontweight("bold")
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_size(11)
        ax.title.set_fontname("Times New Roman")
        ax.title.set_fontweight("bold")
        ax.title.set_size(12)
        for text in ax.texts:
            text.set_fontname("Times New Roman")
            text.set_fontweight("bold")
            text.set_size(max(text.get_size(), 9))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname("Times New Roman")
                text.set_fontweight("bold")
                text.set_size(11)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontname("Times New Roman")
        fig._suptitle.set_fontweight("bold")
        fig._suptitle.set_size(min(max(fig._suptitle.get_size(), 12), 13))
    fig.savefig(out_prefix + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(out_prefix + ".svg", bbox_inches="tight")
    plt.close(fig)


def summarize_metric_by_group(df, group_col, group_order=None, metric="ndcg10"):
    base_col = f"base_{metric}"
    ours_col = f"ours_{metric}"
    rng = np.random.default_rng(42)
    rows = []
    grouped = df.groupby(group_col, observed=True)
    groups = group_order if group_order is not None else list(grouped.groups.keys())
    for group in groups:
        if group not in grouped.groups:
            continue
        sub = grouped.get_group(group)
        for method, col in [("BSARec", base_col), ("Our", ours_col)]:
            lo, hi = bootstrap_ci(sub[col].values, rng)
            rows.append(
                {
                    "group": str(group),
                    "method": method,
                    "mean": sub[col].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(sub),
                }
            )
        gain = sub[ours_col] - sub[base_col]
        lo, hi = bootstrap_ci(gain.values, rng)
        rows.append(
            {
                "group": str(group),
                "method": "Gain",
                "mean": gain.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def add_shift_group(df):
    labels = ["Low shift", "Medium shift", "High shift"]
    df = df.copy()
    df["shift_group"] = pd.qcut(df["shift_strength"].rank(method="first"), q=3, labels=labels)
    return df, labels


def add_length_group(df):
    labels = ["Short", "Medium", "Long"]
    df = df.copy()
    df["length_group"] = pd.qcut(df["seq_len"].rank(method="first"), q=3, labels=labels)
    return df, labels


def add_fcm_group(df):
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    df = df.copy()
    df["fcm_group"] = pd.qcut(df["fcm_weight"].rank(method="first"), q=5, labels=labels)
    return df, labels


def make_fcm_weight_summary(df):
    df, order = add_fcm_group(df)
    rng = np.random.default_rng(42)
    rows = []
    grouped = df.groupby("fcm_group", observed=True)
    for group in order:
        if group not in grouped.groups:
            continue
        sub = grouped.get_group(group)
        gain = sub["ours_ndcg10"] - sub["base_ndcg10"]
        lo, hi = bootstrap_ci(gain.values, rng)
        rows.append(
            {
                "group": str(group),
                "fcm_weight": sub["fcm_weight"].mean(),
                "gain": gain.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def colors():
    return {
        "BSARec": "#5b677a",
        "Our": "#d08159",
        "Gain": "#4f8a6b",
        "Random negative": "#8fa7c2",
        "SHN negative": "#c7665a",
    }


def plot_shift_performance(df, out_prefix):
    palette = colors()
    df, order = add_shift_group(df)
    summary = summarize_metric_by_group(df, "shift_group", order)
    perf = summary[summary["method"].isin(["BSARec", "Our"])]
    gain = summary[summary["method"] == "Gain"].set_index("group").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.55, 1.0]})
    x = np.arange(len(order))
    width = 0.34
    for offset, method in [(-width / 2, "BSARec"), (width / 2, "Our")]:
        sub = perf[perf["method"] == method].set_index("group").loc[order]
        y = sub["mean"].values
        err = np.vstack([y - sub["ci_low"].values, sub["ci_high"].values - y])
        axes[0].bar(x + offset, y, width, color=palette[method], label=method, edgecolor="none")
        axes[0].errorbar(x + offset, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Performance by interest-shift strength", loc="left", fontweight="bold")
    axes[0].legend(ncol=2, loc="upper right")

    y = gain["mean"].values * 100.0
    err = np.vstack([(gain["mean"] - gain["ci_low"]).values, (gain["ci_high"] - gain["mean"]).values]) * 100.0
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].bar(x, y, width=0.55, color=palette["Gain"], edgecolor="none")
    axes[1].errorbar(x, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    for xi, yi in zip(x, y):
        axes[1].text(xi, yi + (0.08 if yi >= 0 else -0.08), f"{yi:+.2f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("Our - BSARec\nNDCG@10 (pp)")
    axes[1].set_title("Gain view", loc="left", fontweight="bold")
    save_all(fig, out_prefix)
    return summary


def plot_frequency_alignment(df, out_prefix):
    palette = colors()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    bp = ax.boxplot(
        [df["base_pred_pos_sim"].values, df["ours_pred_pos_sim"].values],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        tick_labels=["BSARec", "Our"],
    )
    for patch, color in zip(bp["boxes"], [palette["BSARec"], palette["Our"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.0)
    ax.set_ylabel(r"$\cos(z_{pred}, z_{pos})$")
    ax.set_title("Frequency alignment to target continuation", loc="left", fontweight="bold")
    save_all(fig, out_prefix)


def plot_negative_similarity(df, out_prefix, rho=0.95):
    palette = colors()
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    bins = np.linspace(
        min(df["random_neg_pos_sim"].min(), df["sfns_neg_pos_sim"].min()),
        max(df["random_neg_pos_sim"].max(), df["sfns_neg_pos_sim"].max()),
        36,
    )
    ax.hist(df["random_neg_pos_sim"].values, bins=bins, density=True, color=palette["Random negative"], alpha=0.55, label="Random negative")
    ax.hist(df["sfns_neg_pos_sim"].values, bins=bins, density=True, color=palette["SHN negative"], alpha=0.55, label="SHN negative")
    random_over = (df["random_neg_pos_sim"].values >= rho).mean() * 100.0
    shn_over = (df["sfns_neg_pos_sim"].values >= rho).mean() * 100.0
    ax.axvline(rho, color="#333333", linestyle="--", linewidth=0.9)
    ax.text(rho + 0.001, ax.get_ylim()[1] * 0.88, r"$\rho=0.95$", ha="left", va="top", fontsize=7, color="#333333")
    ax.text(0.02, 0.78, f">=rho: random {random_over:.1f}%\n>=rho: SHN {shn_over:.1f}%", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#333333")
    ax.set_xlabel(r"$\cos(z_{pos}, z_{neg})$")
    ax.set_ylabel("Density")
    ax.set_title("SHN keeps hard negatives below the false-negative threshold", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    save_all(fig, out_prefix)


def plot_length_performance(df, out_prefix):
    palette = colors()
    df, order = add_length_group(df)
    summary = summarize_metric_by_group(df, "length_group", order)
    perf = summary[summary["method"].isin(["BSARec", "Our"])]
    gain = summary[summary["method"] == "Gain"].set_index("group").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.55, 1.0]})
    x = np.arange(len(order))
    width = 0.34
    for offset, method in [(-width / 2, "BSARec"), (width / 2, "Our")]:
        sub = perf[perf["method"] == method].set_index("group").loc[order]
        y = sub["mean"].values
        err = np.vstack([y - sub["ci_low"].values, sub["ci_high"].values - y])
        axes[0].bar(x + offset, y, width, color=palette[method], label=method, edgecolor="none")
        axes[0].errorbar(x + offset, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("NDCG@10")
    axes[0].set_title("Performance by sequence length", loc="left", fontweight="bold")
    axes[0].legend(ncol=2, loc="upper right")

    y = gain["mean"].values * 100.0
    err = np.vstack([(gain["mean"] - gain["ci_low"]).values, (gain["ci_high"] - gain["mean"]).values]) * 100.0
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].bar(x, y, width=0.55, color=palette["Gain"], edgecolor="none")
    axes[1].errorbar(x, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    for xi, yi in zip(x, y):
        axes[1].text(xi, yi + (0.08 if yi >= 0 else -0.08), f"{yi:+.2f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("Our - BSARec\nNDCG@10 (pp)")
    axes[1].set_title("Gain view", loc="left", fontweight="bold")
    save_all(fig, out_prefix)
    return summary


def plot_fcm_weight_gain(df, out_prefix):
    palette = colors()
    summary = make_fcm_weight_summary(df)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(summary))
    y = summary["gain"].values * 100.0
    err = np.vstack([(summary["gain"] - summary["ci_low"]).values, (summary["ci_high"] - summary["gain"]).values]) * 100.0
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.plot(x, y, color=palette["Gain"], marker="o", linewidth=1.6)
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{w:.2f}" for g, w in zip(summary["group"], summary["fcm_weight"])])
    ax.set_xlabel("FCM weight quantile\n(mean weight)")
    ax.set_ylabel("Our - BSARec\nNDCG@10 (pp)")
    ax.set_title("Performance gain along FCM-weighted interest novelty", loc="left", fontweight="bold")
    save_all(fig, out_prefix)
    return summary


def plot_overview(df, out_prefix):
    colors = {
        "BSARec": "#5b677a",
        "Our": "#d08159",
        "Random negative": "#8fa7c2",
        "SHN negative": "#c7665a",
        "Gain": "#4f8a6b",
    }
    df, shift_order = add_shift_group(df)
    shift_summary = summarize_metric_by_group(df, "shift_group", shift_order)

    fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[0, 1])

    groups = shift_order
    x = np.arange(len(groups))
    width = 0.34
    perf = shift_summary[shift_summary["method"].isin(["BSARec", "Our"])]
    for offset, method in [(-width / 2, "BSARec"), (width / 2, "Our")]:
        sub = perf[perf["method"] == method].set_index("group").loc[groups]
        y = sub["mean"].values
        err = np.vstack([y - sub["ci_low"].values, sub["ci_high"].values - y])
        ax1.bar(x + offset, y, width, color=colors[method], label=method, edgecolor="none")
        ax1.errorbar(x + offset, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("NDCG@10")
    ax1.set_title("A. Interest shift", loc="left", fontweight="bold")
    ax1.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.96),
        columnspacing=0.6,
        handletextpad=0.35,
        borderaxespad=0.1,
    )

    gain = shift_summary[shift_summary["method"] == "Gain"].set_index("group").loc[groups]
    ax4.axhline(0, color="#444444", linewidth=0.8)
    ax4.bar(x, gain["mean"].values * 100.0, color=colors["Gain"], edgecolor="none")
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.set_ylabel("NDCG@10 gain (pp)")
    ax4.set_title("B. Gain view", loc="left", fontweight="bold")

    bp = ax2.boxplot(
        [df["base_pred_pos_sim"].values, df["ours_pred_pos_sim"].values],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        tick_labels=["BSARec", "Our"],
    )
    for patch, color in zip(bp["boxes"], [colors["BSARec"], colors["Our"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.0)
    ax2.set_ylabel(r"$\cos(z_{pred}, z_{pos})$")
    ax2.set_title("C. FCM alignment", loc="left", fontweight="bold")

    bins = np.linspace(
        min(df["random_neg_pos_sim"].min(), df["sfns_neg_pos_sim"].min()),
        max(df["random_neg_pos_sim"].max(), df["sfns_neg_pos_sim"].max()),
        36,
    )
    ax3.hist(
        df["random_neg_pos_sim"].values,
        bins=bins,
        density=True,
        color=colors["Random negative"],
        alpha=0.55,
        label="Random negative",
    )
    ax3.hist(
        df["sfns_neg_pos_sim"].values,
        bins=bins,
        density=True,
        color=colors["SHN negative"],
        alpha=0.55,
        label="SHN negative",
    )
    rho = 0.95
    random_over = (df["random_neg_pos_sim"].values >= rho).mean() * 100.0
    sfns_over = (df["sfns_neg_pos_sim"].values >= rho).mean() * 100.0
    ax3.axvline(rho, color="#333333", linestyle="--", linewidth=0.9)
    ax3.text(
        rho + 0.001,
        ax3.get_ylim()[1] * 0.88,
        r"$\rho=0.95$",
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    ax3.text(
        0.03,
        0.58,
        f">=rho: random {random_over:.1f}%\n>=rho: SHN {sfns_over:.1f}%",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    ax3.set_xlabel(r"$\cos(z_{pos}, z_{neg})$")
    ax3.set_ylabel("Density")
    ax3.set_title("D. SHN thresholding", loc="left", fontweight="bold")
    ax3.legend(loc="upper left")

    save_all(fig, out_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "Toys_and_Games.txt"))
    parser.add_argument("--base-ckpt", default=os.path.join(os.path.dirname(ROOT), "BSARec_Toys_best.pt"))
    parser.add_argument("--ours-ckpt", default=os.path.join(os.path.dirname(ROOT), "BSARec_iPCMv3_SFNS_Toys_final.pt"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(ROOT)), "frequency_sr_report", "figures"))
    parser.add_argument("--max-users", type=int, default=0, help="0 means all users; use a smaller value for quick debugging.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    device = torch.device(device)

    user_seq, max_item, _ = get_user_seqs(args.data)
    item_size = max_item + 1
    input_ids, answers, histories = make_test_arrays(user_seq, 50, args.max_users)

    base_args = build_args(item_size=item_size, c=3, alpha=0.7, ipcm=False)
    ours_args = build_args(item_size=item_size, c=7, alpha=0.3, ipcm=True)
    base_model = load_model(args.base_ckpt, base_args, device)
    ours_model = load_model(args.ours_ckpt, ours_args, device)

    df, sim_df = collect_evidence(
        base_model,
        ours_model,
        input_ids,
        answers,
        histories,
        device,
        batch_size=args.batch_size,
        window_size=ours_args.ipcm_window_size,
        sfns_candidates=ours_args.ipcm_sfns_candidates,
        sfns_rho=ours_args.ipcm_sfns_rho,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, "toys_fcm_shn_evidence")
    df.to_csv(out_prefix + "_source.csv", index=False)
    sim_df.to_csv(out_prefix + "_negative_similarity_long.csv", index=False)
    shift_summary = plot_shift_performance(df, os.path.join(args.out_dir, "toys_fcm_shn_shift_performance"))
    plot_frequency_alignment(df, os.path.join(args.out_dir, "toys_fcm_alignment"))
    plot_negative_similarity(df, os.path.join(args.out_dir, "toys_shn_negative_similarity"))
    length_summary = plot_length_performance(df, os.path.join(args.out_dir, "toys_fcm_shn_length_performance"))
    fcm_summary = plot_fcm_weight_gain(df, os.path.join(args.out_dir, "toys_fcm_weight_gain"))
    plot_overview(df, out_prefix)
    shift_summary.to_csv(out_prefix + "_shift_group_summary.csv", index=False)
    length_summary.to_csv(out_prefix + "_length_group_summary.csv", index=False)
    fcm_summary.to_csv(out_prefix + "_fcm_weight_summary.csv", index=False)

    print("Saved:")
    for name in [
        "toys_fcm_shn_shift_performance",
        "toys_fcm_alignment",
        "toys_shn_negative_similarity",
        "toys_fcm_shn_length_performance",
        "toys_fcm_weight_gain",
        "toys_fcm_shn_evidence",
    ]:
        print(os.path.join(args.out_dir, name + ".png"))
    print(out_prefix + "_source.csv")
    print(out_prefix + "_shift_group_summary.csv")
    print(out_prefix + "_length_group_summary.csv")
    print(out_prefix + "_fcm_weight_summary.csv")


if __name__ == "__main__":
    main()
