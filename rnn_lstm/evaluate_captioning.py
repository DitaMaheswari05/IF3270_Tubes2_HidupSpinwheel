# bisa langsung diimport ke notebook untuk evaluasi & plotting hasil captioning
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from collections import defaultdict
from rnn_lstm.train_captioning import generate_caption_greedy


def _ensure_nltk():
    for pkg in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# BLEU-4 & METEOR

def compute_bleu4(hypotheses: list, references: list) -> float:
    sf   = SmoothingFunction().method1
    refs = [[r.lower().split() for r in ref_group] for ref_group in references]
    hyps = [h.lower().split() for h in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=sf)


def compute_meteor(hypotheses: list, references: list) -> float:
    _ensure_nltk()
    scores = []
    for hyp, refs in zip(hypotheses, references):
        h = hyp.lower()
        r_list = [ref.lower() for ref in refs]
        score = meteor_score(r_list, h)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


# caption generation helper

def keras_generate_batch(
    model,
    features: dict,
    captions: dict,
    word_to_idx: dict,
    idx_to_word: dict,
    max_len: int = 34,
    n_images: int = None,
) -> dict:
    fnames = [f for f in features if f in captions]
    if n_images:
        fnames = fnames[:n_images]

    t0 = time.time()
    hyps, refs = [], []
    for i, fname in enumerate(fnames):
        cap = generate_caption_greedy(model, features[fname], word_to_idx, idx_to_word, max_len)
        hyps.append(cap)
        refs.append(captions[fname])
        if (i + 1) % 100 == 0:
            print(f"  [Keras generate] {i+1}/{len(fnames)}", end="\r")
    print()
    elapsed = time.time() - t0

    return {"hypotheses": hyps, "references": refs, "fnames": fnames, "time_s": elapsed}


# a) Variasi jumlah layer & hidden state

def analyze_hyperparameter_variations(
    all_results: list,
    save_dir: str = "plots/captioning",
):
    os.makedirs(save_dir, exist_ok=True)

    for rnn_type in ("rnn", "lstm"):
        subset = [r for r in all_results if r.get("rnn_type") == rnn_type]
        if not subset:
            continue

        # bar chart: BLEU-4 per config
        names   = [r["name"] for r in subset]
        bleu4s  = [r.get("bleu4", 0.0) for r in subset]
        meteors = [r.get("meteor", 0.0) for r in subset]

        x = np.arange(len(names))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(x, bleu4s,  color="steelblue")
        axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axes[0].set_title(f"{rnn_type.upper()} — BLEU-4 per Konfigurasi")
        axes[0].set_ylabel("BLEU-4"); axes[0].set_ylim(0, max(bleu4s + [0.01]) * 1.2)

        axes[1].bar(x, meteors, color="tomato")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axes[1].set_title(f"{rnn_type.upper()} — METEOR per Konfigurasi")
        axes[1].set_ylabel("METEOR"); axes[1].set_ylim(0, max(meteors + [0.01]) * 1.2)

        plt.tight_layout()
        path = os.path.join(save_dir, f"{rnn_type}_hyperparams_bar.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"[Plot saved] {path}")

        by_layers  = defaultdict(list)
        by_hidden  = defaultdict(list)
        for r in subset:
            by_layers[r["num_layers"]].append(r.get("bleu4", 0))
            by_hidden[r["hidden_units"]].append(r.get("bleu4", 0))

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, groups, xlabel, title_sfx in [
            (axes2[0], by_layers, "Jumlah Layer", "Pengaruh Jumlah Layer"),
            (axes2[1], by_hidden, "Hidden Units",  "Pengaruh Hidden Units"),
        ]:
            keys   = sorted(groups.keys())
            means  = [np.mean(groups[k]) for k in keys]
            bars   = ax.bar([str(k) for k in keys], means, color="seagreen")
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_title(f"{rnn_type.upper()} — {title_sfx}")
            ax.set_xlabel(xlabel); ax.set_ylabel("Rata-Rata BLEU-4")
            ax.set_ylim(0, max(means + [0.01]) * 1.3)
        plt.tight_layout()
        path2 = os.path.join(save_dir, f"{rnn_type}_hyperparams_effect.png")
        fig2.savefig(path2, dpi=150); plt.close(fig2)
        print(f"[Plot saved] {path2}")


# training curves

def plot_training_curves(
    all_results: list,
    save_dir: str = "plots/captioning",
):
    os.makedirs(save_dir, exist_ok=True)
    for r in all_results:
        hist = r.get("history")
        if hist is None:
            log = f"logs/captioning/{r['name']}_history.json"
            if os.path.exists(log):
                with open(log) as f:
                    hist = json.load(f)
        if hist is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (tk, vk), ylabel in [
            (axes[0], ("loss",     "val_loss"),     "Loss"),
            (axes[1], ("accuracy", "val_accuracy"), "Accuracy"),
        ]:
            if tk in hist: ax.plot(hist[tk],     label="Train")
            if vk in hist: ax.plot(hist[vk], "--", label="Validation")
            ax.set_title(f"{r['name']} — {ylabel}")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
        plt.tight_layout()
        path = os.path.join(save_dir, f"{r['name']}_curve.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"[Plot saved] {path}")


# b) Keras vs Scratch comparison

def compare_keras_vs_scratch(
    keras_hyps: list,    keras_refs: list,  keras_time: float,
    scratch_hyps: list,  scratch_refs: list, scratch_time: float,
    rnn_type: str = "lstm",
    save_dir: str = "plots/captioning",
) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    bleu4_keras   = compute_bleu4(keras_hyps, keras_refs)
    bleu4_scratch = compute_bleu4(scratch_hyps, scratch_refs)
    met_keras     = compute_meteor(keras_hyps, keras_refs)
    met_scratch   = compute_meteor(scratch_hyps, scratch_refs)

    print(f"\n{'='*55}")
    print(f"{'Model':<25} {'BLEU-4':>8} {'METEOR':>8} {'Time(s)':>10}")
    print("=" * 55)
    print(f"{'Keras '+rnn_type.upper():<25} {bleu4_keras:>8.4f} {met_keras:>8.4f} {keras_time:>10.1f}")
    print(f"{'Scratch '+rnn_type.upper():<25} {bleu4_scratch:>8.4f} {met_scratch:>8.4f} {scratch_time:>10.1f}")
    print("=" * 55)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"Keras\n{rnn_type.upper()}", f"Scratch\n{rnn_type.upper()}"]
    b4s    = [bleu4_keras, bleu4_scratch]
    bars   = ax.bar(labels, b4s, color=["steelblue", "tomato"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title(f"BLEU-4: Keras vs From-Scratch ({rnn_type.upper()})")
    ax.set_ylabel("BLEU-4"); ax.set_ylim(0, max(b4s + [0.01]) * 1.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{rnn_type}_keras_vs_scratch.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[Plot saved] {path}")

    return {"bleu4_keras": bleu4_keras, "bleu4_scratch": bleu4_scratch,
            "meteor_keras": met_keras, "meteor_scratch": met_scratch}


# c) RNN vs LSTM

def compare_rnn_vs_lstm(
    rnn_hyps:  list, rnn_refs:  list, rnn_time:  float,
    lstm_hyps: list, lstm_refs: list, lstm_time: float,
    image_features: dict,
    image_captions: dict,
    fnames_test: list,
    n_qualitative: int = 10,
    save_dir: str = "plots/captioning",
) -> dict:
    os.makedirs(save_dir, exist_ok=True)

    bleu4_rnn  = compute_bleu4(rnn_hyps, rnn_refs)
    bleu4_lstm = compute_bleu4(lstm_hyps, lstm_refs)
    met_rnn    = compute_meteor(rnn_hyps, rnn_refs)
    met_lstm   = compute_meteor(lstm_hyps, lstm_refs)

    print(f"\n{'='*55}")
    print(f"{'Model':<20} {'BLEU-4':>8} {'METEOR':>8} {'Time(s)':>10}")
    print("=" * 55)
    print(f"{'SimpleRNN':<20} {bleu4_rnn:>8.4f} {met_rnn:>8.4f} {rnn_time:>10.1f}")
    print(f"{'LSTM':<20} {bleu4_lstm:>8.4f} {met_lstm:>8.4f} {lstm_time:>10.1f}")
    print("=" * 55)

    # bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (rv, lv, ylabel) in zip(axes, [
        (bleu4_rnn, bleu4_lstm, "BLEU-4"),
        (met_rnn,   met_lstm,   "METEOR"),
        (rnn_time,  lstm_time,  "Time (s)"),
    ]):
        bars = ax.bar(["SimpleRNN", "LSTM"], [rv, lv], color=["steelblue", "tomato"])
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(ylabel); ax.set_ylabel(ylabel)
        ax.set_ylim(0, max([rv, lv, 0.01]) * 1.3)
    plt.suptitle("RNN vs LSTM: Quantitative Comparison")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "rnn_vs_lstm_quantitative.png"), dpi=150)
    plt.close(fig)



    sf = SmoothingFunction().method1
    def per_image_bleu(hyp, refs):
        refs_tok = [r.lower().split() for r in refs]
        hyp_tok  = hyp.lower().split()
        return corpus_bleu([refs_tok], [hyp_tok], smoothing_function=sf)

    n = min(len(fnames_test), len(rnn_hyps), len(lstm_hyps))
    per_bleu = [(per_image_bleu(lstm_hyps[i], lstm_refs[i]), i) for i in range(n)]
    per_bleu.sort()

    step = max(1, n // n_qualitative)
    idxs = [per_bleu[j * step][1] for j in range(n_qualitative)]

    fig2, axes2 = plt.subplots(n_qualitative, 1, figsize=(14, n_qualitative * 3.5))
    if n_qualitative == 1:
        axes2 = [axes2]

    for ax, idx in zip(axes2, idxs):
        fname = fnames_test[idx]
        rnn_cap  = rnn_hyps[idx]
        lstm_cap = lstm_hyps[idx]
        refs_str = " / ".join(image_captions.get(fname, ["N/A"])[:2])

        ax.axis("off")
        ax.text(0.01, 0.8, f"File: {fname}", fontsize=8, transform=ax.transAxes)
        ax.text(0.01, 0.6, f"RNN  : {rnn_cap}",  fontsize=9, color="steelblue", transform=ax.transAxes)
        ax.text(0.01, 0.4, f"LSTM : {lstm_cap}", fontsize=9, color="tomato",    transform=ax.transAxes)
        ax.text(0.01, 0.15, f"GT   : {refs_str}", fontsize=8, color="gray",      transform=ax.transAxes, wrap=True)

    plt.suptitle("Qualitative Caption Comparison: RNN vs LSTM (10 samples)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    q_path = os.path.join(save_dir, "rnn_vs_lstm_qualitative.png")
    fig2.savefig(q_path, dpi=100, bbox_inches="tight"); plt.close(fig2)
    print(f"[Plot saved] {q_path}")

    return {
        "bleu4_rnn": bleu4_rnn, "bleu4_lstm": bleu4_lstm,
        "meteor_rnn": met_rnn,  "meteor_lstm": met_lstm,
    }


# d) Pengaruh panjang maksimum caption

def analyze_max_len_effect(
    model,
    image_features: dict,
    image_captions: dict,
    word_to_idx: dict,
    idx_to_word: dict,
    max_len_values: list = (15, 25, 34),
    is_scratch: bool = False,
    model_name: str = "model",
    save_dir: str = "plots/captioning",
    n_images: int = 500,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for ml in max_len_values:
        fnames = [f for f in image_features if f in image_captions][:n_images]

        if is_scratch:
            hyps = model.generate_batch(
                np.array([image_features[f] for f in fnames]),
                word_to_idx, idx_to_word, max_len=ml,
            )
        else:
            from train_captioning import generate_caption_greedy
            hyps = [generate_caption_greedy(model, image_features[f], word_to_idx, idx_to_word, ml)
                    for f in fnames]

        refs    = [image_captions[f] for f in fnames]
        bleu4   = compute_bleu4(hyps, refs)
        results[ml] = bleu4
        print(f"  max_len={ml:3d}  BLEU-4={bleu4:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = sorted(results.keys())
    vals = [results[k] for k in keys]
    ax.plot(keys, vals, "o-", color="steelblue", linewidth=2, markersize=8)
    for k, v in zip(keys, vals):
        ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_title(f"Pengaruh Panjang Maksimum Caption terhadap BLEU-4 ({model_name})")
    ax.set_xlabel("Panjang Maksimum Caption (max_len)")
    ax.set_ylabel("BLEU-4")
    ax.set_ylim(0, max(vals + [0.01]) * 1.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_max_len_effect.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[Plot saved] {path}")

    return results