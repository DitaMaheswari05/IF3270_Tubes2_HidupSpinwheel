import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")

def evaluate_keras(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    f1     = macro_f1(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"y_pred": y_pred, "f1": f1, "report": report}

def evaluate_scratch(scratch_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = scratch_model.predict(X_test)
    f1     = macro_f1(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"y_pred": y_pred, "f1": f1, "report": report}

def compare_keras_vs_scratch(
    keras_result: dict,
    scratch_result: dict,
    class_names: list = None,
    save_dir: str = "plots",
):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Keras  macro F1 : {:.4f}".format(keras_result["f1"]))
    print("Scratch macro F1: {:.4f}".format(scratch_result["f1"]))
    delta = abs(keras_result["f1"] - scratch_result["f1"])
    print(f"Selisih         : {delta:.4f}  {'(implementasi akurat)' if delta < 0.01 else '(ada perbedaan)'}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, result, title in zip(
        axes,
        [keras_result, scratch_result],
        ["Keras", "From Scratch"],
    ):
        cm  = confusion_matrix(result["y_pred"], result["y_pred"])
        ax.set_title(f"{title} — macro F1: {result['f1']:.4f}")
        ax.axis("off")

    # Bar chart F1 per class
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    labels = class_names or [f"class_{i}" for i in range(
        len(keras_result["report"]) - 3
    )]
    keras_f1s   = [keras_result["report"].get(str(i), {}).get("f1-score", 0)   for i in range(len(labels))]
    scratch_f1s = [scratch_result["report"].get(str(i), {}).get("f1-score", 0) for i in range(len(labels))]

    x = np.arange(len(labels))
    ax2.bar(x - 0.2, keras_f1s,   0.4, label="Keras",   color="steelblue")
    ax2.bar(x + 0.2, scratch_f1s, 0.4, label="Scratch", color="tomato")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("F1-Score")
    ax2.set_title("Per-Class F1: Keras vs From Scratch")
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "keras_vs_scratch_f1.png"), dpi=150)
    plt.close(fig2)
    print(f"[Plot saved] keras_vs_scratch_f1.png")

def plot_history(
    history: dict,
    name: str,
    save_dir: str = "plots",
):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, ylabel in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"],
    ):
        train_key, val_key = metric
        if train_key in history:
            ax.plot(history[train_key],     label="Train")
        if val_key in history:
            ax.plot(history[val_key], "--", label="Validation")
        ax.set_title(f"{name} — {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot saved] {path}")


def plot_all_histories(all_results: list, logs_dir: str = "logs", save_dir: str = "plots"):
    for r in all_results:
        hist = r.get("history")
        if hist is None:
            log_path = os.path.join(logs_dir, f"{r['name']}_history.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    hist = json.load(f)
        if hist is not None:
            plot_history(hist, r["name"], save_dir)

def _parse_name(name: str) -> dict:
    parts = name.split("_")
    return {
        "num_conv": int(parts[0].replace("conv", "")),
        "filters":  parts[1].replace("f", ""),
        "kernel":   int(parts[2].replace("k", "")),
        "pool":     parts[3],
    }


def analyze_hyperparameter_effect(
    all_results: list,
    save_dir: str = "plots",
):
    os.makedirs(save_dir, exist_ok=True)

    parsed = [(r["best_val_f1"], _parse_name(r["name"])) for r in all_results]

    for param_key, param_name in [
        ("num_conv", "Jumlah Conv Layer"),
        ("filters",  "Banyak Filter"),
        ("kernel",   "Ukuran Filter (kernel)"),
        ("pool",     "Jenis Pooling"),
    ]:
        groups = {}
        for f1, p in parsed:
            key = str(p[param_key])
            groups.setdefault(key, []).append(f1)

        avg_f1 = {k: np.mean(v) for k, v in groups.items()}

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(avg_f1.keys())
        values = [avg_f1[k] for k in labels]
        bars = ax.bar(labels, values, color=["steelblue", "tomato", "seagreen", "gold"][:len(labels)])
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(f"Pengaruh {param_name} terhadap Macro F1")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Rata-Rata Macro F1 (val)")
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        path = os.path.join(save_dir, f"hp_effect_{param_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Plot saved] {path}")

def compare_shared_vs_nonshared(
    keras_shared_model,
    keras_nonshared_model,
    scratch_shared,
    scratch_nonshared,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = "plots",
):
    os.makedirs(save_dir, exist_ok=True)

    n_params_shared    = keras_shared_model.count_params()
    n_params_nonshared = keras_nonshared_model.count_params()

    f1_keras_shared    = macro_f1(y_test, np.argmax(keras_shared_model.predict(X_test, verbose=0), axis=-1))
    f1_keras_nonshared = macro_f1(y_test, np.argmax(keras_nonshared_model.predict(X_test, verbose=0), axis=-1))
    f1_scratch_shared    = macro_f1(y_test, scratch_shared.predict(X_test))
    f1_scratch_nonshared = macro_f1(y_test, scratch_nonshared.predict(X_test))

    print("\n" + "=" * 55)
    print(f"{'Model':<30} {'#Params':>10} {'F1':>8}")
    print("=" * 55)
    print(f"{'Keras Shared':<30} {n_params_shared:>10,} {f1_keras_shared:>8.4f}")
    print(f"{'Keras Non-Shared':<30} {n_params_nonshared:>10,} {f1_keras_nonshared:>8.4f}")
    print(f"{'Scratch Shared':<30} {'—':>10} {f1_scratch_shared:>8.4f}")
    print(f"{'Scratch Non-Shared':<30} {'—':>10} {f1_scratch_nonshared:>8.4f}")
    print("=" * 55)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # F1 comparison
    labels_f1 = ["Keras\nShared", "Keras\nNon-Shared", "Scratch\nShared", "Scratch\nNon-Shared"]
    f1_vals   = [f1_keras_shared, f1_keras_nonshared, f1_scratch_shared, f1_scratch_nonshared]
    bars = axes[0].bar(labels_f1, f1_vals, color=["steelblue", "tomato", "steelblue", "tomato"], alpha=0.8)
    axes[0].bar_label(bars, fmt="%.4f", padding=3)
    axes[0].set_title("Macro F1: Shared vs Non-Shared")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 1.1)

    # Parameter count
    labels_p = ["Shared", "Non-Shared"]
    param_vals = [n_params_shared, n_params_nonshared]
    bars2 = axes[1].bar(labels_p, param_vals, color=["steelblue", "tomato"])
    axes[1].bar_label(bars2, fmt="%d", padding=3)
    axes[1].set_title("Jumlah Parameter")
    axes[1].set_ylabel("# Parameters")

    plt.tight_layout()
    path = os.path.join(save_dir, "shared_vs_nonshared.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot saved] {path}")

    return {
        "f1_keras_shared":    f1_keras_shared,
        "f1_keras_nonshared": f1_keras_nonshared,
        "f1_scratch_shared":    f1_scratch_shared,
        "f1_scratch_nonshared": f1_scratch_nonshared,
        "params_shared":    n_params_shared,
        "params_nonshared": n_params_nonshared,
    }

def visualize_predictions(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    n_samples: int = 10,
    save_dir: str = "plots",
    title: str = "predictions",
):
    os.makedirs(save_dir, exist_ok=True)

    idxs = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    ncols = 5
    nrows = (len(idxs) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for ax_i, idx in enumerate(idxs):
        ax = axes[ax_i]
        ax.imshow(X_test[idx])
        ax.axis("off")
        true_lbl = class_names[y_true[idx]]
        pred_lbl = class_names[y_pred[idx]]
        color    = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=8, color=color)

    for ax in axes[len(idxs):]:
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Plot saved] {path}")