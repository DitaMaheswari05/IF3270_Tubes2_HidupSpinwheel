"""
Variasi yang dilakukan sesuai spek (total 16 arsitektur):
  A. Jumlah layer konvolusi  : 2 variasi  (mis. 2 vs 4 conv layers)
  B. Banyak filter per layer  : 2 variasi  (mis. [32,64] vs [64,128])
  C. Ukuran filter per layer  : 2 variasi  (mis. 3x3 vs 5x5)
  D. Jenis pooling layer      : 2 variasi  (MaxPooling vs AveragePooling)

Total kombinasi: 2 × 2 × 2 × 2 = 16 arsitektur.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL
from sklearn.metrics import f1_score

HYPERPARAMETER_GRID = {
    "num_conv_layers": [2, 4],
    "filters": [[32, 64], [64, 128]],
    "kernel_size": [3, 5],
    "pooling_type": ["max", "average"],
}

# Konfigurasi umum
IMG_SIZE    = (150, 150)
NUM_CLASSES = 6
EPOCHS      = 20
BATCH_SIZE  = 32
WEIGHTS_DIR = "weights"
LOGS_DIR    = "logs"

def build_cnn_model(
    num_conv_layers: int,
    filters: list,
    kernel_size: int,
    pooling_type: str,
    num_classes: int = NUM_CLASSES,
    img_size: tuple = IMG_SIZE,
    use_locally_connected: bool = False,
) -> keras.Model:
    if len(filters) < num_conv_layers:
        filters = (filters * (num_conv_layers // len(filters) + 1))[:num_conv_layers]
    else:
        filters = filters[:num_conv_layers]

    PoolLayer = KL.MaxPooling2D if pooling_type == "max" else KL.AveragePooling2D

    inp = keras.Input(shape=(*img_size, 3), name="input")
    x = inp

    for i in range(num_conv_layers):
        if use_locally_connected:
            x = KL.LocallyConnected2D(
                filters[i], kernel_size=kernel_size,
                activation="relu",
                name=f"lc2d_{i+1}",
            )(x)
        else:
            x = KL.Conv2D(
                filters[i], kernel_size=kernel_size,
                padding="same", activation="relu",
                name=f"conv2d_{i+1}",
            )(x)

        if (i + 1) % 2 == 0 or i == num_conv_layers - 1:
            x = PoolLayer(pool_size=2, name=f"pool_{i+1}")(x)

    x = KL.Flatten(name="flatten")(x)
    x = KL.Dense(128, activation="relu", name="dense_1")(x)
    x = KL.Dropout(0.5, name="dropout")(x)
    out = KL.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.Model(inputs=inp, outputs=out)

def experiment_name(num_conv_layers, filters, kernel_size, pooling_type) -> str:
    f_str = "-".join(str(f) for f in filters[:num_conv_layers])
    return f"conv{num_conv_layers}_f{f_str}_k{kernel_size}_{pooling_type}"

def train_one(
    name: str,
    model: keras.Model,
    X_train, y_train,
    X_val,   y_val,
    epochs: int   = EPOCHS,
    batch_size: int = BATCH_SIZE,
    weights_dir: str = WEIGHTS_DIR,
    logs_dir: str    = LOGS_DIR,
) -> dict:
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    weight_path = os.path.join(weights_dir, f"{name}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            weight_path, save_best_only=True,
            monitor="val_accuracy", verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluasi macro F1
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=-1)
    val_f1 = f1_score(y_val, y_pred, average="macro")

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    log_path  = os.path.join(logs_dir, f"{name}_history.json")
    with open(log_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    print(f"  [{name}] Best val macro-F1: {val_f1:.4f} | Saved: {weight_path}")

    return {
        "name":       name,
        "model":      model,
        "history":    hist_dict,
        "best_val_f1": val_f1,
        "weight_path": weight_path,
    }

def train_all_experiments(
    X_train, y_train,
    X_val,   y_val,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    weights_dir: str = WEIGHTS_DIR,
    logs_dir: str    = LOGS_DIR,
    skip_existing: bool = True,
) -> list:
    all_results = []
    grid = HYPERPARAMETER_GRID
    exp_id = 0

    for num_conv in grid["num_conv_layers"]:
        for filters in grid["filters"]:
            for ksz in grid["kernel_size"]:
                for pool in grid["pooling_type"]:
                    exp_id += 1
                    name = experiment_name(num_conv, filters, ksz, pool)
                    weight_path = os.path.join(weights_dir, f"{name}.keras")

                    print(f"\n[{exp_id}/16] Experiment: {name}")

                    if skip_existing and os.path.exists(weight_path):
                        print(f"  Already trained. Loading from {weight_path}")
                        model = keras.models.load_model(weight_path)
                        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=-1)
                        val_f1 = f1_score(y_val, y_pred, average="macro")
                        all_results.append({
                            "name":        name,
                            "model":       model,
                            "history":     None,
                            "best_val_f1": val_f1,
                            "weight_path": weight_path,
                        })
                        continue

                    model = build_cnn_model(
                        num_conv_layers=num_conv,
                        filters=filters,
                        kernel_size=ksz,
                        pooling_type=pool,
                    )

                    result = train_one(
                        name=name,
                        model=model,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val,
                        epochs=epochs,
                        batch_size=batch_size,
                        weights_dir=weights_dir,
                        logs_dir=logs_dir,
                    )
                    all_results.append(result)

    return all_results

def summarize_results(all_results: list) -> dict:
    print("\n" + "=" * 60)
    print(f"{'Experiment':<50} {'Val F1':>8}")
    print("=" * 60)

    best_name = None
    best_f1   = -1.0

    for r in sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True):
        print(f"{r['name']:<50} {r['best_val_f1']:>8.4f}")
        if r["best_val_f1"] > best_f1:
            best_f1   = r["best_val_f1"]
            best_name = r["name"]

    print("=" * 60)
    print(f"Best model: {best_name}  (F1={best_f1:.4f})")
    return {"best_name": best_name, "best_f1": best_f1}