import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter


EMBED_DIM   = 256
VOCAB_SIZE  = 5000
MAX_LEN     = 34
FEATURE_DIM = 2048
EPOCHS      = 20
BATCH_SIZE  = 64

HYPERPARAMETER_GRID = {
    "num_rnn_layers": [1, 2, 3],
    "hidden_units":   [128, 512],
}

START_TOKEN = "<start>"
END_TOKEN   = "<end>"
PAD_TOKEN   = "<pad>"
UNK_TOKEN   = "<unk>"


def build_vocab(captions_list, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for cap in captions_list:
        tokens = cap.lower().split()
        counter.update(tokens)

    # Reserve indices: 0=<pad>, 1=<start>, 2=<end>, 3=<unk>
    special = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
    most_common = [w for w, _ in counter.most_common(vocab_size - len(special))]
    vocab = special + most_common

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    return word_to_idx, idx_to_word


def encode_caption(caption, word_to_idx, max_len=MAX_LEN):
    tokens = [START_TOKEN] + caption.lower().split() + [END_TOKEN]
    ids    = [word_to_idx.get(t, word_to_idx[UNK_TOKEN]) for t in tokens]
    if len(ids) < max_len + 1:
        ids += [word_to_idx[PAD_TOKEN]] * (max_len + 1 - len(ids))
    else:
        ids = ids[: max_len + 1]
    return ids


def make_caption_dataset(
    image_features: dict,
    image_captions: dict,
    word_to_idx: dict,
    max_len: int = MAX_LEN,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
):
    feats, inputs, targets = [], [], []

    for fname, caps in image_captions.items():
        if fname not in image_features:
            continue
        feat = image_features[fname].astype(np.float32)
        for cap in caps:
            full_seq = encode_caption(cap, word_to_idx, max_len)
            feats.append(feat)
            inputs.append(full_seq[:-1])
            targets.append(full_seq[1:])

    feats   = np.array(feats,   dtype=np.float32)
    inputs  = np.array(inputs,  dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((feats, inputs, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(feats), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_caption_decoder(
    rnn_type: str = "lstm",
    num_rnn_layers: int = 1,
    hidden_units: int = 256,
    embed_dim: int = EMBED_DIM,
    vocab_size: int = VOCAB_SIZE,
    feature_dim: int = FEATURE_DIM,
    max_len: int = MAX_LEN,
) -> keras.Model:
    feat_in   = keras.Input(shape=(feature_dim,),  name="cnn_feature")
    token_in  = keras.Input(shape=(max_len,),      name="caption_tokens", dtype=tf.int32)

    img_proj = KL.Dense(embed_dim, activation="relu", name="img_proj")(feat_in)
    img_proj = KL.Reshape((1, embed_dim), name="img_expand")(img_proj)

    word_emb = KL.Embedding(vocab_size, embed_dim, mask_zero=True, name="word_emb")(token_in)

    seq = KL.Concatenate(axis=1, name="prepend_img")([img_proj, word_emb])

    RNNLayer = KL.LSTM if rnn_type == "lstm" else KL.SimpleRNN

    for i in range(num_rnn_layers):
        return_seq = True
        x = RNNLayer(
            hidden_units,
            return_sequences=return_seq,
            name=f"{rnn_type}_{i+1}",
            dropout=0.3,
        )(seq if i == 0 else x)
    x = KL.Lambda(lambda t: t[:, :-1, :], name="trim_last")(x)
    out = KL.Dense(vocab_size, activation="softmax", name="output")(x)
    model = keras.Model(inputs=[feat_in, token_in], outputs=out,
                        name=f"captioner_{rnn_type}_L{num_rnn_layers}_H{hidden_units}")
    return model


def experiment_name(rnn_type, num_layers, hidden_units):
    return f"{rnn_type}_L{num_layers}_H{hidden_units}"


def train_one_decoder(
    name: str,
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int = EPOCHS,
    weights_dir: str = "weights/captioning",
    logs_dir: str    = "logs/captioning",
) -> dict:
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    weight_path = os.path.join(weights_dir, f"{name}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            weight_path, save_best_only=True,
            monitor="val_loss", verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds.map(lambda f, i, t: ((f, i), t)),
        validation_data=val_ds.map(lambda f, i, t: ((f, i), t)),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    hist_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    log_path  = os.path.join(logs_dir, f"{name}_history.json")
    with open(log_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    print(f"  [{name}] Training time: {train_time:.1f}s | Saved: {weight_path}")

    return {
        "name":       name,
        "model":      model,
        "history":    hist_dict,
        "train_time": train_time,
        "weight_path": weight_path,
    }


def train_all_decoders(
    train_ds,
    val_ds,
    rnn_types=("rnn", "lstm"),
    epochs: int = EPOCHS,
    weights_dir: str = "weights/captioning",
    logs_dir: str    = "logs/captioning",
    skip_existing: bool = True,
    vocab_size: int = VOCAB_SIZE,
    embed_dim:  int = EMBED_DIM,
    feature_dim: int = FEATURE_DIM,
    max_len: int = MAX_LEN,
) -> list:
    all_results = []
    grid = HYPERPARAMETER_GRID
    exp_id = 0

    for rnn_type in rnn_types:
        for num_layers in grid["num_rnn_layers"]:
            for hidden in grid["hidden_units"]:
                exp_id += 1
                name = experiment_name(rnn_type, num_layers, hidden)
                weight_path = os.path.join(weights_dir, f"{name}.keras")

                print(f"\n[{exp_id}] Experiment: {name}")

                if skip_existing and os.path.exists(weight_path):
                    print(f"  Already trained. Loading from {weight_path}")
                    model = keras.models.load_model(weight_path)
                    all_results.append({
                        "name": name, "model": model,
                        "history": None, "train_time": 0,
                        "weight_path": weight_path,
                        "rnn_type": rnn_type,
                        "num_layers": num_layers,
                        "hidden_units": hidden,
                    })
                    continue

                model = build_caption_decoder(
                    rnn_type=rnn_type,
                    num_rnn_layers=num_layers,
                    hidden_units=hidden,
                    vocab_size=vocab_size,
                    embed_dim=embed_dim,
                    feature_dim=feature_dim,
                    max_len=max_len,
                )

                result = train_one_decoder(
                    name=name, model=model,
                    train_ds=train_ds, val_ds=val_ds,
                    epochs=epochs,
                    weights_dir=weights_dir,
                    logs_dir=logs_dir,
                )
                result.update({"rnn_type": rnn_type, "num_layers": num_layers, "hidden_units": hidden})
                all_results.append(result)

    return all_results

def generate_caption_greedy(
    model: keras.Model,
    feature: np.ndarray,
    word_to_idx: dict,
    idx_to_word: dict,
    max_len: int = MAX_LEN,
) -> str:
    feat = feature[np.newaxis, :]
    start_idx = word_to_idx[START_TOKEN]
    end_idx   = word_to_idx[END_TOKEN]
    pad_idx   = word_to_idx[PAD_TOKEN]

    tokens = [start_idx]

    for _ in range(max_len):
        seq = np.array(tokens + [pad_idx] * (max_len - len(tokens)), dtype=np.int32)[np.newaxis, :]
        preds = model.predict([feat, seq], verbose=0)
        next_idx = int(np.argmax(preds[0, len(tokens) - 1, :]))
        tokens.append(next_idx)
        if next_idx == end_idx:
            break

    words = [idx_to_word.get(t, UNK_TOKEN) for t in tokens[1:]]
    if END_TOKEN in words:
        words = words[: words.index(END_TOKEN)]
    return " ".join(words)

def evaluate_bleu4(
    model: keras.Model,
    image_features: dict,
    image_captions: dict,
    word_to_idx: dict,
    idx_to_word: dict,
    max_len: int = MAX_LEN,
    n_images: int = None,
) -> float:
    sf = SmoothingFunction().method1
    references = []
    hypotheses = []

    fnames = list(image_features.keys())
    if n_images:
        fnames = fnames[:n_images]

    for i, fname in enumerate(fnames):
        if fname not in image_captions:
            continue
        feat  = image_features[fname]
        hyp   = generate_caption_greedy(model, feat, word_to_idx, idx_to_word, max_len)
        refs  = [cap.lower().split() for cap in image_captions[fname]]
        hypotheses.append(hyp.split())
        references.append(refs)

        if (i + 1) % 100 == 0:
            print(f"  BLEU eval: {i+1}/{len(fnames)}", end="\r")

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=sf)
    return bleu4