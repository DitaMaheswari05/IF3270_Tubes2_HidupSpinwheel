import time
import numpy as np
from rnn_lstm.rnn_layers import Embedding, Dense, SimpleRNNCell, LSTMCell, softmax, ACTIVATIONS
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


class ImageProjection:
    def __init__(self):
        self.W = self.b = None

    def load_from_keras(self, keras_layer):
        ws = keras_layer.get_weights()
        self.W = ws[0].astype(np.float32)
        self.b = ws[1].astype(np.float32)

    def forward(self, x):
        out = x @ self.W + self.b
        return np.maximum(0.0, out)   # ReLU


class CaptioningModelScratch:
    def __init__(self, rnn_type: str = "lstm"):
        self.rnn_type    = rnn_type
        self.img_proj    = ImageProjection()
        self.embedding   = Embedding()
        self.rnn_cells   = []
        self.output_proj = Dense()


    def load_weights_from_keras(self, keras_model):
        layer_dict = {l.name: l for l in keras_model.layers}

        # Image projection
        self.img_proj.load_from_keras(layer_dict["img_proj"])

        # Word embedding
        self.embedding.load_weights_from_keras(layer_dict["word_emb"])

        # RNN / LSTM cells (supports multi-layer)
        i = 1
        self.rnn_cells = []
        while True:
            key = f"{self.rnn_type}_{i}"
            if key not in layer_dict:
                break
            kl = layer_dict[key]
            if self.rnn_type == "lstm":
                cell = LSTMCell()
            else:
                cell = SimpleRNNCell()
            cell.W_i = kl.get_weights()[0].astype(np.float32)
            cell.W_h = kl.get_weights()[1].astype(np.float32)
            cell.b   = kl.get_weights()[2].astype(np.float32)
            cell.units = cell.W_h.shape[0]
            self.rnn_cells.append(cell)
            i += 1

        # Output Dense
        self.output_proj.load_weights_from_keras(layer_dict["output"])
        # Ensure softmax activation
        self.output_proj.act = softmax

    def _rnn_step(self, x_t, states):
        new_states = []
        inp = x_t
        for layer_idx, cell in enumerate(self.rnn_cells):
            if self.rnn_type == "lstm":
                h, c = states[layer_idx]
                h, c = cell.forward(inp, h, c)
                new_states.append((h, c))
                inp = h
            else:
                h = states[layer_idx]
                h = cell.forward(inp, h)
                new_states.append(h)
                inp = h
        return new_states, inp   # inp is final layer output

    def _init_states(self, batch, units):
        if self.rnn_type == "lstm":
            return [(np.zeros((batch, units), np.float32),
                     np.zeros((batch, units), np.float32))
                    for _ in self.rnn_cells]
        else:
            return [np.zeros((batch, units), np.float32) for _ in self.rnn_cells]

    # ── greedy decode ────────────────────────────────────────────────────────

    def generate_caption(
        self,
        feature: np.ndarray,       # (FEATURE_DIM,)
        word_to_idx: dict,
        idx_to_word: dict,
        max_len: int = 34,
    ) -> str:
        feat = feature[np.newaxis, :]        # (1, FEATURE_DIM)
        units = self.rnn_cells[-1].units

        # Step -1: inject image feature
        x_img = self.img_proj.forward(feat)  # (1, embed_dim)
        states = self._init_states(1, units)
        states, _ = self._rnn_step(x_img, states)

        start_idx = word_to_idx["<start>"]
        end_idx   = word_to_idx["<end>"]

        token = np.array([[start_idx]], dtype=np.int32)
        generated = []

        for _ in range(max_len):
            x_t = self.embedding.forward(token).reshape(1, -1)   # (1, embed_dim)
            states, h_out = self._rnn_step(x_t, states)
            logits  = self.output_proj.forward(h_out)              # (1, vocab_size)
            next_id = int(np.argmax(logits[0]))
            if next_id == end_idx:
                break
            generated.append(idx_to_word.get(next_id, "<unk>"))
            token = np.array([[next_id]], dtype=np.int32)

        return " ".join(generated)

    def generate_batch(
        self,
        features: np.ndarray,      # (N, FEATURE_DIM)
        word_to_idx: dict,
        idx_to_word: dict,
        max_len: int = 34,
    ) -> list:
        captions = []
        for i, feat in enumerate(features):
            cap = self.generate_caption(feat, word_to_idx, idx_to_word, max_len)
            captions.append(cap)
            if (i + 1) % 50 == 0:
                print(f"  [Scratch decode] {i+1}/{len(features)}", end="\r")
        print()
        return captions


# BLEU-4 + timing helper
def compute_bleu4(hypotheses: list, references: list) -> float:
    sf   = SmoothingFunction().method1
    refs = [[r.lower().split() for r in ref_group] for ref_group in references]
    hyps = [h.split() for h in hypotheses]
    return corpus_bleu(refs, hyps, smoothing_function=sf)


def batch_bleu4_scratch(
    scratch_model,
    image_features: dict,
    image_captions: dict,
    word_to_idx: dict,
    idx_to_word: dict,
    max_len: int = 34,
    n_images: int = None,
) -> dict:

    fnames = list(image_features.keys())
    if n_images:
        fnames = fnames[:n_images]
    fnames = [f for f in fnames if f in image_captions]

    t0 = time.time()
    hypotheses = []
    references = []
    for i, fname in enumerate(fnames):
        feat = image_features[fname]
        cap  = scratch_model.generate_caption(feat, word_to_idx, idx_to_word, max_len)
        hypotheses.append(cap)
        references.append(image_captions[fname])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(fnames)}", end="\r")
    elapsed = time.time() - t0

    bleu4 = compute_bleu4(hypotheses, references)
    print(f"\n  BLEU-4 (scratch): {bleu4:.4f}  ({elapsed:.1f}s)")
    return {"bleu4": bleu4, "time_s": elapsed, "hypotheses": hypotheses, "references": references}