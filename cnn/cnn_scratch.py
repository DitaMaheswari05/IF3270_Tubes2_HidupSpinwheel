import numpy as np

from layers import (
    Conv2D,
    LocallyConnected2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    ACTIVATIONS,
    linear,
)

LAYER_MAP_SHARED = {
    "Conv2D":                  Conv2D,
    "MaxPooling2D":            MaxPooling2D,
    "AveragePooling2D":        AveragePooling2D,
    "GlobalMaxPooling2D":      GlobalMaxPooling2D,
    "GlobalAveragePooling2D":  GlobalAveragePooling2D,
    "Flatten":                 Flatten,
    "Dense":                   Dense,
}

LAYER_MAP_NON_SHARED = {
    **LAYER_MAP_SHARED,
    "Conv2D": LocallyConnected2D,   # tinggal ganti Conv2D dengan LocallyConnected2D
}

def _build_scratch_layer(keras_layer, layer_map: dict):
    cls_name = keras_layer.__class__.__name__
    if cls_name not in layer_map:
        return None

    scratch_cls = layer_map[cls_name]
    layer = scratch_cls()

    try:
        layer.load_weights_from_keras(keras_layer)
    except NotImplementedError:
        pass

    return layer

class CNNScratch:
    def __init__(self, layers: list):
        self.layers = [l for l in layers if l is not None]

    @classmethod
    def from_keras(cls, keras_model, non_shared: bool = False):
        layer_map = LAYER_MAP_NON_SHARED if non_shared else LAYER_MAP_SHARED
        scratch_layers = []
        for kl in keras_model.layers:
            sl = _build_scratch_layer(kl, layer_map)
            scratch_layers.append(sl)
        return cls(scratch_layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        results = []
        n = X.shape[0]
        for start in range(0, n, batch_size):
            batch = X[start:start + batch_size]
            probs = self.forward(batch)
            results.append(probs)
            print(f"  [Scratch forward] {min(start+batch_size, n)}/{n}", end="\r")
        print()
        return np.concatenate(results, axis=0)

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        probs = self.predict_proba(X, batch_size)
        return np.argmax(probs, axis=-1)

class CNNScratchLC(CNNScratch):
    @classmethod
    def from_keras(cls, keras_model):
        return super().from_keras(keras_model, non_shared=True)