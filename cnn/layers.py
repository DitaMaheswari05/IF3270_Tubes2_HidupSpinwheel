import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def linear(x: np.ndarray) -> np.ndarray:
    return x

ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "linear": linear,
    None: linear,
}

class Layer: # Layer base class
    def load_weights_from_keras(self, keras_layer):
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Dense(Layer):
    def __init__(self, units: int = None, activation: str = None):
        self.units = units
        self.activation_fn = ACTIVATIONS.get(activation, linear)
        self.W = None
        self.b = None

    def load_weights_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W = weights[0].astype(np.float32)
        self.b = weights[1].astype(np.float32)
        self.units = self.W.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        return self.activation_fn(out)

class Conv2D(Layer):
    def __init__(
        self,
        filters: int = None,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding: str = "valid",
        activation: str = None,
    ):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides     = strides     if isinstance(strides, tuple)     else (strides, strides)
        self.padding     = padding.lower()
        self.activation_fn = ACTIVATIONS.get(activation, linear)

        self.kernel = None
        self.bias   = None

    def load_weights_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.kernel = weights[0].astype(np.float32)
        self.bias   = weights[1].astype(np.float32)
        self.filters     = self.kernel.shape[3]
        self.kernel_size = self.kernel.shape[:2]

        cfg = keras_layer.get_config()
        self.strides = tuple(cfg["strides"])
        self.padding = cfg["padding"]

        act_name = cfg.get("activation", "linear")
        self.activation_fn = ACTIVATIONS.get(act_name, linear)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "valid":
            return x
        H, W = x.shape[1], x.shape[2]
        kH, kW = self.kernel_size
        sH, sW = self.strides

        pad_h = max((H - 1) * sH + kH - H, 0)
        pad_w = max((W - 1) * sW + kW - W, 0)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        return np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode="constant")

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        x_pad = self._pad(x)
        N, H_pad, W_pad, C_in = x_pad.shape
        kH, kW = self.kernel_size
        sH, sW = self.strides
        C_out = self.filters

        H_out = (H_pad - kH) // sH + 1
        W_out = (W_pad - kW) // sW + 1

        out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

        K = self.kernel.reshape(-1, C_out)

        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                patch_flat = patch.reshape(N, -1)
                out[:, i, j, :] = patch_flat @ K + self.bias

        out = self.activation_fn(out)

        return out[0] if single else out

class LocallyConnected2D(Layer): # Convolution tanpa sharing params
    def __init__(
        self,
        filters: int = None,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding: str = "valid",
        activation: str = None,
    ):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides     = strides     if isinstance(strides, tuple)     else (strides, strides)
        self.padding     = padding.lower()
        self.activation_fn = ACTIVATIONS.get(activation, linear)

        self.kernel = None
        self.bias   = None
        self.H_out  = None
        self.W_out  = None

    def load_weights_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()   # [kernel, bias]
        self.kernel = weights[0].astype(np.float32)
        self.bias   = weights[1].astype(np.float32)
        self.filters = self.kernel.shape[2]

        cfg = keras_layer.get_config()
        self.kernel_size = tuple(cfg["kernel_size"])
        self.strides     = tuple(cfg["strides"])
        self.padding     = cfg["padding"]

        act_name = cfg.get("activation", "linear")
        self.activation_fn = ACTIVATIONS.get(act_name, linear)

        n_positions = self.kernel.shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C_in = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.strides

        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1
        C_out = self.filters

        out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

        pos = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                patch_flat = patch.reshape(N, -1)
                K_pos = self.kernel[pos]
                b_pos = self.bias[pos]
                out[:, i, j, :] = patch_flat @ K_pos + b_pos
                pos += 1

        out = self.activation_fn(out)
        return out[0] if single else out

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)

    def load_weights_from_keras(self, keras_layer):
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides   = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.strides

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        out = np.zeros((N, H_out, W_out, C), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                region = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]
                out[:, i, j, :] = region.max(axis=(1, 2))

        return out[0] if single else out


class AveragePooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        if strides is None:
            strides = self.pool_size
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)

    def load_weights_from_keras(self, keras_layer):
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides   = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]

        N, H, W, C = x.shape
        pH, pW = self.pool_size
        sH, sW = self.strides

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1

        out = np.zeros((N, H_out, W_out, C), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                region = x[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :]
                out[:, i, j, :] = region.mean(axis=(1, 2))

        return out[0] if single else out


class GlobalMaxPooling2D(Layer):
    def load_weights_from_keras(self, keras_layer):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]
        out = x.max(axis=(1, 2))
        return out[0] if single else out


class GlobalAveragePooling2D(Layer):
    def load_weights_from_keras(self, keras_layer):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]
        out = x.mean(axis=(1, 2))
        return out[0] if single else out

class Flatten(Layer):
    def load_weights_from_keras(self, keras_layer):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 3
        if single:
            x = x[np.newaxis]
        out = x.reshape(x.shape[0], -1)
        return out[0] if single else out

class Embedding(Layer):
    def __init__(self, vocab_size: int = None, embed_dim: int = None):
        self.W = None
    
    def load_weights_from_keras(self, keras_layer):
        self.W = keras_layer.get_weights().astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W[x]

class SimpleRNNCell(Layer):
    def __init__(self, units: int = None):
        self.units = units
        self.W, self.U, self.b = None, None, None

    def load_weights_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W, self.U, self.b = weights, weights[1], weights[2]

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        z = x_t @ self.W + h_prev @ self.U + self.b
        return tanh(z)

class LSTMCell(Layer):
    def __init__(self, units: int = None):
        self.units = units
        self.W, self.U, self.b = None, None, None

    def load_weights_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W, self.U, self.b = weights, weights[1], weights[2]

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        z = x_t @ self.W + h_prev @ self.U + self.b
        z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)
        
        i, f, o = sigmoid(z_i), sigmoid(z_f), sigmoid(z_o)
        c_tilde = tanh(z_c)
        
        c_next = f * c_prev + i * c_tilde
        h_next = o * tanh(c_next)
        return h_next, c_next