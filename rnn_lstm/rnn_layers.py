import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0.0, x)

def linear(x):
    return x

ACTIVATIONS = {
    "relu": relu, "sigmoid": sigmoid, "tanh": tanh,
    "softmax": softmax, "linear": linear, None: linear,
}


class Layer:
    def load_weights_from_keras(self, keras_layer):
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Dense(Layer):
    def __init__(self, activation=None):
        self.W = self.b = None
        self.act = ACTIVATIONS.get(activation, linear)

    def load_weights_from_keras(self, keras_layer):
        W, b = keras_layer.get_weights()
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        cfg = keras_layer.get_config()
        self.act = ACTIVATIONS.get(cfg.get("activation"), linear)

    def forward(self, x):
        return self.act(x @ self.W + self.b)


class Embedding(Layer):
    def __init__(self):
        self.W = None

    def load_weights_from_keras(self, keras_layer):
        self.W = keras_layer.get_weights()[0].astype(np.float32)

    def forward(self, x):
        return self.W[x]


class SimpleRNNCell(Layer):
    def __init__(self):
        self.W_i = self.W_h = self.b = None

    def load_weights_from_keras(self, keras_layer):
        ws = keras_layer.get_weights()
        self.W_i = ws[0].astype(np.float32)
        self.W_h = ws[1].astype(np.float32)
        self.b   = ws[2].astype(np.float32)
        self.units = self.W_h.shape[0]

    def forward(self, x_t, h_prev):
        return tanh(x_t @ self.W_i + h_prev @ self.W_h + self.b)



class LSTMCell(Layer):
    def __init__(self):
        self.W_i = self.W_h = self.b = None

    def load_weights_from_keras(self, keras_layer):
        ws = keras_layer.get_weights()
        self.W_i = ws[0].astype(np.float32)
        self.W_h = ws[1].astype(np.float32)
        self.b   = ws[2].astype(np.float32)
        self.units = self.W_h.shape[0]

    def forward(self, x_t, h_prev, c_prev):
        z = x_t @ self.W_i + h_prev @ self.W_h + self.b
        z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)

        i_gate = sigmoid(z_i)
        f_gate = sigmoid(z_f)
        o_gate = sigmoid(z_o)
        g      = tanh(z_c)

        c_next = f_gate * c_prev + i_gate * g
        h_next = o_gate * tanh(c_next)
        return h_next, c_next


class SimpleRNN(Layer):
    def __init__(self, cells=None):
        self.cells = cells or []

    @classmethod
    def from_keras(cls, keras_rnn_layer):
        obj = cls()
        cell = SimpleRNNCell()
        cell.W_i = keras_rnn_layer.get_weights()[0].astype(np.float32)
        cell.W_h = keras_rnn_layer.get_weights()[1].astype(np.float32)
        cell.b   = keras_rnn_layer.get_weights()[2].astype(np.float32)
        cell.units = cell.W_h.shape[0]
        obj.cells = [cell]
        return obj

    def forward(self, x, h0=None):
        batch, T, _ = x.shape
        units = self.cells[-1].units
        outputs = []

        h = h0 if h0 is not None else np.zeros((batch, self.cells[0].units), np.float32)

        # Only one layer supported in this simplified version
        cell = self.cells[0]
        for t in range(T):
            h = cell.forward(x[:, t, :], h)
            outputs.append(h[:, np.newaxis, :])   # (batch, 1, units)

        return np.concatenate(outputs, axis=1)   # (batch, T, units)


class LSTM(Layer):
    def __init__(self, cell=None):
        self.cell = cell

    @classmethod
    def from_keras(cls, keras_lstm_layer):
        obj = cls()
        cell = LSTMCell()
        cell.W_i = keras_lstm_layer.get_weights()[0].astype(np.float32)
        cell.W_h = keras_lstm_layer.get_weights()[1].astype(np.float32)
        cell.b   = keras_lstm_layer.get_weights()[2].astype(np.float32)
        cell.units = cell.W_h.shape[0]
        obj.cell = cell
        return obj

    def forward(self, x, h0=None, c0=None):
        batch, T, _ = x.shape
        units = self.cell.units
        h = h0 if h0 is not None else np.zeros((batch, units), np.float32)
        c = c0 if c0 is not None else np.zeros((batch, units), np.float32)
        outputs = []

        for t in range(T):
            h, c = self.cell.forward(x[:, t, :], h, c)
            outputs.append(h[:, np.newaxis, :])

        return np.concatenate(outputs, axis=1)   # (batch, T, units)