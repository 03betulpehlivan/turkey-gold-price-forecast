"""
LSTM (numpy, sifirdan)
======================
Tek katmanli LSTM + dogrusal cikis, cok-girdi -> tek cikti (dizinin son adimi). Kayip: ortalama kare hata.
Egitim: zamanda geriye yayilim (BPTT), Adam, gradyan normu kirpma, sabit tohum (tekrarlanabilir).

Neden numpy: PyTorch/TensorFlow bu proje ve CI icin cok agir (GB'larca). Burada her denklem acik ve
dogrulugu sayisal gradyan kontrolu ile sinanir (tests/test_leakage_demo.py).

Kapilar (sira i, f, o, g):  z = x W_x + h W_h + b
  i = s(z_i), f = s(z_f), o = s(z_o), g = tanh(z_g);  c = f*c + i*g;  h = o*tanh(c);  y = h_T W_y + b_y
"""

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTM:
    def __init__(self, n_in: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = hidden
        s_in, s_h = 1.0 / np.sqrt(n_in), 1.0 / np.sqrt(H)
        self.p = {"Wx": rng.uniform(-s_in, s_in, (n_in, 4 * H)), "Wh": rng.uniform(-s_h, s_h, (H, 4 * H)),
                  "b": np.zeros(4 * H), "Wy": rng.uniform(-s_h, s_h, (H, 1)), "by": np.zeros(1)}
        self.p["b"][H:2 * H] = 1.0                       # unutma kapisi baslangicta acik
        self.H = H

    # -- ileri ---------------------------------------------------------------
    def forward(self, X: np.ndarray, keep: bool = False):
        N, T, _ = X.shape
        H, p = self.H, self.p
        h, c = np.zeros((N, H)), np.zeros((N, H))
        cache = []
        for t in range(T):
            z = X[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["b"]
            i, f, o, g = _sig(z[:, :H]), _sig(z[:, H:2 * H]), _sig(z[:, 2 * H:3 * H]), np.tanh(z[:, 3 * H:])
            c_prev, h_prev = c, h
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            if keep:
                cache.append((X[:, t, :], h_prev, c_prev, i, f, o, g, tc))
        y = (h @ p["Wy"] + p["by"])[:, 0]
        return (y, (cache, h)) if keep else y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(X, float))

    # -- geri ----------------------------------------------------------------
    def loss_grad(self, X: np.ndarray, t: np.ndarray):
        y, (cache, hT) = self.forward(X, keep=True)
        N, H, p = len(t), self.H, self.p
        err = y - t
        loss = float(np.mean(err ** 2))
        dy = (2.0 / N) * err[:, None]
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wy"] = hT.T @ dy
        g["by"] = dy.sum(axis=0)
        dh = dy @ p["Wy"].T
        dc = np.zeros((N, H))
        for x_t, h_prev, c_prev, i, f, o, gg, tc in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc ** 2)
            di, dgg, df = dc * gg, dc * i, dc * c_prev
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dgg * (1 - gg ** 2)], axis=1)
            g["Wx"] += x_t.T @ dz
            g["Wh"] += h_prev.T @ dz
            g["b"] += dz.sum(axis=0)
            dh = dz @ p["Wh"].T
            dc = dc * f
        return loss, g

    # -- egitim --------------------------------------------------------------
    def fit(self, X, t, epochs: int = 40, batch: int = 128, lr: float = 1e-3, clip: float = 1.0, seed: int = 0):
        X, t = np.asarray(X, float), np.asarray(t, float)
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps, step, hist = 0.9, 0.999, 1e-8, 0, []
        for _ in range(epochs):
            order = rng.permutation(len(t))
            tot = 0.0
            for s in range(0, len(t), batch):
                idx = order[s:s + batch]
                loss, g = self.loss_grad(X[idx], t[idx])
                tot += loss * len(idx)
                norm = np.sqrt(sum(float((gv ** 2).sum()) for gv in g.values()))
                scale = min(1.0, clip / (norm + 1e-12))
                step += 1
                for k in self.p:
                    gk = g[k] * scale
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    v[k] = b2 * v[k] + (1 - b2) * gk ** 2
                    self.p[k] -= lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
            hist.append(tot / len(t))
        return hist
