"""
UDL Assignment 1 - Representation Learning with Autoencoder Variants
==================================================================
Group 93 | AIMLZG533 (Unsupervised Deep Learning)

Single-file, structured implementation.  Once the numbers look right this file
is converted to the submission notebook:

    pip install jupytext
    jupytext --to notebook assignment1.py

--------------------------------------------------------------------------------
Datasets
    MNIST   (Keras)                       -> already 28x28 grayscale
    CIFAR10 (Keras, RGB -> grayscale)     -> resized 32x32 -> 28x28
Shared preprocessing (one pipeline, reused by every task)
    * intensity linearly rescaled to [50, 200]
    * 70 / 20 / 10  train / val / test   (stratified)
    * mean-centered with the TRAIN mean

Task 1  Standard PCA vs Randomized PCA (top-30)
        -> logistic-regression classifier, ROC curves, 30-component recon SNR
Task 2  Tied-weight, unit-norm single-layer LINEAR autoencoder (30-D bottleneck)
        -> eigenvector / weight visualisation, principal-angle subspace metric,
           logistic regression on AE features vs PCA features
Task 3  Shallow nonlinear AE / deep dense AE / deep conv AE (all 30-D latent)
        -> architecture, params, train/val loss, train/test recon SNR,
           vs the PCA-30 linear baseline

Usage
    python assignment1.py                          # everything, both datasets
    python assignment1.py --tasks 1 2 --datasets mnist
    python assignment1.py --quick                  # few epochs, smoke test
    python assignment1.py --show                   # also pop up figures
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieten TF import chatter

import numpy as np
import matplotlib

# Use a non-interactive backend when run as a head-less script (not when this
# module is imported, e.g. from the notebook).  Must happen before pyplot import.
if (__name__ == "__main__"
        and os.environ.get("MPLBACKEND") is None
        and "--show" not in sys.argv):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


# ==========================================================================
# 0. Configuration
# ==========================================================================
@dataclass
class Config:
    seed: int = 42
    n_components: int = 30                 # latent / PCA dimensionality
    img_size: int = 28
    intensity_lo: float = 50.0
    intensity_hi: float = 200.0
    n_classes: int = 10

    # split fractions (train / val / test)
    test_size: float = 0.30               # val+test taken out first
    test_of_temp: float = 1.0 / 3.0       # 1/3 of the 30% -> 10% test, 20% val

    # training budgets
    epochs_linear_ae: int = 80
    epochs_shallow: int = 60
    epochs_deep_dense: int = 80
    epochs_cnn: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3

    outdir: Path = field(default_factory=lambda: Path("outputs"))
    show: bool = False
    quick: bool = False

    def finalize(self) -> "Config":
        if self.quick:
            self.epochs_linear_ae = 4
            self.epochs_shallow = 4
            self.epochs_deep_dense = 4
            self.epochs_cnn = 3
        self.outdir.mkdir(parents=True, exist_ok=True)
        return self


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


# ==========================================================================
# 1. Shared data pipeline
# ==========================================================================
@dataclass
class DataBundle:
    """Everything the three tasks need for one dataset, computed exactly once."""
    name: str
    # mean-centered (feed these to PCA / autoencoders)
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    # [50, 200] intensity space, NOT centered (SNR reference / display)
    X_train_orig: np.ndarray
    X_test_orig: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    mean_vec: np.ndarray                  # train mean, add back to reconstructions

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]

    def as_images(self, flat: np.ndarray, size: int = 28) -> np.ndarray:
        return flat.reshape(-1, size, size, 1)


def _load_raw(name: str, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened float32 images in the native 0-255 range and int labels."""
    name = name.lower()
    if name == "mnist":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
        X = np.concatenate([x_tr, x_te], axis=0).astype("float32")[..., None]
        y = np.concatenate([y_tr, y_te], axis=0).ravel()
    elif name == "cifar10":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
        X = np.concatenate([x_tr, x_te], axis=0).astype("float32")
        y = np.concatenate([y_tr, y_te], axis=0).ravel()
        X = tf.image.rgb_to_grayscale(X).numpy()                    # RGB -> gray
        X = tf.image.resize(X, [cfg.img_size, cfg.img_size]).numpy()  # -> 28x28
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return X.reshape(len(X), -1), y


def preprocess(name: str, cfg: Config) -> DataBundle:
    """Load -> [50,200] rescale -> 70/20/10 stratified split -> mean-center."""
    print(f"\n--- preprocessing {name.upper()} ---")
    X, y = _load_raw(name, cfg)

    # intensity rescaled to [50, 200] over the whole dataset so it lies exactly
    # in the requested band regardless of the split.
    lo, hi = float(X.min()), float(X.max())
    span = cfg.intensity_hi - cfg.intensity_lo
    X = cfg.intensity_lo + (X - lo) / (hi - lo) * span

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=cfg.test_of_temp, random_state=cfg.seed, stratify=y_tmp)

    mean_vec = X_tr.mean(axis=0)
    print(f"train {X_tr.shape} | val {X_val.shape} | test {X_te.shape} | "
          f"intensity [{X.min():.1f}, {X.max():.1f}]")

    return DataBundle(
        name=name,
        X_train=(X_tr - mean_vec).astype("float32"),
        X_val=(X_val - mean_vec).astype("float32"),
        X_test=(X_te - mean_vec).astype("float32"),
        X_train_orig=X_tr.astype("float32"),
        X_test_orig=X_te.astype("float32"),
        y_train=y_tr, y_val=y_val, y_test=y_te,
        mean_vec=mean_vec.astype("float32"),
    )


# ==========================================================================
# 2. Shared metrics & plotting helpers
# ==========================================================================
def average_snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean per-image SNR: 10 * log10( ||x||^2 / ||x - x_hat||^2 ), in dB."""
    signal = np.sum(original ** 2, axis=1)
    noise = np.sum((original - reconstructed) ** 2, axis=1)
    noise = np.where(noise == 0.0, 1e-12, noise)
    return float(np.mean(10.0 * np.log10(signal / noise)))


def principal_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))


def subspace_metrics(V_pca: np.ndarray, W_ae: np.ndarray) -> dict:
    """
    Quantitative agreement between the PCA principal subspace (V_pca, D x k,
    orthonormal) and the linear-AE weight subspace (W_ae, D x k, unit columns).

    Uses principal angles theta_1..theta_k:
      * mean / max principal angle           - 0 deg  => identical subspaces
      * mean cos(theta)                       - 1.0    => identical subspaces
      * chordal (Grassmann) distance         - sqrt(sum sin^2 theta), 0 => identical
      * normalised projection Frobenius gap   - ||P_pca - P_ae||_F / sqrt(2k)  in [0, 1]
    Principal angles are the canonical, basis- and sign-invariant way to compare
    two subspaces, so they measure *subspace alignment* rather than a
    vector-by-vector match (which is meaningless because neither PCA nor the AE
    fixes the ordering or sign of individual directions).
    """
    k = V_pca.shape[1]
    Qa, _ = np.linalg.qr(V_pca)
    Qb, _ = np.linalg.qr(W_ae)
    cos_theta = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    P_pca = Qa @ Qa.T
    P_ae = Qb @ Qb.T
    proj_gap = np.linalg.norm(P_pca - P_ae, ord="fro") / np.sqrt(2.0 * k)

    return {
        "mean_angle_deg": float(theta.mean()),
        "max_angle_deg": float(theta.max()),
        "mean_cos": float(cos_theta.mean()),
        "chordal_distance": float(np.sqrt(np.sum(1.0 - cos_theta ** 2))),
        "projection_gap": float(proj_gap),
        "principal_angles_deg": theta,
    }


class _Fig:
    """Context manager: build a figure, save it to outdir, optionally show it."""

    def __init__(self, cfg: Config, fname: str, **kw):
        self.cfg = cfg
        self.fname = fname
        self.kw = kw

    def __enter__(self):
        self.fig = plt.figure(**self.kw)
        return self.fig

    def __exit__(self, *exc):
        self.fig.tight_layout()
        path = self.cfg.outdir / self.fname
        self.fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved {path}")
        if self.cfg.show:
            plt.show()
        plt.close(self.fig)
        return False


def plot_roc_ovr(y_true, y_score, title, ax, n_classes=10) -> float:
    """One-vs-rest ROC for every class on one axis; returns macro-average AUC."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, lw=1.3, label=f"class {i} (AUC={a:.3f})")
    macro = float(np.mean(aucs))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title}\nmacro-avg AUC = {macro:.3f}")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    return macro


def plot_image_grid(cfg, columns, title, fname, n=30, ncol=10, size=28):
    """Render the first n columns of `columns` (D x k) as size x size images."""
    n = min(n, columns.shape[1])
    nrow = int(np.ceil(n / ncol))
    with _Fig(cfg, fname, figsize=(1.4 * ncol, 1.5 * nrow)) as fig:
        fig.suptitle(title)
        for i in range(nrow * ncol):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.axis("off")
            if i < n:
                img = columns[:, i].reshape(size, size)
                ax.imshow(img, cmap="gray")
                ax.set_title(str(i + 1), fontsize=7)


def plot_reconstructions(cfg, orig, recon, title, fname, n=10, size=28):
    with _Fig(cfg, fname, figsize=(1.5 * n, 3.4)) as fig:
        fig.suptitle(title)
        for i in range(n):
            for row, img in ((0, orig[i]), (1, recon[i])):
                ax = fig.add_subplot(2, n, row * n + i + 1)
                ax.imshow(img.reshape(size, size), cmap="gray",
                          vmin=cfg.intensity_lo, vmax=cfg.intensity_hi)
                ax.axis("off")
                if i == 0:
                    ax.set_ylabel("original" if row == 0 else "recon")


def plot_history(cfg, histories: dict, title, fname):
    """histories: {label: keras.History}; plots train + val loss vs epoch."""
    with _Fig(cfg, fname, figsize=(7, 4.5)) as fig:
        ax = fig.add_subplot(111)
        for label, h in histories.items():
            ep = range(1, len(h.history["loss"]) + 1)
            line, = ax.plot(ep, h.history["loss"], lw=1.6, label=f"{label} train")
            if "val_loss" in h.history:
                ax.plot(ep, h.history["val_loss"], lw=1.2, ls="--",
                        color=line.get_color(), label=f"{label} val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


def model_summary_text(model: keras.Model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    return buf.getvalue()


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str, str]]):
    """columns: list of (key, header, fmt)."""
    print(f"\n=== {title} ===")
    hdr = "".join(f"{h:>{max(len(h) + 2, 12)}}" for _, h, _ in columns)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = ""
        for key, h, fmt in columns:
            width = max(len(h) + 2, 12)
            val = r.get(key, "")
            line += f"{format(val, fmt) if fmt else str(val):>{width}}"
        print(line)


# ==========================================================================
# 3. Task 1 - Standard PCA vs Randomized PCA
# ==========================================================================
def run_task1(data: DataBundle, cfg: Config) -> dict:
    print(f"\n################  TASK 1 - {data.name.upper()}  ################")
    X_tr, X_te = data.X_train, data.X_test
    y_tr, y_te = data.y_train, data.y_test

    solvers = [("Standard PCA", "full"), ("Randomized PCA", "randomized")]
    rows = []
    with _Fig(cfg, f"task1_{data.name}_roc.png", figsize=(15, 6)) as fig:
        for idx, (label, solver) in enumerate(solvers, start=1):
            t0 = time.time()
            pca = PCA(n_components=cfg.n_components, svd_solver=solver,
                      random_state=cfg.seed)
            Z_tr = pca.fit_transform(X_tr)
            fit_time = time.time() - t0
            Z_te = pca.transform(X_te)

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, solver="lbfgs"),
            )
            clf.fit(Z_tr, y_tr)
            acc_tr = accuracy_score(y_tr, clf.predict(Z_tr))
            acc_te = accuracy_score(y_te, clf.predict(Z_te))

            ax = fig.add_subplot(1, 2, idx)
            macro_auc = plot_roc_ovr(
                y_te, clf.predict_proba(Z_te),
                f"{data.name.upper()} - {label}", ax, cfg.n_classes)

            X_te_rec = pca.inverse_transform(Z_te) + data.mean_vec
            snr = average_snr_db(data.X_test_orig, X_te_rec)

            rows.append(dict(
                method=label, fit_s=fit_time,
                expl_var=float(pca.explained_variance_ratio_.sum()),
                train_acc=acc_tr, test_acc=acc_te,
                macro_auc=macro_auc, test_snr_db=snr,
            ))

    # qualitative visualisations from the standard-PCA subspace
    std_pca = PCA(n_components=cfg.n_components, svd_solver="full",
                  random_state=cfg.seed).fit(X_tr)
    plot_image_grid(
        cfg, std_pca.components_.T,
        f"{data.name.upper()} - top {cfg.n_components} principal components",
        f"task1_{data.name}_pcs.png", n=cfg.n_components)
    rec = std_pca.inverse_transform(std_pca.transform(X_te)) + data.mean_vec
    plot_reconstructions(
        cfg, data.X_test_orig, rec,
        f"{data.name.upper()} - test images reconstructed from {cfg.n_components} PCs",
        f"task1_{data.name}_recon.png")

    print_table(
        f"TASK 1 SUMMARY - {data.name.upper()} "
        f"({X_tr.shape[0]} train / {X_te.shape[0]} test)",
        rows,
        [("method", "method", ""), ("fit_s", "fit (s)", ".3f"),
         ("expl_var", "expl.var", ".3f"), ("train_acc", "train acc", ".3f"),
         ("test_acc", "test acc", ".3f"), ("macro_auc", "macro AUC", ".3f"),
         ("test_snr_db", "test SNR dB", ".2f")],
    )
    return {"rows": rows, "std_pca": std_pca}


# ==========================================================================
# 4. Task 2 - tied-weight, unit-norm linear autoencoder vs PCA
# ==========================================================================
class TiedLinearAutoencoder(keras.Model):
    """
    x_hat = (x @ W) @ W^T  with tied decoder weights and each encoder weight
    vector (column of W, in R^D) constrained to unit L2 norm.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int, **kw):
        super().__init__(**kw)
        self.W = self.add_weight(
            name="encoder_weights",
            shape=(input_dim, bottleneck_dim),
            initializer="glorot_uniform",
            constraint=keras.constraints.UnitNorm(axis=0),
            trainable=True,
        )

    def encode(self, x):
        return tf.matmul(x, self.W)

    def decode(self, z):
        return tf.matmul(z, self.W, transpose_b=True)

    def call(self, x):
        return self.decode(self.encode(x))


def run_task2(data: DataBundle, cfg: Config) -> dict:
    print(f"\n################  TASK 2 - {data.name.upper()}  ################")
    X_tr, X_te = data.X_train, data.X_test

    # reference PCA subspace (standard, full SVD) -----------------------------
    pca = PCA(n_components=cfg.n_components, svd_solver="full",
              random_state=cfg.seed).fit(X_tr)
    V_pca = pca.components_.T                       # (D, k) orthonormal
    Z_tr_pca = pca.transform(X_tr)
    Z_te_pca = pca.transform(X_te)

    # tied linear autoencoder ---------------------------------------------------
    ae = TiedLinearAutoencoder(data.input_dim, cfg.n_components,
                               name=f"tied_linear_ae_{data.name}")
    ae.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    hist = ae.fit(X_tr, X_tr, validation_data=(data.X_val, data.X_val),
                  epochs=cfg.epochs_linear_ae, batch_size=cfg.batch_size, verbose=0)
    W_ae = ae.W.numpy()                             # (D, k), unit columns
    print(f"  linear AE trained: final train MSE {hist.history['loss'][-1]:.4f} | "
          f"val MSE {hist.history['val_loss'][-1]:.4f}")

    plot_history(cfg, {"tied linear AE": hist},
                 f"{data.name.upper()} - linear AE reconstruction loss",
                 f"task2_{data.name}_loss.png")

    # subspace comparison -----------------------------------------------------
    m = subspace_metrics(V_pca, W_ae)
    print("\n  --- PCA vs linear-AE subspace alignment ---")
    print(f"  mean principal angle      : {m['mean_angle_deg']:.3f} deg")
    print(f"  max  principal angle      : {m['max_angle_deg']:.3f} deg")
    print(f"  mean cos(theta)           : {m['mean_cos']:.5f}")
    print(f"  chordal (Grassmann) dist  : {m['chordal_distance']:.5f}")
    print(f"  normalised projection gap : {m['projection_gap']:.5f}  (0 = identical, 1 = orthogonal)")

    plot_image_grid(cfg, V_pca,
                    f"{data.name.upper()} - top {cfg.n_components} PCA eigenvectors",
                    f"task2_{data.name}_pca_vectors.png", n=cfg.n_components)
    plot_image_grid(cfg, W_ae,
                    f"{data.name.upper()} - linear AE encoder weight vectors",
                    f"task2_{data.name}_ae_weights.png", n=cfg.n_components)

    # logistic regression: AE features vs PCA features ----------------------
    Z_tr_ae = ae.encode(X_tr).numpy()
    Z_te_ae = ae.encode(X_te).numpy()

    def logreg_acc(Z_tr, Z_te):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, solver="lbfgs"))
        clf.fit(Z_tr, data.y_train)
        return accuracy_score(data.y_test, clf.predict(Z_te))

    acc_pca = logreg_acc(Z_tr_pca, Z_te_pca)
    acc_ae = logreg_acc(Z_tr_ae, Z_te_ae)

    # reconstruction SNR for both, same 30-D budget ------------------------
    snr_pca = average_snr_db(
        data.X_test_orig, pca.inverse_transform(Z_te_pca) + data.mean_vec)
    snr_ae = average_snr_db(
        data.X_test_orig, ae.predict(X_te, verbose=0) + data.mean_vec)

    print_table(
        f"TASK 2 SUMMARY - {data.name.upper()}",
        [dict(feat="PCA-30", test_acc=acc_pca, test_snr_db=snr_pca),
         dict(feat="Linear AE-30", test_acc=acc_ae, test_snr_db=snr_ae)],
        [("feat", "features", ""), ("test_acc", "test acc", ".3f"),
         ("test_snr_db", "test SNR dB", ".2f")],
    )
    return {"metrics": m, "acc_pca": acc_pca, "acc_ae": acc_ae}


# ==========================================================================
# 5. Task 3 - nonlinearity, depth, convolution
# ==========================================================================
def build_shallow_ae(cfg: Config) -> keras.Model:
    """Single hidden layer == the 30-D bottleneck; nonlinear (only place it can be)."""
    inp = layers.Input(shape=(cfg.img_size ** 2,))
    z = layers.Dense(cfg.n_components, activation="relu", name="bottleneck")(inp)
    out = layers.Dense(cfg.img_size ** 2, activation="linear", name="reconstruction")(z)
    return keras.Model(inp, out, name="Shallow_Nonlinear_AE_30")


def build_deep_dense_ae(cfg: Config, hidden: int = 256) -> keras.Model:
    """Symmetric: hidden -> 30 -> hidden  (three hidden layers incl. bottleneck)."""
    inp = layers.Input(shape=(cfg.img_size ** 2,))
    x = layers.Dense(hidden, activation="relu")(inp)
    z = layers.Dense(cfg.n_components, activation="relu", name="bottleneck")(x)
    x = layers.Dense(hidden, activation="relu")(z)
    out = layers.Dense(cfg.img_size ** 2, activation="linear", name="reconstruction")(x)
    return keras.Model(inp, out, name="Deep_Dense_AE_30")


def build_deep_cnn_ae(cfg: Config) -> keras.Model:
    """Conv encoder -> 30-D dense bottleneck -> conv-transpose decoder, linear output."""
    s = cfg.img_size
    inp = layers.Input(shape=(s, s, 1))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)                 # 14x14x32
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)                 # 7x7x64
    x = layers.Flatten()(x)
    z = layers.Dense(cfg.n_components, activation="relu", name="bottleneck")(x)

    x = layers.Dense(7 * 7 * 64, activation="relu")(z)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    out = layers.Conv2D(1, 3, activation="linear", padding="same",
                        name="reconstruction")(x)                 # 28x28x1
    return keras.Model(inp, out, name="Deep_CNN_AE_30")


def _pca_baseline(data: DataBundle, cfg: Config) -> dict:
    pca = PCA(n_components=cfg.n_components, svd_solver="full",
              random_state=cfg.seed).fit(data.X_train)
    rec_tr_c = pca.inverse_transform(pca.transform(data.X_train))
    rec_val_c = pca.inverse_transform(pca.transform(data.X_val))
    rec_te_c = pca.inverse_transform(pca.transform(data.X_test))
    return dict(
        model="PCA-30", params=0,
        loss_tr=float(np.mean((data.X_train - rec_tr_c) ** 2)),
        loss_val=float(np.mean((data.X_val - rec_val_c) ** 2)),
        snr_tr=average_snr_db(data.X_train_orig, rec_tr_c + data.mean_vec),
        snr_te=average_snr_db(data.X_test_orig, rec_te_c + data.mean_vec),
        arch="linear projection onto top-30 eigenvectors",
    )


def run_task3(data: DataBundle, cfg: Config) -> dict:
    print(f"\n################  TASK 3 - {data.name.upper()}  ################")
    rows = [_pca_baseline(data, cfg)]

    builders = [
        (build_shallow_ae(cfg), cfg.epochs_shallow, False,
         "784 - [30 relu] - 784 linear"),
        (build_deep_dense_ae(cfg), cfg.epochs_deep_dense, False,
         "784 - 256 - [30] - 256 - 784 (relu, linear out)"),
        (build_deep_cnn_ae(cfg), cfg.epochs_cnn, True,
         "conv32/pool - conv64/pool - [30] - deconv64 - deconv32 - conv1 linear"),
    ]

    histories = {}
    for model, epochs, is_cnn, arch in builders:
        print(f"\n  training {model.name} ({epochs} epochs)...")
        print(model_summary_text(model))
        model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="mse")

        if is_cnn:
            Xtr = data.as_images(data.X_train, cfg.img_size)
            Xval = data.as_images(data.X_val, cfg.img_size)
            Xte = data.as_images(data.X_test, cfg.img_size)
        else:
            Xtr, Xval, Xte = data.X_train, data.X_val, data.X_test

        h = model.fit(Xtr, Xtr, validation_data=(Xval, Xval),
                      epochs=epochs, batch_size=cfg.batch_size, verbose=0)
        histories[model.name] = h

        rec_tr = model.predict(Xtr, verbose=0).reshape(len(Xtr), -1) + data.mean_vec
        rec_te = model.predict(Xte, verbose=0).reshape(len(Xte), -1) + data.mean_vec

        rows.append(dict(
            model=model.name, params=model.count_params(),
            loss_tr=float(h.history["loss"][-1]),
            loss_val=float(h.history["val_loss"][-1]),
            snr_tr=average_snr_db(data.X_train_orig, rec_tr),
            snr_te=average_snr_db(data.X_test_orig, rec_te),
            arch=arch,
        ))

        plot_reconstructions(
            cfg, data.X_test_orig, rec_te,
            f"{data.name.upper()} - {model.name} test reconstructions",
            f"task3_{data.name}_{model.name}_recon.png")

    plot_history(cfg, histories,
                 f"{data.name.upper()} - Task 3 reconstruction loss vs epoch",
                 f"task3_{data.name}_loss.png")

    print_table(
        f"TASK 3 SUMMARY - {data.name.upper()} "
        f"(PCA-30 vs Shallow vs Deep Dense vs Deep CNN)",
        rows,
        [("model", "model", ""), ("params", "params", ","),
         ("loss_tr", "loss_tr", ".4f"), ("loss_val", "loss_val", ".4f"),
         ("snr_tr", "SNR_tr dB", ".2f"), ("snr_te", "SNR_te dB", ".2f")],
    )
    for r in rows:
        print(f"  {r['model']:<22} {r['arch']}")
    return {"rows": rows}


# ==========================================================================
# 6. Orchestration
# ==========================================================================
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", type=int, choices=[1, 2, 3],
                   default=[1, 2, 3])
    p.add_argument("--datasets", nargs="+", choices=["mnist", "cifar10"],
                   default=["mnist", "cifar10"])
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    p.add_argument("--quick", action="store_true",
                   help="few epochs - smoke test only")
    p.add_argument("--show", action="store_true", help="also display figures")
    a = p.parse_args(argv)

    cfg = Config(outdir=a.outdir, quick=a.quick, show=a.show).finalize()
    cfg.tasks = a.tasks          # type: ignore[attr-defined]
    cfg.datasets = a.datasets    # type: ignore[attr-defined]
    return cfg


def main(argv=None) -> None:
    cfg = parse_args(argv)
    set_global_seed(cfg.seed)
    print(f"TensorFlow {tf.__version__} | tasks {cfg.tasks} | datasets {cfg.datasets}"
          f"{' | QUICK' if cfg.quick else ''}")

    results: dict = {}
    for name in cfg.datasets:                       # type: ignore[attr-defined]
        data = preprocess(name, cfg)
        results[name] = {}
        if 1 in cfg.tasks:                          # type: ignore[attr-defined]
            results[name]["task1"] = run_task1(data, cfg)
        if 2 in cfg.tasks:                          # type: ignore[attr-defined]
            results[name]["task2"] = run_task2(data, cfg)
        if 3 in cfg.tasks:                          # type: ignore[attr-defined]
            results[name]["task3"] = run_task3(data, cfg)

    print(f"\nDone. Figures written to {cfg.outdir.resolve()}")


if __name__ == "__main__":
    main()
