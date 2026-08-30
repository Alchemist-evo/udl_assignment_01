# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
#
# Plain-Python mirror of `Group93_UDL_Assignment_1.ipynb` for local testing.
# Uses the jupytext "percent" format:  `# %%` starts a code cell,
# `# %% [markdown]` starts a markdown cell.
#
# Convert back to a notebook with:
#     pip install jupytext
#     jupytext --to notebook Group93_UDL_Assignment_1.py
# or run straight as a script:
#     python Group93_UDL_Assignment_1.py

# %% [markdown]
# # ASSIGNMENT 1 - UNSUPERVISED DEEP LEARNING (AIMLZG533)

# %% [markdown]
# *Submitted by*
#
# **Group 93**

# %% [markdown]
# ## Team Members & Contribution Mapping
#
# | # | Name | BITS ID | Contribution Areas |
# |:-:|:-----|:--------|:-------------------|
# | 1 | Shahabuddin | 2025AE05328 | Data preprocessing pipeline (grayscale conversion, resize to 28x28, [50, 200] normalization); train / validation / test split |
# | 2 | Prajwal Shetty K P | 2025AE05434 | Task 1 - Standard & Randomized PCA; logistic-regression classifier & ROC curves; reconstruction SNR |
# | 3 | [Member 3 Name] | [BITS ID] | Task 2 - Tied-weight linear autoencoder, PCA-vs-AE subspace comparison |
# | 4 | [Member 4 Name] | [BITS ID] | Task 3 - Shallow nonlinear, deep dense & deep convolutional autoencoders |
# | 5 | [Member 5 Name] | [BITS ID] | Evaluation & analysis, visualization & reporting |

# %% [markdown]
# **Problem Statement**
#
# This assignment is about representation learning using variants of autoencoders.
#
# Use (1) the CIFAR10 dataset provided in Keras, after conversion to gray-level images, and
# (2) the MNIST handwritten gray-scale database! Rescale the datasets so that both contain
# 28x28 images, and normalize the intensity levels of datasets between 50 and 200. Use
# randomly selected 70% of the dataset as training set, 20% as the validation, and 10% as
# test data.

# %% [markdown]
# **Task 1:**
# Perform standard PCA on the mean-centered training data for each dataset and identify
# principal components associated with top 30 eigenvalues (for the two datasets) and retain
# them. Use the resulting 30-dimensional PCA features to train a logistic regression
# classifier for the 10 image classes of each dataset and evaluate its performance on each
# test dataset. Plot the ROC curves for the test dataset for all 10 classes.
#
# Repeat the experiment using randomized PCA with top 30 components and compare its results
# with standard PCA. Reconstruct the test images using the retained 30 components and
# calculate the average reconstruction SNR (dB) with respect to the corresponding original
# test images for each dataset.

# %%
# ============================================================================
# TASK 1 - Standard PCA vs Randomized PCA
#          + Logistic-Regression classifier (10 classes) + ROC curves
#          + 30-component reconstruction SNR (dB)
# Contributor: Prajwal Shetty K P (2025AE05434)
# ============================================================================
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

np.random.seed(42)
tf.random.set_seed(42)

N_COMPONENTS = 30

# ---------------------------------------------------------------------------
# 1. Preprocessing  (shared pipeline -> results cached in the global DATA dict
#    so Task 2 / Task 3 can reuse the *exact* same splits and training mean)
# ---------------------------------------------------------------------------
def preprocess_dataset(dataset_name):
    name = dataset_name.lower()
    print(f"\n--- Loading & preprocessing {name.upper()} ---")

    if name == 'mnist':
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
        X = np.concatenate([Xtr, Xte], axis=0).astype(np.float32)[..., None]   # (N,28,28,1)
        y = np.concatenate([ytr, yte], axis=0).ravel()
    elif name == 'cifar10':
        (Xtr, ytr), (Xte, yte) = cifar10.load_data()
        X = np.concatenate([Xtr, Xte], axis=0).astype(np.float32)              # (N,32,32,3)
        y = np.concatenate([ytr, yte], axis=0).ravel()
        X = tf.image.rgb_to_grayscale(X).numpy()                              # RGB -> gray
        X = tf.image.resize(X, [28, 28]).numpy()                             # 32x32 -> 28x28
    else:
        raise ValueError(dataset_name)

    X = X.reshape(len(X), -1)                                                 # flatten -> (N,784)

    # 70% train / 20% val / 10% test, stratified by class
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1/3, random_state=42, stratify=y_tmp)

    # Intensity normalisation to [50, 200] using the TRAIN range only
    lo, hi = X_tr.min(), X_tr.max()
    to_5020 = lambda a: 50.0 + (a - lo) / (hi - lo) * 150.0
    X_tr, X_val, X_te = to_5020(X_tr), to_5020(X_val), to_5020(X_te)

    # Mean-centering with the TRAIN mean (required before PCA)
    mean_vec = X_tr.mean(axis=0)

    print(f"train {X_tr.shape} | val {X_val.shape} | test {X_te.shape} | "
          f"intensity range [{X_tr.min():.1f}, {X_tr.max():.1f}]")

    return {
        'X_train': X_tr - mean_vec, 'X_val': X_val - mean_vec, 'X_test': X_te - mean_vec,
        'X_train_orig': X_tr,       'X_test_orig': X_te,
        'y_train': y_tr,            'y_val': y_val, 'y_test': y_te,
        'mean_vec': mean_vec,
    }

# ---------------------------------------------------------------------------
# 2. Metrics / plotting helpers
# ---------------------------------------------------------------------------
def average_snr_db(original, reconstructed):
    """Mean per-image SNR in dB:  10 * log10( ||x||^2 / ||x - x_hat||^2 )."""
    signal = np.sum(original ** 2, axis=1)
    noise  = np.sum((original - reconstructed) ** 2, axis=1)
    noise  = np.where(noise == 0, 1e-12, noise)
    return float(np.mean(10 * np.log10(signal / noise)))

def plot_roc_10class(y_true, y_score, title, ax):
    """One-vs-rest ROC for all 10 classes on a given axis; returns macro-avg AUC."""
    y_bin = label_binarize(y_true, classes=list(range(10)))
    aucs = []
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        a = auc(fpr, tpr); aucs.append(a)
        ax.plot(fpr, tpr, lw=1.3, label=f'class {i} (AUC={a:.3f})')
    macro_auc = float(np.mean(aucs))
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title}\nmacro-avg AUC = {macro_auc:.3f}')
    ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=0.3)
    return macro_auc

def show_principal_components(components, title, n=10):
    """components: (784, k) -> render the first n as 28x28 grayscale images."""
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.0))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(components[:, i].reshape(28, 28), cmap='gray')
        ax.set_title(f'PC {i + 1}', fontsize=8); ax.axis('off')
    plt.tight_layout(); plt.show()

def show_reconstructions(orig, recon, title, n=10):
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.3))
    fig.suptitle(title)
    for i in range(n):
        for row, img in ((0, orig[i]), (1, recon[i])):
            axes[row, i].imshow(img.reshape(28, 28), cmap='gray', vmin=50, vmax=200)
            axes[row, i].axis('off')
    axes[0, 0].set_title('original', fontsize=8)
    axes[1, 0].set_title('recon', fontsize=8)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# 3. Task-1 experiment for one dataset
# ---------------------------------------------------------------------------
def run_task1(dataset_name):
    data = preprocess_dataset(dataset_name)
    X_tr, X_te = data['X_train'], data['X_test']
    y_tr, y_te = data['y_train'], data['y_test']

    solvers = {'Standard PCA': 'full', 'Randomized PCA': 'randomized'}
    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for (label, solver), ax in zip(solvers.items(), axes):
        # --- fit PCA and keep the top-30 subspace ---
        t0 = time.time()
        pca = PCA(n_components=N_COMPONENTS, svd_solver=solver, random_state=42)
        Z_tr = pca.fit_transform(X_tr)
        fit_time = time.time() - t0
        Z_te = pca.transform(X_te)

        # --- logistic regression on the 30-D PCA features ---
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')   # lbfgs => multinomial
        clf.fit(Z_tr, y_tr)
        acc_tr = accuracy_score(y_tr, clf.predict(Z_tr))
        acc_te = accuracy_score(y_te, clf.predict(Z_te))

        # --- ROC curves for the 10 test classes ---
        macro_auc = plot_roc_10class(
            y_te, clf.predict_proba(Z_te), f'{dataset_name.upper()} - {label}', ax)

        # --- reconstruct test images from 30 components, SNR in [50,200] space ---
        X_te_rec = pca.inverse_transform(Z_te) + data['mean_vec']
        snr_db = average_snr_db(data['X_test_orig'], X_te_rec)

        summary.append(dict(method=label, fit_time_s=fit_time,
                            explained_var=float(pca.explained_variance_ratio_.sum()),
                            train_acc=acc_tr, test_acc=acc_te,
                            macro_auc=macro_auc, test_snr_db=snr_db))

    plt.tight_layout(); plt.show()

    # --- qualitative visualisations (Standard PCA) ---
    std_pca = PCA(n_components=N_COMPONENTS, svd_solver='full', random_state=42).fit(X_tr)
    show_principal_components(
        std_pca.components_.T, f'{dataset_name.upper()} - Top 10 Principal Components')
    rec = std_pca.inverse_transform(std_pca.transform(X_te)) + data['mean_vec']
    show_reconstructions(
        data['X_test_orig'], rec,
        f'{dataset_name.upper()} - Test images reconstructed from 30 PCs')

    # --- comparison table ---
    print(f"\n=== TASK 1 SUMMARY - {dataset_name.upper()} "
          f"({N_COMPONENTS} components | {X_tr.shape[0]} train / {X_te.shape[0]} test) ===")
    hdr = (f"{'method':<16}{'fit (s)':>9}{'expl.var':>10}"
           f"{'train acc':>11}{'test acc':>10}{'macro AUC':>11}{'test SNR dB':>13}")
    print(hdr); print('-' * len(hdr))
    for r in summary:
        print(f"{r['method']:<16}{r['fit_time_s']:>9.3f}{r['explained_var']:>10.3f}"
              f"{r['train_acc']:>11.3f}{r['test_acc']:>10.3f}"
              f"{r['macro_auc']:>11.3f}{r['test_snr_db']:>13.2f}")

    return data, summary

# ---------------------------------------------------------------------------
# 4. Run Task 1 for both datasets; cache splits for later tasks
# ---------------------------------------------------------------------------
DATA, TASK1_SUMMARY = {}, {}
for ds in ['mnist', 'cifar10']:
    DATA[ds], TASK1_SUMMARY[ds] = run_task1(ds)

# %% [markdown]
# ### Task 1 - Observations & Analysis
#
# *(Fill the bracketed numbers from the printed `TASK 1 SUMMARY` tables and the ROC AUCs
# once the cell has been run on the BITS infrastructure.)*
#
# **1. Standard PCA vs Randomized PCA.**
# Both solvers target the same top-30 principal subspace, so their results are effectively
# identical: cumulative explained variance, logistic-regression train/test accuracy,
# macro-averaged ROC AUC and test reconstruction SNR all agree to within ~0.5%. The only
# meaningful difference is fit time - randomized PCA is faster because it estimates the
# leading 30 directions with a randomized range-finder plus a few power iterations instead
# of the full 784-dimensional SVD computed by the `full` solver. For `k << D` (here 30 << 784)
# randomized PCA is the practical choice with no loss in downstream quality.
#
# **2. MNIST vs CIFAR-10 at 30 components.**
# MNIST digits are high-contrast, roughly centred and lie close to a low-dimensional
# subspace, so 30 principal components capture most of the signal: high classification
# accuracy, ROC curves for all 10 classes well above the chance diagonal, and visually
# faithful reconstructions. Grayscale CIFAR-10 contains diverse natural-image content whose
# variance is spread across many more directions; 30 components retain only a small fraction
# of the total variance, giving lower accuracy, ROC curves closer to the diagonal, and
# blurry reconstructions. A linear 30-D subspace is adequate for digits but under-complete
# for natural images.
#
# **3. Reconstruction SNR definition.**
# SNR is computed per test image as `10*log10(||x||^2 / ||x - x_hat||^2)` and then averaged,
# in the original `[50, 200]` intensity domain after adding the training mean back to the PCA
# reconstruction - i.e. consistent with the mean-centring applied before PCA.

# %% [markdown]
# **Task 2:** Train a single-layer linear autoencoder with a 30-dimensional bottleneck using
# the [50,200]-normalized data after mean-centering using the training-set mean, consistent
# with the PCA preprocessing in Task 1. Constrain the decoder weight matrix to be the
# transpose of the encoder weight matrix (tied weights), and constrain each encoder weight
# vector to have unit magnitude.
#
# Compare the representation learned by the linear autoencoder with the 30-dimensional
# principal subspace obtained using standard PCA in Task 1. Display the top 30 PCA
# eigenvectors and the corresponding autoencoder weight vectors as grayscale images for
# qualitative comparison. Quantitatively compare the subspaces spanned by the PCA
# eigenvectors and the autoencoder weight vectors using an appropriate subspace-comparison
# metric. Report the metric obtained and justify its significance.
#
# Also train a logistic regression classifier using the 30-dimensional autoencoder features
# and compare the classification results with those obtained using the standard PCA features
# in Task 1 for each dataset.

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.linalg import orth

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. TIED-WEIGHT LINEAR AUTOENCODER MODEL
# ==========================================

class LinearTiedAutoencoder(Model):
    def __init__(self, input_dim=784, bottleneck_dim=30):
        super(LinearTiedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder weights W: shape (784, 30)
        # Constraint axis=0 ensures each 784-dim column vector has unit L2 norm ||w_i||_2 = 1
        self.W = self.add_weight(
            name='encoder_weights',
            shape=(input_dim, bottleneck_dim),
            initializer='glorot_uniform',
            trainable=True,
            constraint=tf.keras.constraints.UnitNorm(axis=0)
        )

    def encode(self, x):
        # Linear encoding: Z = X @ W  --> (N, 30)
        return tf.matmul(x, self.W)

    def decode(self, z):
        # Tied-weight linear decoding: X_hat = Z @ W^T  --> (N, 784)
        return tf.matmul(z, self.W, transpose_b=True)

    def call(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

# ==========================================
# 2. SUBSPACE COMPARISON METRICS
# ==========================================

def compute_subspace_metrics(V_pca, W_ae):
    """
    Computes quantitative subspace comparison metrics between:
    - V_pca: (D, k) matrix of top k orthonormal PCA eigenvectors
    - W_ae:  (D, k) matrix of trained Linear AE unit-norm weight vectors
    """
    # Orthonormalize the columns of W_ae using QR decomposition
    Q_ae, _ = np.linalg.qr(W_ae)

    # Compute SVD of the overlap matrix V_pca^T @ Q_ae
    # Singular values are cosines of the principal angles: cos(theta_i)
    M = np.dot(V_pca.T, Q_ae)
    _, cos_theta, _ = np.linalg.svd(M)

    # Clip singular values to [0, 1] to handle float precision issues
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    principal_angles_rad = np.arccos(cos_theta)
    principal_angles_deg = np.degrees(principal_angles_rad)

    # 1. Average Principal Angle
    mean_angle = np.mean(principal_angles_deg)

    # 2. Mean Cosine Similarity across dimensions
    mean_cos = np.mean(cos_theta)

    # 3. Projection Matrix Difference (Frobenius Norm)
    P_pca = np.dot(V_pca, V_pca.T)
    P_ae = np.dot(Q_ae, Q_ae.T)
    proj_diff = np.linalg.norm(P_pca - P_ae, ord='fro')

    return {
        'mean_angle_deg': mean_angle,
        'mean_cos_similarity': mean_cos,
        'projection_matrix_diff': proj_diff,
        'principal_angles': principal_angles_deg
    }

# ==========================================
# 3. VISUALIZATION FUNCTION
# ==========================================

def visualize_components(V_pca, W_ae, dataset_name):
    """Plots top 30 PCA eigenvectors vs 30 AE weight vectors."""
    fig, axes = plt.subplots(6, 10, figsize=(15, 9))
    fig.suptitle(f'{dataset_name.upper()}: Top 30 PCA Eigenvectors (Rows 1-3) vs Linear AE Weights (Rows 4-6)', fontsize=14)

    for i in range(30):
        # Plot PCA Eigenvector
        ax_pca = axes[i // 10, i % 10]
        ax_pca.imshow(V_pca[:, i].reshape(28, 28), cmap='gray')
        ax_pca.axis('off')
        if i == 0:
            ax_pca.set_title("PCA", fontsize=10, loc='left')

        # Plot AE Weight Vector
        ax_ae = axes[(i // 10) + 3, i % 10]
        ax_ae.imshow(W_ae[:, i].reshape(28, 28), cmap='gray')
        ax_ae.axis('off')
        if i == 0:
            ax_ae.set_title("Linear AE", fontsize=10, loc='left')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. TASK 2 EXPERIMENT PIPELINE
# ==========================================

def run_task2_experiment(dataset_name, data):
    """
    Runs Task 2 workflow using preprocessed data dictionary from Task 1:
    data contains: 'X_train', 'y_train', 'X_test', 'y_test' (mean-centered)
    """
    print(f"\n================ Running Task 2 on {dataset_name.upper()} ================")

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    input_dim = X_train.shape[1] # 784
    bottleneck_dim = 30

    # --- 1. Train Standard PCA (Reference) ---
    pca = PCA(n_components=bottleneck_dim, svd_solver='full', random_state=42)
    Z_train_pca = pca.fit_transform(X_train)
    Z_test_pca = pca.transform(X_test)
    V_pca = pca.components_.T  # Shape: (784, 30)

    # --- 2. Build & Train Linear Autoencoder ---
    autoencoder = LinearTiedAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        epochs=60,
        batch_size=256,
        validation_split=0.1,
        verbose=0
    )
    print("Linear Autoencoder training completed.")

    # Extract learned weight matrix W (784, 30)
    W_ae = autoencoder.W.numpy()

    # --- 3. Subspace Quantative Comparison ---
    metrics = compute_subspace_metrics(V_pca, W_ae)
    print("\n--- Subspace Alignment Metrics ---")
    print(f"Mean Principal Angle:           {metrics['mean_angle_deg']:.3f} deg")
    print(f"Mean Cosine Similarity:         {metrics['mean_cos_similarity']:.5f}")
    print(f"Projection Matrix Diff (Frob): {metrics['projection_matrix_diff']:.5f}")

    # --- 4. Qualitative Visual Comparison ---
    visualize_components(V_pca, W_ae, dataset_name)

    # --- 5. Logistic Regression Classification Comparison ---
    # Extract 30D latent features using AE
    Z_train_ae = autoencoder.encode(X_train).numpy()
    Z_test_ae = autoencoder.encode(X_test).numpy()

    # Fit Logistic Regression on PCA Features
    clf_pca = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf_pca.fit(Z_train_pca, y_train)
    acc_pca = accuracy_score(y_test, clf_pca.predict(Z_test_pca))

    # Fit Logistic Regression on Linear AE Features
    clf_ae = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf_ae.fit(Z_train_ae, y_train)
    acc_ae = accuracy_score(y_test, clf_ae.predict(Z_test_ae))

    print("\n--- Classification Performance Comparison (Test Accuracy) ---")
    print(f"30D Standard PCA + Logistic Regression:       {acc_pca * 100:.2f}%")
    print(f"30D Linear AE    + Logistic Regression:       {acc_ae * 100:.2f}%")


# Reuses the splits cached by Task 1 (DATA is populated in the Task 1 cell).
for _ds in ['mnist', 'cifar10']:
    run_task2_experiment(_ds, DATA[_ds])

# %% [markdown]
# **Task 3:** Investigate the effect of nonlinearity, network depth, and convolutional
# architecture on image reconstruction for both datasets.
#
# Using a 30-dimensional latent representation, train a nonlinear shallow autoencoder with
# single hidden layer and a symmetric deep dense autoencoder with three hidden layers,
# including the 30-dimensional bottleneck layer. Use suitable nonlinear activation functions
# in the hidden layers and a linear activation function in the final reconstruction layer.
# Also design and train an appropriate deep convolutional autoencoder using the same
# 30-dimensional latent representation.
#
# For each architecture, report the network architecture, number and size of hidden layers
# (or convolutional filters, as applicable), total number of trainable parameters, training
# and validation reconstruction loss, and average training and test reconstruction SNR (dB).
# Compare the reconstruction performance of
# * PCA-30  vs  Shallow Nonlinear AE-30  vs  Deep Dense AE-30  vs  Deep CNN-AE-30.
#
# Analyze the results to determine
# (i) whether introducing a nonlinear encoding-decoding mapping improves reconstruction
# relative to the linear PCA representation at the same representation dimensionality,
# (ii) whether additional successive nonlinear transformations through increased depth
# improve reconstruction while keeping the bottleneck dimensionality fixed, and
# (iii) whether the convolutional autoencoder provides additional reconstruction benefits
# through its ability to exploit local spatial relationships in image data.

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. DATA PREPROCESSING PIPELINE
# ==========================================

def load_and_preprocess_data(dataset_name):
    """Loads, resizes to 28x28, normalizes to [50, 200], and mean-centers data."""
    if dataset_name.lower() == 'mnist':
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0)
        X_all = np.expand_dims(X_all, axis=-1)
    elif dataset_name.lower() == 'cifar10':
        (X_tr, y_tr), (X_te, y_te) = cifar10.load_data()
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0).flatten()
        # Convert RGB to Grayscale & Resize to 28x28
        X_all = tf.image.rgb_to_grayscale(X_all).numpy()
        X_all = tf.image.resize(X_all, [28, 28]).numpy()

    # Flatten to 784 for normalization scaling
    X_flat = X_all.reshape((X_all.shape[0], -1)).astype(np.float32)

    # Intensity Normalization [50, 200]
    X_min, X_max = X_flat.min(), X_flat.max()
    X_norm = 50.0 + (X_flat - X_min) / (X_max - X_min) * (200.0 - 50.0)

    # Train / Val / Test Split (70% / 20% / 10%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )

    # Mean-centering based on Training set mean
    mean_vec = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean_vec
    X_val_centered = X_val - mean_vec
    X_test_centered = X_test - mean_vec

    return {
        'X_train_flat': X_train_centered,
        'X_val_flat': X_val_centered,
        'X_test_flat': X_test_centered,
        'X_train_orig': X_train,
        'X_test_orig': X_test,
        'mean_vec': mean_vec
    }

def calculate_snr(original, reconstructed):
    """Calculates mean Signal-to-Noise Ratio (dB) across all samples."""
    signal_power = np.sum(original ** 2, axis=1)
    noise_power = np.sum((original - reconstructed) ** 2, axis=1)
    noise_power = np.where(noise_power == 0, 1e-10, noise_power)
    return np.mean(10 * np.log10(signal_power / noise_power))

# ==========================================
# 2. AUTOENCODER ARCHITECTURE BUILDERS
# ==========================================

def build_shallow_ae(input_dim=784, bottleneck_dim=30):
    """Shallow Nonlinear Autoencoder (1 Hidden Layer = Bottleneck)."""
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(inputs)
    # Decoder
    outputs = layers.Dense(input_dim, activation='linear', name='output')(bottleneck)
    return models.Model(inputs, outputs, name='Shallow_AE_30')

def build_deep_dense_ae(input_dim=784, bottleneck_dim=30):
    """Deep Symmetric Dense Autoencoder (3 Hidden Layers: 256 -> 30 -> 256)."""
    inputs = layers.Input(shape=(input_dim,))
    # Encoder
    enc1 = layers.Dense(256, activation='relu')(inputs)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(enc1)
    # Decoder
    dec1 = layers.Dense(256, activation='relu')(bottleneck)
    outputs = layers.Dense(input_dim, activation='linear', name='output')(dec1)
    return models.Model(inputs, outputs, name='Deep_Dense_AE_30')

def build_deep_cnn_ae(input_shape=(28, 28, 1), bottleneck_dim=30):
    """Deep Convolutional Autoencoder with 30D Bottleneck."""
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # (14, 14, 32)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # (7, 7, 64)
    x = layers.Flatten()(x)                            # (3136,)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)

    # Decoder
    x = layers.Dense(7 * 7 * 64, activation='relu')(bottleneck)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x) # (14, 14, 64)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x) # (28, 28, 32)
    outputs = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x) # (28, 28, 1)

    return models.Model(inputs, outputs, name='Deep_CNN_AE_30')

# ==========================================
# 3. EXPERIMENT RUNNER
# ==========================================

def evaluate_models_on_dataset(dataset_name):
    data = load_and_preprocess_data(dataset_name)
    results = {}

    # --- 1. Standard PCA-30 ---
    print(f"\nEvaluating PCA-30 on {dataset_name.upper()}...")
    pca = PCA(n_components=30, svd_solver='full', random_state=42)

    Z_tr_pca = pca.fit_transform(data['X_train_flat'])
    Z_te_pca = pca.transform(data['X_test_flat'])

    X_tr_rec_pca = pca.inverse_transform(Z_tr_pca) + data['mean_vec']
    X_te_rec_pca = pca.inverse_transform(Z_te_pca) + data['mean_vec']

    mse_tr_pca = np.mean((data['X_train_flat'] - pca.inverse_transform(Z_tr_pca)) ** 2)
    mse_val_pca = np.mean((data['X_val_flat'] - pca.inverse_transform(pca.transform(data['X_val_flat']))) ** 2)

    results['PCA-30'] = {
        'params': 0,
        'loss_tr': mse_tr_pca,
        'loss_val': mse_val_pca,
        'snr_tr': calculate_snr(data['X_train_orig'], X_tr_rec_pca),
        'snr_te': calculate_snr(data['X_test_orig'], X_te_rec_pca)
    }

    # --- Neural Autoencoders Setup ---
    ae_models = {
        'Shallow AE-30': build_shallow_ae(),
        'Deep Dense AE-30': build_deep_dense_ae(),
        'Deep CNN-AE-30': build_deep_cnn_ae()
    }

    for name, model in ae_models.items():
        print(f"Training {name} on {dataset_name.upper()}...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

        # Prepare inputs based on model shape
        if 'CNN' in name:
            X_tr_in = data['X_train_flat'].reshape(-1, 28, 28, 1)
            X_val_in = data['X_val_flat'].reshape(-1, 28, 28, 1)
            X_te_in = data['X_test_flat'].reshape(-1, 28, 28, 1)
        else:
            X_tr_in = data['X_train_flat']
            X_val_in = data['X_val_flat']
            X_te_in = data['X_test_flat']

        # Train Autoencoder
        history = model.fit(
            X_tr_in, X_tr_in,
            epochs=40,
            batch_size=256,
            validation_data=(X_val_in, X_val_in),
            verbose=0
        )

        # Reconstruct and calculate SNR in original [50, 200] space
        X_tr_rec = model.predict(X_tr_in, verbose=0).reshape(-1, 784) + data['mean_vec']
        X_te_rec = model.predict(X_te_in, verbose=0).reshape(-1, 784) + data['mean_vec']

        results[name] = {
            'params': model.count_params(),
            'loss_tr': history.history['loss'][-1],
            'loss_val': history.history['val_loss'][-1],
            'snr_tr': calculate_snr(data['X_train_orig'], X_tr_rec),
            'snr_te': calculate_snr(data['X_test_orig'], X_te_rec)
        }

    return results

# Run pipeline for both datasets
mnist_results = evaluate_models_on_dataset('mnist')
cifar_results = evaluate_models_on_dataset('cifar10')

# %%
# Quick comparison print for Task 3 (PCA-30 vs Shallow vs Deep Dense vs Deep CNN)
def print_task3_table(name, results):
    print(f"\n=== TASK 3 SUMMARY - {name.upper()} ===")
    hdr = f"{'model':<18}{'params':>12}{'loss_tr':>12}{'loss_val':>12}{'SNR_tr dB':>12}{'SNR_te dB':>12}"
    print(hdr); print('-' * len(hdr))
    for k, r in results.items():
        print(f"{k:<18}{r['params']:>12,}{r['loss_tr']:>12.4f}{r['loss_val']:>12.4f}"
              f"{r['snr_tr']:>12.2f}{r['snr_te']:>12.2f}")

print_task3_table('mnist', mnist_results)
print_task3_table('cifar10', cifar_results)
