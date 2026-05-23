"""
=============================================================================
  DEEP LEARNING — NEURAL NETWORK CLASSIFIER
  Omalizumab Responder vs Non-Responder — 5-Gene Signature
  Lopez-Rincon et al. (2021/2023)

  Three network depths compared:
    Model A — 2-layer  network  (shallow)
    Model B — 5-layer  network  (medium)
    Model C — 7-layer  network  (deep)

  Same premise as Linear vs Logistic script:
    - Load real 39-patient data
    - Use the 5-gene signature
    - Stratified 5-Fold Cross-Validation
    - Measure Accuracy, F1, AUC-ROC
    - Visualise training curves + decision boundaries + ROC curves
=============================================================================

WHAT IS A NEURAL NETWORK?
  A neural network is a chain of mathematical layers.
  Each layer transforms the data, passing it to the next.
  The DEPTH (number of layers) controls how complex the
  patterns it can learn are.

  Input (5 genes)
      ↓
  [Hidden Layer 1]  → learns simple patterns
      ↓
  [Hidden Layer 2]  → learns combinations of patterns
      ↓
      ...           → deeper = more complex patterns
      ↓
  Output → P(Responder)  [0.0 to 1.0]

EACH LAYER DOES:
  output = activation_function( weights × input + bias )
  The activation function (ReLU or Sigmoid) adds non-linearity
  so the network can learn curved, complex decision boundaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# SECTION 1 — LOAD REAL PATIENT DATA
#
# Same loading logic as the logistic regression script.
# 39 patients, 5 gene expression values each, binary label.
# =============================================================================

print("=" * 65)
print("  SECTION 1 — LOADING REAL PATIENT DATA")
print("=" * 65)

features_df = pd.read_csv('\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\features_0.csv')
probe_ids   = [features_df.columns[0]] + list(features_df.iloc[:, 0])

data = pd.read_csv('\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\data_0.csv', header=0)
data.columns = probe_ids

labels_df = pd.read_csv('\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\labels.csv', header=0)
y = labels_df.iloc[:, 0].values  # 1=Responder, 0=Non-Responder

FIVE_GENES = {
    'CCDC113':       'ILMN_1775520',
    'SLC26A8':       'ILMN_1893728',
    'PPP1R3D':       'ILMN_1757262',
    'LOC100131780':  'ILMN_1680347',
    'CLEC4C':        'ILMN_3306019',
}

X = data[[probe for probe in FIVE_GENES.values()]].values
gene_names = list(FIVE_GENES.keys())

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Patients         : {X.shape[0]}")
print(f"  Genes (features) : {X.shape[1]} → {gene_names}")
print(f"  Responders  (1)  : {(y==1).sum()}")
print(f"  Non-Resp.   (0)  : {(y==0).sum()}")


# =============================================================================
# SECTION 2 — DEFINE THE THREE NEURAL NETWORK ARCHITECTURES
#
# All three networks:
#   - Take 5 inputs (one per gene)
#   - Output 1 value → sigmoid → P(Responder)
#   - Use ReLU activation between hidden layers
#   - Use Dropout to prevent overfitting (critical with only 39 samples)
#   - Use Batch Normalisation to stabilise training
#
# LAYER ANATOMY:
#   Linear(in, out)    → matrix multiplication: output = W*input + b
#   BatchNorm1d(out)   → normalises each layer's output (faster, stable training)
#   ReLU()             → activation: max(0, x) — adds non-linearity
#   Dropout(p)         → randomly zeros p% of neurons during training
#                        forces the network to not rely on any single neuron
#   Sigmoid()          → final squeeze to [0,1] = probability
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 2 — NEURAL NETWORK ARCHITECTURES")
print("=" * 65)

INPUT_DIM = 5   # 5 genes
OUTPUT_DIM = 1  # P(Responder)


class NeuralNet2Layer(nn.Module):
    """
    2-LAYER NETWORK (Shallow)
    ─────────────────────────
    Input(5) → Hidden(16) → Hidden(8) → Output(1)

    Only 2 hidden layers. Learns simple, linear-ish boundaries.
    Least expressive but least likely to overfit on small data.
    Best suited when the signal is strong and simple — which is
    true here since the 5 genes already separate classes well.

    Architecture diagram:
      [5 genes] → [16 neurons] → [8 neurons] → [P(Responder)]
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Hidden Layer 1: 5 inputs → 16 neurons
            nn.Linear(5, 16),           # learns 5×16=80 weights + 16 biases
            nn.BatchNorm1d(16),         # normalise outputs of this layer
            nn.ReLU(),                  # activation: kill negative values
            nn.Dropout(0.3),            # drop 30% of neurons randomly

            # Hidden Layer 2: 16 → 8 neurons
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output Layer: 8 → 1 probability
            nn.Linear(8, 1),
            nn.Sigmoid()                # squish to [0,1]
        )

    def forward(self, x):
        return self.network(x)


class NeuralNet5Layer(nn.Module):
    """
    5-LAYER NETWORK (Medium)
    ────────────────────────
    Input(5) → 32 → 64 → 32 → 16 → 8 → Output(1)

    5 hidden layers. Bottleneck architecture — expands then contracts.
    Learns more complex gene interaction patterns.
    Medium overfitting risk on small datasets.

    Architecture diagram:
      [5] → [32] → [64] → [32] → [16] → [8] → [P(Responder)]
                    ↑ widest point (bottleneck peak)
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 64),          # expand — learns richer representations
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),          # contract — compresses learned features
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class NeuralNet7Layer(nn.Module):
    """
    7-LAYER NETWORK (Deep)
    ──────────────────────
    Input(5) → 32 → 64 → 128 → 64 → 32 → 16 → 8 → Output(1)

    7 hidden layers. Diamond architecture — expands to 128, then contracts.
    Most expressive — can learn very complex non-linear boundaries.
    Highest overfitting risk with only 39 samples.
    Requires stronger regularisation (higher Dropout).

    Architecture diagram:
      [5]→[32]→[64]→[128]→[64]→[32]→[16]→[8]→[P(Responder)]
                      ↑ widest point
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),            # higher dropout for deeper net

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 128),         # peak width — most expressive layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Print architecture summaries
model_classes = {
    '2-Layer  (Shallow)': NeuralNet2Layer,
    '5-Layer  (Medium) ': NeuralNet5Layer,
    '7-Layer  (Deep)   ': NeuralNet7Layer,
}

print(f"\n  {'Model':<22}  {'Parameters':>12}  {'Architecture'}")
print(f"  {'-'*70}")
for name, cls in model_classes.items():
    m = cls()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    layers = [str(l) for l in m.network if isinstance(l, nn.Linear)]
    arch = ' → '.join([l.split('(')[1].split(',')[0] for l in layers])
    print(f"  {name}  {n_params:>12,}  5 → {arch} → 1")


# =============================================================================
# SECTION 3 — TRAINING FUNCTION
#
# HOW NEURAL NETWORK TRAINING WORKS (one epoch):
#
#   1. FORWARD PASS
#      Feed patient gene values → network → get P(Responder)
#
#   2. COMPUTE LOSS
#      Compare predicted P to actual label (0 or 1)
#      Loss = Binary Cross-Entropy:
#        BCE = -[y*log(P) + (1-y)*log(1-P)]
#      If y=1 (Responder) and P=0.9 → low loss (good)
#      If y=1 (Responder) and P=0.1 → high loss (bad)
#
#   3. BACKWARD PASS (backpropagation)
#      Compute gradient: how much does each weight affect the loss?
#      dLoss/dWeight for every weight in every layer
#
#   4. UPDATE WEIGHTS (gradient descent)
#      weight = weight - learning_rate × gradient
#      Small step in the direction that reduces loss
#
#   Repeat for N epochs until loss converges.
# =============================================================================

def train_one_fold(model_class, X_train, y_train, X_val, y_val,
                   epochs=300, lr=0.001, batch_size=16):
    """
    Trains one neural network on one fold of data.
    Returns: trained model, train losses, val losses
    """
    # Convert numpy arrays → PyTorch tensors (the data format PyTorch needs)
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train).unsqueeze(1)   # shape: (n,1)
    X_vl = torch.FloatTensor(X_val)
    y_vl = torch.FloatTensor(y_val).unsqueeze(1)

    # DataLoader batches the data for mini-batch gradient descent
    dataset    = TensorDataset(X_tr, y_tr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialise model and optimizer
    model     = model_class()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=1e-3)   # L2 regularisation
    criterion = nn.BCELoss()                    # Binary Cross-Entropy loss

    # Learning rate scheduler — reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5
    )

    train_losses = []
    val_losses   = []

    for epoch in range(epochs):

        # ── TRAINING MODE ──
        # Dropout IS active, BatchNorm uses batch statistics
        model.train()
        batch_losses = []

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()           # clear previous gradients

            y_hat = model(X_batch)          # forward pass
            loss  = criterion(y_hat, y_batch)  # compute loss
            loss.backward()                 # backpropagation
            optimizer.step()               # update weights

            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # ── EVALUATION MODE ──
        # Dropout is OFF, BatchNorm uses running statistics
        model.eval()
        with torch.no_grad():              # no gradient computation needed
            val_out  = model(X_vl)
            val_loss = criterion(val_out, y_vl).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

    return model, train_losses, val_losses


# =============================================================================
# SECTION 4 — STRATIFIED 5-FOLD CROSS-VALIDATION
#
# Identical strategy to the logistic regression script:
#   - 5 folds, stratified (preserve 77/23 class ratio)
#   - Each patient predicted exactly once on held-out data
#   - But now instead of sklearn's cross_val_predict,
#     we manually loop because PyTorch needs custom training
#
# For each fold:
#   Train neural network on 4 folds (31 patients)
#   Predict on remaining fold (8 patients)
#   Store predictions → compute metrics at the end
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 3 — STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_configs = {
    '2-Layer':  NeuralNet2Layer,
    '5-Layer':  NeuralNet5Layer,
    '7-Layer':  NeuralNet7Layer,
}

# Storage for results
all_results     = {}   # metrics per model
all_train_loss  = {}   # training curves per model
all_val_loss    = {}   # validation curves per model
all_probs       = {}   # predicted probabilities per model

for model_name, model_class in model_configs.items():
    print(f"\n  Training {model_name} network...")

    fold_probs   = np.zeros(len(y))   # store predicted prob for each patient
    fold_labels  = np.zeros(len(y))   # store true label for each patient
    all_tr_loss  = []                 # avg train loss across folds per epoch
    all_vl_loss  = []                 # avg val loss across folds per epoch

    fold_tr_losses = []
    fold_vl_losses = []

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):

        # Split data into this fold's train and validation sets
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx],         y[val_idx]

        # Train the network on this fold
        model, tr_losses, vl_losses = train_one_fold(
            model_class, X_train, y_train, X_val, y_val,
            epochs=300, lr=0.001
        )

        # Predict probabilities for the held-out validation patients
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val)
            probs   = model(X_val_t).squeeze().numpy()

        # Handle case where val set has only 1 sample (scalar → array)
        if probs.ndim == 0:
            probs = np.array([probs])

        fold_probs[val_idx]  = probs
        fold_labels[val_idx] = y_val

        fold_tr_losses.append(tr_losses)
        fold_vl_losses.append(vl_losses)

        # Print fold progress
        fold_acc = accuracy_score(y_val, (probs > 0.5).astype(int))
        print(f"    Fold {fold_num+1}/5 — "
              f"Val patients: {len(val_idx)} — "
              f"Fold accuracy: {fold_acc:.3f}")

    # Average loss curves across all 5 folds
    all_train_loss[model_name] = np.mean(fold_tr_losses, axis=0)
    all_val_loss[model_name]   = np.mean(fold_vl_losses, axis=0)
    all_probs[model_name]      = fold_probs

    # Compute final metrics from all 39 cross-validated predictions
    y_pred_class = (fold_probs > 0.5).astype(int)
    acc  = accuracy_score(fold_labels, y_pred_class)
    f1   = f1_score(fold_labels, y_pred_class, average='weighted')
    auc  = roc_auc_score(fold_labels, fold_probs)
    cm   = confusion_matrix(fold_labels, y_pred_class)

    all_results[model_name] = {
        'accuracy': acc, 'f1': f1, 'auc': auc, 'cm': cm,
        'probs': fold_probs, 'preds': y_pred_class
    }

    print(f"\n    ── {model_name} RESULTS ──")
    print(f"    Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      Actual Non-Resp  | Pred Non-Resp: {cm[0,0]}  Pred Resp: {cm[0,1]}")
    print(f"      Actual Responder | Pred Non-Resp: {cm[1,0]}  Pred Resp: {cm[1,1]}")


# =============================================================================
# SECTION 5 — PRINT FULL COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 4 — FULL RESULTS COMPARISON")
print("=" * 65)

print(f"\n  {'Model':<12}  {'Accuracy':>9}  {'F1':>7}  {'AUC-ROC':>9}  {'Layers':>7}")
print(f"  {'-'*52}")
layer_counts = {'2-Layer': 2, '5-Layer': 5, '7-Layer': 7}
for name, res in all_results.items():
    print(f"  {name:<12}  {res['accuracy']:>9.4f}  "
          f"{res['f1']:>7.4f}  {res['auc']:>9.4f}  "
          f"{layer_counts[name]:>7}")

best = max(all_results, key=lambda k: all_results[k]['auc'])
print(f"\n  ✔ Best model by AUC-ROC: {best}  "
      f"(AUC = {all_results[best]['auc']:.4f})")

print(f"\n  Classification Reports:")
for name, res in all_results.items():
    print(f"\n  [{name}]")
    report = classification_report(
        y, res['preds'], target_names=['Non-Responder','Responder']
    )
    for line in report.split('\n'):
        print(f"    {line}")


# =============================================================================
# SECTION 6 — VISUALISATION (5-panel figure)
#
# Panel 1: Network architecture diagrams (2 vs 5 vs 7 layers)
# Panel 2: Training loss curves (train vs validation per model)
# Panel 3: Predicted P(Responder) per patient (all 3 models)
# Panel 4: ROC curves (all 3 models + logistic regression baseline)
# Panel 5: Accuracy/F1/AUC bar chart comparison
# =============================================================================

print("\nGenerating visualisation...")

COLORS = {
    '2-Layer': '#2255aa',
    '5-Layer': '#1D9E75',
    '7-Layer': '#D85A30',
}
COLOR_RESP    = '#1D9E75'
COLOR_NONRESP = '#D85A30'

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#fafaf9')
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.45, wspace=0.35)


# ── PANEL 1: Architecture Diagram ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # full top row
ax1.set_facecolor('#f5f4f1')
ax1.set_title('Neural Network Architectures — 2 Layer vs 5 Layer vs 7 Layer',
              fontsize=13, fontweight='500', color='#222222', pad=10)
ax1.axis('off')

arch_configs = {
    '2-Layer\n(Shallow)': [5, 16, 8, 1],
    '5-Layer\n(Medium)':  [5, 32, 64, 32, 16, 8, 1],
    '7-Layer\n(Deep)':    [5, 32, 64, 128, 64, 32, 16, 8, 1],
}

x_starts  = [0.04, 0.37, 0.67]
net_colors = ['#2255aa', '#1D9E75', '#D85A30']

for net_idx, (net_name, layers) in enumerate(arch_configs.items()):
    x_base  = x_starts[net_idx]
    n_layers = len(layers)
    spacing  = 0.27 / n_layers
    max_n    = max(layers)

    for li, n_neurons in enumerate(layers):
        x = x_base + li * spacing
        neuron_spacing = 0.18 / (max_n + 1)
        y_center = 0.5

        # Draw neurons
        for ni in range(min(n_neurons, 6)):    # show max 6 neurons
            y = y_center + (ni - min(n_neurons, 6) / 2) * neuron_spacing
            circle = plt.Circle(
                (x, y), 0.012,
                color=net_colors[net_idx], alpha=0.8,
                transform=ax1.transAxes, zorder=3
            )
            ax1.add_patch(circle)

        # "..." if more than 6 neurons
        if n_neurons > 6:
            ax1.text(x, y_center - 0.09, '⋮',
                     transform=ax1.transAxes, ha='center',
                     fontsize=14, color=net_colors[net_idx], alpha=0.7)

        # Layer label below
        layer_label = (
            f'Input\n({n_neurons})' if li == 0 else
            f'Output\n({n_neurons})' if li == n_layers - 1 else
            f'H{li}\n({n_neurons})'
        )
        ax1.text(x, 0.16, layer_label,
                 transform=ax1.transAxes, ha='center',
                 fontsize=7, color='#555555')

        # Draw connections to next layer
        if li < n_layers - 1:
            x_next = x_base + (li + 1) * spacing
            n_next = layers[li + 1]
            for ni in range(min(n_neurons, 6)):
                y1 = y_center + (ni - min(n_neurons,6)/2) * neuron_spacing
                for nj in range(min(n_next, 6)):
                    y2 = y_center + (nj - min(n_next,6)/2) * neuron_spacing
                    ax1.plot([x, x_next], [y1, y2],
                             transform=ax1.transAxes,
                             color=net_colors[net_idx],
                             alpha=0.08, linewidth=0.5, zorder=1)

    # Network title
    ax1.text(x_base + (n_layers - 1) * spacing / 2,
             0.93, net_name,
             transform=ax1.transAxes, ha='center',
             fontsize=11, fontweight='600',
             color=net_colors[net_idx])

    # Parameter count
    m = list(model_configs.values())[net_idx]()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    ax1.text(x_base + (n_layers - 1) * spacing / 2,
             0.05, f'{n_params:,} parameters',
             transform=ax1.transAxes, ha='center',
             fontsize=8, color='#888888', style='italic')

# Divider lines between networks
for xd in [0.345, 0.655]:
    ax1.axvline(xd, color='#dddddd', linewidth=1, linestyle='--')

# Common labels
ax1.text(0.01, 0.5, 'Each node = neuron\nLines = weights',
         transform=ax1.transAxes, fontsize=8, color='#888888',
         va='center', style='italic')


# ── PANEL 2: Training Loss Curves ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#f5f4f1')
ax2.set_title('Training vs Validation Loss\n(averaged across 5 folds)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

epochs_range = range(1, 301)
for name, color in COLORS.items():
    tr = all_train_loss[name]
    vl = all_val_loss[name]
    ax2.plot(epochs_range, tr, color=color, linewidth=1.5,
             label=f'{name} Train', alpha=0.9)
    ax2.plot(epochs_range, vl, color=color, linewidth=1.5,
             linestyle='--', label=f'{name} Val', alpha=0.6)

ax2.set_xlabel('Epoch', fontsize=9, color='#444444')
ax2.set_ylabel('Binary Cross-Entropy Loss', fontsize=9, color='#444444')
ax2.legend(fontsize=7, framealpha=0.9, ncol=2)
ax2.spines[['top','right']].set_visible(False)
ax2.tick_params(labelsize=8, colors='#666666')
ax2.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
ax2.text(0.97, 0.97,
         'Train (solid) vs Val (dashed)\nGap = overfitting',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=7.5, color='#666666',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#dddddd', alpha=0.9))


# ── PANEL 3: Predicted Probabilities per Patient ───────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#f5f4f1')
ax3.set_title('P(Responder) per Patient\n(cross-validated predictions)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

x_pos = np.arange(len(y_labels := labels_df.iloc[:, 0].values))
width = 0.25
offsets = [-width, 0, width]

for idx, (name, color) in enumerate(COLORS.items()):
    probs = all_results[name]['probs']
    sort_idx = np.argsort(probs)
    bar_colors = [COLOR_RESP if y_labels[i]==1 else COLOR_NONRESP
                  for i in sort_idx]
    ax3.bar(x_pos + offsets[idx], probs[sort_idx],
            width=width, color=bar_colors, alpha=0.55,
            edgecolor=color, linewidth=0.8, label=name)

ax3.axhline(0.5, color='#333333', linewidth=1.5, linestyle='--',
            label='Decision threshold')
ax3.set_xlabel('Patients (sorted by 2-Layer prediction)',
               fontsize=9, color='#444444')
ax3.set_ylabel('P(Responder)', fontsize=9, color='#444444')
ax3.set_ylim(0, 1.15)
ax3.set_xticks([])
ax3.legend(fontsize=7.5, framealpha=0.9)
ax3.spines[['top','right']].set_visible(False)
ax3.tick_params(labelsize=8, colors='#666666')

patch_r  = plt.matplotlib.patches.Patch(color=COLOR_RESP,    label='True Responder')
patch_nr = plt.matplotlib.patches.Patch(color=COLOR_NONRESP, label='True Non-Responder')
ax3.legend(handles=[patch_r, patch_nr], fontsize=7.5,
           loc='lower right', framealpha=0.9)


# Reload true labels (y may have been overwritten in loop)
y_true = labels_df.iloc[:, 0].values

# ── PANEL 4: ROC Curves ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor('#f5f4f1')
ax4.set_title('ROC Curves — All Models\n(AUC-ROC = primary accuracy metric)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

for name, color in COLORS.items():
    probs = all_results[name]['probs']
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = all_results[name]['auc']
    ax4.plot(fpr, tpr, color=color, linewidth=2.0,
             label=f'{name}  (AUC={auc:.3f})')

# Logistic Regression baseline for reference
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict as cvp
lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_probs = cvp(lr_model, X_scaled, y_true,
               cv=StratifiedKFold(5, shuffle=True, random_state=42),
               method='predict_proba')[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_probs)
auc_lr = roc_auc_score(y_true, lr_probs)
ax4.plot(fpr_lr, tpr_lr, color='#888888', linewidth=1.5,
         linestyle=':', label=f'Logistic Reg  (AUC={auc_lr:.3f})')
ax4.plot([0,1], [0,1], color='#cccccc', linewidth=1.0,
         linestyle='--', label='Random (AUC=0.500)')

ax4.set_xlabel('False Positive Rate', fontsize=9, color='#444444')
ax4.set_ylabel('True Positive Rate',  fontsize=9, color='#444444')
ax4.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
ax4.spines[['top','right']].set_visible(False)
ax4.tick_params(labelsize=8, colors='#666666')
ax4.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(-0.02, 1.02)


# ── PANEL 5: Metric Comparison Bar Chart ─────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
ax5.set_facecolor('#f5f4f1')
ax5.set_title('Model Comparison — Accuracy, F1 Score, AUC-ROC\n'
              '(all evaluated via Stratified 5-Fold Cross-Validation)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

metrics     = ['Accuracy', 'F1 Score', 'AUC-ROC']
metric_keys = ['accuracy', 'f1',       'auc']
model_names = list(all_results.keys())
n_models    = len(model_names)
n_metrics   = len(metrics)
x           = np.arange(n_metrics)
bar_width   = 0.18

for i, (name, color) in enumerate(COLORS.items()):
    vals = [all_results[name][k] for k in metric_keys]
    bars = ax5.bar(x + (i - 1) * bar_width, vals,
                   bar_width, color=color, alpha=0.82,
                   edgecolor='white', linewidth=0.8, label=name)
    for bar, val in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=8.5, color='#333333', fontweight='500')

# Add logistic regression as reference bars
lr_acc  = accuracy_score(y_true, (lr_probs > 0.5).astype(int))
lr_f1   = f1_score(y_true, (lr_probs > 0.5).astype(int), average='weighted')
lr_vals = [lr_acc, lr_f1, auc_lr]
bars = ax5.bar(x + 1.5 * bar_width, lr_vals, bar_width,
               color='#888888', alpha=0.6, edgecolor='white',
               linewidth=0.8, label='Logistic Reg (baseline)',
               linestyle='--')
for bar, val in zip(bars, lr_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom',
             fontsize=8.5, color='#555555')

ax5.set_xticks(x + 0.25 * bar_width)
ax5.set_xticklabels(metrics, fontsize=11, color='#333333')
ax5.set_ylim(0, 1.15)
ax5.set_ylabel('Score', fontsize=10, color='#444444')
ax5.axhline(0.5, color='#cccccc', linewidth=0.8, linestyle=':')
ax5.legend(fontsize=9, framealpha=0.9, loc='upper left')
ax5.spines[['top','right']].set_visible(False)
ax5.tick_params(axis='y', labelsize=8, colors='#666666')
ax5.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)

# Highlight best model
best_auc = max(all_results[k]['auc'] for k in all_results)
ax5.text(0.99, 0.97,
         f'Best AUC-ROC: {best_auc:.3f} ({best} network)',
         transform=ax5.transAxes, ha='right', va='top',
         fontsize=9, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#aaaaaa', alpha=0.9))


# Main title
fig.suptitle(
    'Deep Learning — Neural Network Classifier\n'
    'Omalizumab Responder vs Non-Responder | 5-Gene Signature | 39 Patients',
    fontsize=14, fontweight='600', color='#111111', y=1.01
)

out_path = '/mnt/user-data/outputs/neural_network_model.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")


# =============================================================================
# SECTION 7 — PLAIN ENGLISH SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("  SECTION 5 — PLAIN ENGLISH SUMMARY")
print("=" * 65)
print(f"""
HOW TRAINING WORKS:
  Each epoch (training round):
  1. Feed 5 gene values of a patient into the network
  2. Get predicted P(Responder)
  3. Compare to real label → compute Binary Cross-Entropy loss
  4. Backpropagate: calculate how much each weight caused the error
  5. Adjust weights slightly to reduce the loss
  6. Repeat 300 times until loss converges

HOW TESTING WORKS (Stratified 5-Fold CV):
  Fold 1: Train on patients 2-39, predict patient 1-8   (held-out)
  Fold 2: Train on other 31,    predict next 8          (held-out)
  ...etc. Each patient predicted once on data never seen.
  Final accuracy = across all 39 held-out predictions.

WHY DIFFERENT LAYERS MATTER:
  2-Layer: Learns simple patterns. Low overfitting risk.
           Best for small datasets with strong signal.
  5-Layer: Learns gene interaction patterns. Medium risk.
           Good middle ground.
  7-Layer: Learns very complex patterns. High overfitting risk.
           Needs more data to reach full potential.

RESULTS SUMMARY:""")

for name, res in all_results.items():
    print(f"  {name:<10} → Accuracy={res['accuracy']:.3f}  "
          f"F1={res['f1']:.3f}  AUC-ROC={res['auc']:.3f}")

print(f"""
  Logistic Reg → Accuracy={lr_acc:.3f}  F1={lr_f1:.3f}  AUC-ROC={auc_lr:.3f}

KEY INSIGHT:
  With only 39 patients, deeper is NOT always better.
  The 2-layer network may outperform 7-layer because:
  - Fewer parameters = less overfitting
  - The 5-gene signal is already strong and simple
  - Deep networks need large datasets to show their full power
  
  This is why the paper used Logistic Regression / SVM —
  they are actually well-matched to this dataset size.
  Neural networks shine with 1,000+ samples.
""")
print("=" * 65)
print("  Complete.")
print("=" * 65)
