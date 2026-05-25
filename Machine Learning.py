"""
=============================================================================
  LINEAR vs LOGISTIC REGRESSION
  Applied to Real Omalizumab Patient Data (39 patients, 5 genes)
  Lopez-Rincon et al. (2021/2023)
=============================================================================

SIMPLE EXPLANATION:
  Linear Regression  → predicts a NUMBER   → wrong tool for our problem
  Logistic Regression → predicts a GROUP   → correct tool for our problem

  Our question is: "Will this patient RESPOND to omalizumab (yes/no)?"
  That is a classification problem → Logistic Regression

  This script shows:
  1. How Linear Regression works (and why it fails here)
  2. How Logistic Regression works (and why it succeeds here)
  3. Both applied to your real 39-patient gene expression data
  4. How each measures its "accuracy"
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    r2_score, mean_squared_error,        # Linear regression metrics
    accuracy_score, f1_score,            # Logistic regression metrics
    roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# =============================================================================
# SECTION 1 — LOAD YOUR REAL PATIENT DATA
#
# 39 patients, each described by 5 gene expression values.
# Labels: 1 = Responder to omalizumab, 0 = Non-Responder
# =============================================================================

print("=" * 60)
print("  LOADING REAL PATIENT DATA")
print("=" * 60)

# replace the file path to your own .csv file destination!!
data = pd.read_csv('C:\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML final\\data\\monte_carlo_800_patients.csv', header=0)
y = data['label'].values

FIVE_GENES = ['CCDC113', 'SLC26A8', 'PPP1R3D', 'LOC100131780', 'CLEC4C']
X = data[FIVE_GENES].values
gene_names = FIVE_GENES

# Scale: bring all gene expression values to same range (mean=0, std=1)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Patients      : {X.shape[0]}")
print(f"  Genes (features): {X.shape[1]} → {gene_names}")
print(f"  Responders (1): {(y==1).sum()}")
print(f"  Non-Resp.  (0): {(y==0).sum()}")
print(f"\n  First 3 patients' scaled gene values:")
print(f"  {'Patient':<10}", end="")
for g in gene_names: print(f"  {g:<14}", end="")
print()
print(f"  {'-'*85}")
for i in range(3):
    label = "Responder" if y[i]==1 else "Non-Resp."
    print(f"  P{i+1:<9}", end="")
    for val in X_scaled[i]: print(f"  {val:<14.4f}", end="")
    print(f"  → {label}")


# =============================================================================
# SECTION 2 — LOGISTIC REGRESSION
#
# WHAT IT DOES:
#   Wraps the linear formula inside a SIGMOID function.
#   Output is ALWAYS between 0.0 and 1.0 — a true probability.
#
#   Step 1: linear score z = w1*CCDC113 + w2*SLC26A8 + ... + b
#   Step 2: probability  P = 1 / (1 + e^(-z))    ← sigmoid function
#
#   If P > 0.5 → predict Responder (1)
#   If P < 0.5 → predict Non-Responder (0)
#
# HOW IT FINDS ACCURACY:
#   It maximises log-likelihood (how probable the real labels are
#   given the predictions) — this is equivalent to minimising
#   cross-entropy loss.
#   Accuracy is measured with: Accuracy, F1, AUC-ROC.
#
# WHY IT WORKS:
#   Every prediction is a valid probability [0,1].
#   A threshold (0.5) cleanly maps to Responder / Non-Responder.
# =============================================================================

print("\n")
print("=" * 60)
print("  LOGISTIC REGRESSION")
print("=" * 60)
print("""
  Formula: z = w1*CCDC113 + ... + w5*CLEC4C + bias
           P(Responder) = 1 / (1 + e^(-z))    ← sigmoid

  The sigmoid "squishes" any number into range [0.0, 1.0].
  P > 0.5 → classify as Responder
  P < 0.5 → classify as Non-Responder
""")

log_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

# Cross-validated predictions (each patient predicted on hold-out fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_log      = cross_val_predict(log_model, X_scaled, y, cv=cv)
y_pred_log_prob = cross_val_predict(log_model, X_scaled, y,
                                     cv=cv, method='predict_proba')[:, 1]

# Also fit fully for weights
log_model.fit(X_scaled, y)

# Metrics for Logistic Regression
accuracy = accuracy_score(y, y_pred_log)
f1       = f1_score(y, y_pred_log, average='weighted')
auc      = roc_auc_score(y, y_pred_log_prob)
cm       = confusion_matrix(y, y_pred_log)

print(f"  Learned weights (how much each gene pushes toward Responder):")
for gene, w in zip(gene_names, log_model.coef_[0]):
    direction = "→ Responder" if w > 0 else "→ Non-Resp."
    bar = '█' * int(abs(w) * 8)
    print(f"    {gene:<16}: {w:+.4f}  {bar}  {direction}")
print(f"    Bias (intercept) : {log_model.intercept_[0]:.4f}")

print(f"\n  LOGISTIC REGRESSION ACCURACY METRICS:")
print(f"    Accuracy = {accuracy:.4f}  ({accuracy*100:.1f}% of patients correctly classified)")
print(f"    F1 Score = {f1:.4f}  (balance of precision & recall)")
print(f"    AUC-ROC  = {auc:.4f}  (ability to rank Responders above Non-Resp.)")
print(f"\n  Confusion Matrix (5-fold cross-validation):")
print(f"                     Predicted")
print(f"                     Non-Resp   Responder")
print(f"    Actual Non-Resp     {cm[0,0]:>3}         {cm[0,1]:>3}   ← {cm[0,0]} correctly caught")
print(f"    Actual Responder    {cm[1,0]:>3}         {cm[1,1]:>3}   ← {cm[1,1]} correctly caught")
print(f"\n  Classification Report:")
for line in classification_report(
        y, y_pred_log,
        target_names=['Non-Responder','Responder']).split('\n'):
    print(f"    {line}")


# =============================================================================
# SECTION 3 — SIDE-BY-SIDE COMPARISON OF BOTH MODELS
# =============================================================================

print("\n" + "=" * 60)
print("  LOGISTIC REGRESSION PERFORMANCE METRIC")
print("=" * 60)

print(f"""
  {'Metric':<28} {'Logistic Reg':>14}
  {'-'*60}
  {'Output type':<28} {'Probability':>14}
  {'Output range':<28} {'0.0 → 1.0':>14}
  {'Accuracy':<28} {accuracy:>14.4f}
  {'F1 Score':<28} {f1:>14.4f}
  {'AUC-ROC':<28} {auc:>14.4f}
""")


# =============================================================================
# SECTION 4 — VISUALIZATION
#
# 4-panel figure:
#   Panel 1: Linear regression predictions — shows values going out of [0,1]
#   Panel 2: Logistic sigmoid — shows how probabilities are "squished"
#   Panel 3: Logistic predictions per patient — probability bar chart
#   Panel 4: ROC curve — the gold-standard accuracy visualization
# =============================================================================

print("Generating visualization...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#fafaf9')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

COLOR_RESP    = '#1D9E75'
COLOR_NONRESP = '#D85A30'
COLOR_WRONG   = '#cc0000'


# ---------------------------------------------------------------
# PANEL 1 — Sigmoid function (the heart of logistic regression)
# ---------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#f5f4f1')
ax2.set_title('Sigmoid function\n'
              'Converts raw score z → probability P(Responder)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

z_range = np.linspace(-6, 6, 300)
sigmoid = 1 / (1 + np.exp(-z_range))

ax2.plot(z_range, sigmoid, color='#2255aa', linewidth=2.5,
         label='P = 1/(1+e^(-z))')
ax2.fill_between(z_range, sigmoid, 0.5,
                 where=(sigmoid >= 0.5), alpha=0.12,
                 color=COLOR_RESP, label='→ Responder (P>0.5)')
ax2.fill_between(z_range, sigmoid, 0.5,
                 where=(sigmoid < 0.5), alpha=0.12,
                 color=COLOR_NONRESP, label='→ Non-Responder (P<0.5)')

ax2.axhline(0.5, color='#333333', linewidth=1.2, linestyle='--',
            label='Decision threshold (0.5)')
ax2.axhline(1.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
ax2.axhline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
ax2.axvline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')

# Annotate real patient examples
z_vals_patients = log_model.decision_function(X_scaled)
for i in [0, 1, 4, 12]:
    z = z_vals_patients[i]
    p = 1 / (1 + np.exp(-z))
    color = COLOR_RESP if y[i]==1 else COLOR_NONRESP
    ax2.scatter(z, p, s=80, color=color, zorder=5,
                edgecolors='white', linewidths=1.0)
    label_txt = f'P{i+1}: P={p:.2f}'
    ax2.annotate(label_txt, (z, p),
                 xytext=(8, 5), textcoords='offset points',
                 fontsize=7.5, color=color)

ax2.set_xlabel('z  (linear score from gene weights)', fontsize=9, color='#444444')
ax2.set_ylabel('P(Responder)', fontsize=9, color='#444444')
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax2.spines[['top','right']].set_visible(False)
ax2.tick_params(labelsize=8, colors='#666666')
ax2.text(3.5, 0.72, '← Responder\n   zone',
         color=COLOR_RESP, fontsize=8.5, fontweight='500', alpha=0.8)
ax2.text(-5.8, 0.22, 'Non-Responder\nzone →',
         color=COLOR_NONRESP, fontsize=8.5, fontweight='500', alpha=0.8)


# ---------------------------------------------------------------
# PANEL 2 — Logistic Regression predicted probabilities per patient
# ---------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#f5f4f1')
ax3.set_title('Logistic Regression — P(Responder) per patient\n'
              '(cross-validated probabilities)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

sorted_by_prob = np.argsort(y_pred_log_prob)
bar_colors = []
bar_edge   = []
for i in sorted_by_prob:
    correct = (y_pred_log_prob[i] > 0.5) == (y[i] == 1)
    if y[i] == 1:
        bar_colors.append(COLOR_RESP)
        bar_edge.append(COLOR_RESP if correct else COLOR_WRONG)
    else:
        bar_colors.append(COLOR_NONRESP)
        bar_edge.append(COLOR_NONRESP if correct else COLOR_WRONG)

bplot = ax3.bar(range(len(y)), y_pred_log_prob[sorted_by_prob],
                color=bar_colors, edgecolor=bar_edge,
                linewidth=1.5, alpha=0.82, width=0.8)

ax3.axhline(0.5, color='#333333', linewidth=1.5, linestyle='--',
            label='Decision threshold (P=0.5)')
ax3.axhspan(0.5, 1.0, alpha=0.06, color=COLOR_RESP)
ax3.axhspan(0.0, 0.5, alpha=0.06, color=COLOR_NONRESP)

ax3.text(1, 0.92, '→ Predicted Responder',
         color=COLOR_RESP, fontsize=8, alpha=0.8)
ax3.text(1, 0.04, '→ Predicted Non-Responder',
         color=COLOR_NONRESP, fontsize=8, alpha=0.8)

# Mark misclassified patients
for j, i in enumerate(sorted_by_prob):
    correct = (y_pred_log_prob[i] > 0.5) == (y[i] == 1)
    if not correct:
        ax3.text(j, y_pred_log_prob[i] + 0.03, '✗',
                 ha='center', fontsize=10, color=COLOR_WRONG, fontweight='bold')

patch_resp    = plt.matplotlib.patches.Patch(color=COLOR_RESP,    label='True Responder')
patch_nonresp = plt.matplotlib.patches.Patch(color=COLOR_NONRESP, label='True Non-Responder')
ax3.legend(handles=[patch_resp, patch_nonresp], fontsize=8,
           loc='upper left', framealpha=0.9)

ax3.set_xlabel('Patients (sorted by P(Responder))', fontsize=9, color='#444444')
ax3.set_ylabel('P(Responder)', fontsize=9, color='#444444')
ax3.set_ylim(0, 1.1)
ax3.set_xticks([])
ax3.spines[['top','right']].set_visible(False)
ax3.tick_params(labelsize=8, colors='#666666')

ax3.text(0.97, 0.55, f'Accuracy = {accuracy:.3f}\nF1 = {f1:.3f}\nAUC = {auc:.3f}',
         transform=ax3.transAxes, ha='right', va='bottom',
         fontsize=9, color='#555555',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#cccccc', alpha=0.9))


# ---------------------------------------------------------------
# PANEL 3 — ROC Curve (gold standard accuracy visualization)
#
# AUC-ROC is the primary metric for imbalanced binary classification.
# The curve shows the trade-off between:
#   True Positive Rate  = of all Responders, how many did we catch?
#   False Positive Rate = of all Non-Resp., how many did we falsely flag?
# A perfect model = curve hugs the top-left corner (AUC = 1.0)
# A random model  = diagonal line (AUC = 0.5)
# ---------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#f5f4f1')
ax4.set_title('ROC Curve — primary accuracy metric for this problem\n'
              'AUC = area under curve (1.0=perfect, 0.5=random)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

fpr, tpr, thresholds = roc_curve(y, y_pred_log_prob)

ax4.fill_between(fpr, tpr, alpha=0.15, color='#2255aa')
ax4.plot(fpr, tpr, color='#2255aa', linewidth=2.5,
         label=f'Logistic Regression (AUC = {auc:.3f})')
ax4.plot([0,1], [0,1], color='#aaaaaa', linewidth=1.2,
         linestyle='--', label='Random classifier (AUC = 0.500)')
ax4.plot([0,0,1], [0,1,1], color='#1D9E75', linewidth=1.2,
         linestyle=':', alpha=0.6, label='Perfect classifier (AUC = 1.000)')

# Annotate the operating point (threshold=0.5)
idx_thresh = np.argmin(np.abs(thresholds - 0.5))
ax4.scatter(fpr[idx_thresh], tpr[idx_thresh],
            s=100, color='#cc3300', zorder=5,
            label=f'Threshold=0.5: TPR={tpr[idx_thresh]:.2f}, FPR={fpr[idx_thresh]:.2f}')

ax4.set_xlabel('False Positive Rate\n(Non-Responders wrongly predicted as Responder)',
               fontsize=9, color='#444444')
ax4.set_ylabel('True Positive Rate\n(Responders correctly predicted)',
               fontsize=9, color='#444444')
ax4.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax4.spines[['top','right']].set_visible(False)
ax4.tick_params(labelsize=8, colors='#666666')
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(-0.02, 1.02)
ax4.grid(True, linestyle='--', linewidth=0.3, color='#dddddd', alpha=0.7)

ax4.text(0.35, 0.18,
         f'AUC = {auc:.3f}\n→ Model correctly ranks a\nResponder above a Non-Responder\n{auc*100:.1f}% of the time',
         fontsize=8.5, color='#2255aa',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='#aaaacc', alpha=0.92))


# ---------------------------------------------------------------
# Main title
# ---------------------------------------------------------------
fig.suptitle(
    'Logistic Regression\n'
    'Applied to Real Omalizumab Patient Data (39 patients · 5 genes)',
    fontsize=13, fontweight='600', color='#111111', y=1.01
)

out_path = 'C:\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\linear_vs_logistic.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())

print(f"\nSaved.")



