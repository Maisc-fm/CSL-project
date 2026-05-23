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
print("  SECTION 1 — LOADING REAL PATIENT DATA")
print("=" * 60)

features_df = pd.read_csv('C:\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\features_0.csv')
probe_ids   = [features_df.columns[0]] + list(features_df.iloc[:, 0])

data = pd.read_csv('C:\\Users\\maisa\\OneDrive\Desktop\\CSL\\Project\\ML new test\\data_0.csv', header=0)
data.columns = probe_ids

labels_df = pd.read_csv('C:\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\labels.csv', header=0)
y = labels_df.iloc[:, 0].values   # 1=Responder, 0=Non-Responder

# The 5 genes identified by the paper
FIVE_GENES = {
    'CCDC113':       'ILMN_1775520',
    'SLC26A8':       'ILMN_1893728',
    'PPP1R3D':       'ILMN_1757262',
    'LOC100131780':  'ILMN_1680347',
    'CLEC4C':        'ILMN_3306019',
}

X = data[[probe for probe in FIVE_GENES.values()]].values
gene_names = list(FIVE_GENES.keys())

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
# SECTION 2 — LINEAR REGRESSION (wrong tool — shown for comparison)
#
# WHAT IT DOES:
#   Tries to fit a straight line through the data.
#   Predicts a continuous number (e.g. 0.73 or 1.24 or -0.12).
#   Formula: y_pred = w1*CCDC113 + w2*SLC26A8 + ... + w5*CLEC4C + b
#
# HOW IT FINDS ACCURACY:
#   It minimises the Mean Squared Error (MSE) between predicted
#   numbers and actual labels (0 or 1).
#   Accuracy is measured with R² and RMSE — but these don't make
#   clinical sense for a yes/no prediction.
#
# WHY IT FAILS HERE:
#   It can predict values like 1.3 or -0.2 which are meaningless
#   as a Responder probability. There is no natural threshold.
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 2 — LINEAR REGRESSION (wrong tool)")
print("=" * 60)
print("""
  Formula:  y_pred = w1*CCDC113 + w2*SLC26A8 + w3*PPP1R3D
                   + w4*LOC100131780 + w5*CLEC4C + bias

  The model fits a hyperplane through 5D gene space.
  Output is a raw number — NOT a probability.
""")

lin_model = LinearRegression()
lin_model.fit(X_scaled, y)

y_pred_linear = lin_model.predict(X_scaled)

# Metrics for Linear Regression
r2   = r2_score(y, y_pred_linear)
rmse = np.sqrt(mean_squared_error(y, y_pred_linear))

print(f"  Learned weights (how much each gene matters):")
for gene, w in zip(gene_names, lin_model.coef_):
    bar = '█' * int(abs(w) * 15)
    sign = '+' if w > 0 else '-'
    print(f"    {gene:<16}: {sign}{abs(w):.4f}  {bar}")
print(f"    Bias (intercept) : {lin_model.intercept_:.4f}")

print(f"\n  Predictions on real patients (first 10):")
print(f"  {'Patient':<10} {'Actual':>8} {'Predicted':>12} {'Valid?':>10}")
print(f"  {'-'*45}")
for i in range(10):
    pred = y_pred_linear[i]
    actual = "Responder" if y[i]==1 else "Non-Resp."
    valid = "✅" if 0 <= pred <= 1 else "⚠️ out of range"
    print(f"  P{i+1:<9} {actual:>8} {pred:>12.4f} {valid:>10}")

print(f"\n  LINEAR REGRESSION ACCURACY METRICS:")
print(f"    R²   = {r2:.4f}  (1.0 = perfect fit, 0.0 = no fit)")
print(f"    RMSE = {rmse:.4f}  (error in same units as label: 0 or 1)")
print(f"""
  ⚠️  PROBLEM: These metrics don't answer clinical question.
      R²={r2:.2f} sounds decent, but some predictions are
      outside [0,1] range (e.g. 1.24 or -0.08).
      What does "predict 1.24" mean clinically? Nothing.
      → Linear Regression is the WRONG tool here.
""")


# =============================================================================
# SECTION 3 — LOGISTIC REGRESSION (correct tool)
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

print("=" * 60)
print("  SECTION 3 — LOGISTIC REGRESSION (correct tool)")
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

print(f"\n  Predictions on real patients (first 10):")
print(f"  {'Patient':<10} {'Actual':>10} {'P(Resp)':>9} {'Predicted':>12} {'Correct?':>10}")
print(f"  {'-'*57}")
for i in range(10):
    prob   = y_pred_log_prob[i]
    pred   = "Responder" if prob > 0.5 else "Non-Resp."
    actual = "Responder" if y[i] == 1   else "Non-Resp."
    correct = "✅" if pred == actual else "❌"
    print(f"  P{i+1:<9} {actual:>10} {prob:>9.4f} {pred:>12} {correct:>10}")

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
# SECTION 4 — SIDE-BY-SIDE COMPARISON OF BOTH MODELS
# =============================================================================

print("\n" + "=" * 60)
print("  SECTION 4 — DIRECT COMPARISON: LINEAR vs LOGISTIC")
print("=" * 60)

print(f"""
  {'Metric':<28} {'Linear Reg':>12}  {'Logistic Reg':>14}
  {'-'*58}
  {'Output type':<28} {'Raw number':>12}  {'Probability':>14}
  {'Output range':<28} {'Unbounded':>12}  {'0.0 → 1.0':>14}
  {'R² score':<28} {r2:>12.4f}  {'N/A':>14}
  {'RMSE':<28} {rmse:>12.4f}  {'N/A':>14}
  {'Accuracy':<28} {'N/A':>12}  {accuracy:>14.4f}
  {'F1 Score':<28} {'N/A':>12}  {f1:>14.4f}
  {'AUC-ROC':<28} {'N/A':>12}  {auc:>14.4f}
  {'Right tool for this problem?':<28} {'❌ No':>12}  {'✅ Yes':>14}
""")


# =============================================================================
# SECTION 5 — VISUALIZATION
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
# PANEL 1 — Linear Regression predictions (why it fails)
# ---------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#f5f4f1')
ax1.set_title('Linear Regression predictions\n(values escape [0,1] range)',
              fontsize=11, fontweight='500', color='#222222', pad=8)

sorted_idx = np.argsort(y_pred_linear)
colors_lin = [COLOR_RESP if y[i]==1 else COLOR_NONRESP for i in sorted_idx]

bars = ax1.bar(range(len(y)), y_pred_linear[sorted_idx],
               color=colors_lin, alpha=0.75, edgecolor='white',
               linewidth=0.5, width=0.8)

# Highlight out-of-range predictions
for j, i in enumerate(sorted_idx):
    pred = y_pred_linear[i]
    if pred > 1.0 or pred < 0.0:
        ax1.bar(j, pred, color=COLOR_WRONG, alpha=0.9,
                edgecolor='#800000', linewidth=1.0, width=0.8,
                label='Out of range!' if j==0 else "")

ax1.axhline(1.0, color='#333333', linewidth=1.5, linestyle='--',
            label='Valid upper limit (1.0)')
ax1.axhline(0.0, color='#333333', linewidth=1.5, linestyle=':',
            label='Valid lower limit (0.0)')
ax1.axhspan(0, 1, alpha=0.05, color='green', label='Valid zone [0,1]')

ax1.set_xlabel('Patients (sorted by prediction)', fontsize=9, color='#444444')
ax1.set_ylabel('Predicted value', fontsize=9, color='#444444')
ax1.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax1.spines[['top','right']].set_visible(False)
ax1.tick_params(labelsize=8, colors='#666666')
ax1.set_xticks([])

# Add R² annotation
ax1.text(0.97, 0.05, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
         transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=9, color='#555555',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='#cccccc', alpha=0.9))

patch_resp    = plt.matplotlib.patches.Patch(color=COLOR_RESP,    label='Responder')
patch_nonresp = plt.matplotlib.patches.Patch(color=COLOR_NONRESP, label='Non-Responder')
ax1.legend(handles=[patch_resp, patch_nonresp], fontsize=8,
           loc='lower right', framealpha=0.9)


# ---------------------------------------------------------------
# PANEL 2 — Sigmoid function (the heart of logistic regression)
# ---------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#f5f4f1')
ax2.set_title('Sigmoid function — core of Logistic Regression\n'
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
# PANEL 3 — Logistic Regression predicted probabilities per patient
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
# PANEL 4 — ROC Curve (gold standard accuracy visualization)
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
    'Linear Regression vs Logistic Regression\n'
    'Applied to Real Omalizumab Patient Data (39 patients · 5 genes)',
    fontsize=13, fontweight='600', color='#111111', y=1.01
)

out_path = 'C:\\Users\\maisa\\OneDrive\\Desktop\\CSL\\Project\\ML new test\\linear_vs_logistic.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())

print(f"\nSaved: {out_path}")


# =============================================================================
# SECTION 6 — FINAL PLAIN-ENGLISH SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"""
  QUESTION: Does the ML use Linear or Logistic Regression?
  ANSWER:   Logistic Regression (and SVM which extends it).

  WHY NOT LINEAR REGRESSION?
    Linear Regression predicts a NUMBER — e.g. 1.24 or -0.08.
    There is no meaningful interpretation of "predict 1.24"
    for a yes/no clinical question.
    R²={r2:.3f} looks OK but the predictions are unreliable
    as probabilities.

  WHY LOGISTIC REGRESSION?
    The sigmoid function forces output into [0.0, 1.0].
    This IS a probability: P=0.85 means "85% chance Responder".
    The model learns one weight per gene:
      - Positive weight → gene pushes toward Responder
      - Negative weight → gene pushes toward Non-Responder
    Then cross-entropy loss trains those weights until
    the probabilities match the real labels as closely as possible.

  HOW ACCURACY IS MEASURED:
    Accuracy = {accuracy:.3f} → {accuracy*100:.1f}% of 39 patients classified correctly
    F1       = {f1:.3f} → balances precision and recall
    AUC-ROC  = {auc:.3f} → ranks a Responder above a Non-Responder
                        {auc*100:.1f}% of the time
    AUC-ROC is the PRIMARY metric because the classes are
    imbalanced (30 Responders vs 9 Non-Responders).

  THE CLINICAL SIMULATION:
    A new patient's blood sample → measure 5 gene expressions
    → scale values → compute z = Σ(wᵢ × geneᵢ) + bias
    → P = sigmoid(z) → if P > 0.5: prescribe omalizumab
    This is the "computer simulation of a patient" in the paper.
""")
print("=" * 60)
print("  Complete.")
print("=" * 60)
