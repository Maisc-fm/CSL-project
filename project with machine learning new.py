#Asthma ODE Model with Machine Learning Prediction - FIXED VERSION

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')


# PART 1: ODE MODEL 

def simulate_asthma_ode(parameters, time_points=None):
    if time_points is None:
        time_points = [7, 42, 98, 182]
    
    x = np.array(parameters)
    
    v0 = [3.768626875, 3.742900853, 3.740509442, 7.933538493, 4.337866717]
    v1 = [3.46085785, 3.457884586, 3.503851115, 7.672636459, 4.80250431]
    v2 = [3.446518574, 3.417206836, 3.398431563, 7.545667958, 4.392954553]
    v3 = [3.518292143, 3.336099096, 3.451343139, 7.578043433, 4.735453756]
    v4 = [3.452836989, 3.505254216, 3.479362452, 7.686156809, 4.77950282]
    
    v_0, v_1, v_2, v_3, v_4 = v0[0], v1[0], v2[0], v3[0], v4[0]
    
    u = (v_0*x[0] + v_1*x[1] + v_2*x[2] + v_3*x[3] + v_4*x[4])
    
    diffT = 0.25
    t = 0.0
    vars_count = int(183 / diffT)
    
    results = {7: [], 42: [], 98: [], 182: []}
    
    for j in range(vars_count):
        u = (v_0*x[15] + v_1*x[16] + v_2*x[17] + v_3*x[18] + v_4*x[19])
        
        dv0 = -x[0]*u + x[5]*np.exp(-x[6]*t)
        v_0 = v_0 + dv0 * diffT
        dv1 = -x[1]*u + x[7]*np.exp(-x[8]*t)
        v_1 = v_1 + dv1 * diffT
        dv2 = -x[2]*u + x[9]*np.exp(-x[10]*t)
        v_2 = v_2 + dv2 * diffT
        dv3 = -x[3]*u + x[11]*np.exp(-x[12]*t)
        v_3 = v_3 + dv3 * diffT
        dv4 = -x[4]*u + x[13]*np.exp(-x[14]*t)
        v_4 = v_4 + dv4 * diffT
        
        t += diffT
        
        if abs(t - 7) < diffT/2:
            results[7] = [v_0, v_1, v_2, v_3, v_4]
        if abs(t - 42) < diffT/2:
            results[42] = [v_0, v_1, v_2, v_3, v_4]
        if abs(t - 98) < diffT/2:
            results[98] = [v_0, v_1, v_2, v_3, v_4]
        if abs(t - 182) < diffT/2:
            results[182] = [v_0, v_1, v_2, v_3, v_4]
    
    return results

# Your optimal parameters
OPTIMAL_PARAMS = np.array([
    0.51232006, 0.5090245, 0.51536566, 0.51423559, 0.48430226,
    0.49994949, 0.49126614, 0.49969414, 0.50124126, 0.49899188,
    0.5086718, 0.49940021, 0.50115575, 0.50132442, 0.5028132,
    0.51604927, 0.4747112, 0.5278044, 0.52298436, 0.45344783
])

print("=" * 60)
print("ASTHMA ODE MODEL - RUNNING")
print("=" * 60)

ode_predictions = simulate_asthma_ode(OPTIMAL_PARAMS)

print("\n--- ODE Predictions ---")
print(f"Day 7:  v0={ode_predictions[7][0]:.4f}")
print(f"Day 42: v0={ode_predictions[42][0]:.4f}")
print(f"Day 98: v0={ode_predictions[98][0]:.4f}")
print(f"Day 182:v0={ode_predictions[182][0]:.4f}")

# ============================================================================
# PART 2: GENERATE SYNTHETIC DATA
# ============================================================================

np.random.seed(42)
n_patients = 200

patient_features = np.zeros((n_patients, 10))
patient_features[:, 0] = np.random.normal(40, 15, n_patients)  # Age
patient_features[:, 1] = np.random.normal(26, 5, n_patients)   # BMI
patient_features[:, 2] = np.random.randint(1, 10, n_patients)  # Severity
patient_features[:, 3] = np.random.beta(2, 5, n_patients)      # Genetic 1
patient_features[:, 4] = np.random.beta(3, 3, n_patients)      # Genetic 2
patient_features[:, 5] = np.random.beta(1, 4, n_patients)      # Genetic 3
patient_features[:, 6] = np.random.choice([0, 1], n_patients)  # Smoking
patient_features[:, 7] = np.random.choice([0, 1], n_patients)  # Allergy
patient_features[:, 8] = np.random.normal(2, 0.5, n_patients)  # IgE
patient_features[:, 9] = np.random.normal(70, 15, n_patients)  # FEV1

patient_targets = np.zeros((n_patients, 5))
base_outcome = ode_predictions[182]

for i in range(n_patients):
    severity_effect = (patient_features[i, 2] - 5) / 20
    genetic_effect = patient_features[i, 3] * 0.1
    smoking_effect = patient_features[i, 6] * 0.05
    patient_targets[i, :] = base_outcome + severity_effect + genetic_effect + smoking_effect
    patient_targets[i, :] += np.random.normal(0, 0.05, 5)

# ============================================================================
# PART 3: TRAIN MODELS
# ============================================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING RESULTS")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    patient_features, patient_targets, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)

 # ✅ ADD SVM HERE (wrapped for multi-output)
    'SVM (SVR)': MultiOutputRegressor(SVR(kernel='rbf', C=10, epsilon=0.1)),

    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    if name in ['Neural Network', 'SVM (SVR)']:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
    
    mse_per_gene = []
    r2_per_gene = []
    
    for i in range(5):
        mse = mean_squared_error(y_test[:, i], predictions[:, i])
        r2 = r2_score(y_test[:, i], predictions[:, i])
        mse_per_gene.append(mse)
        r2_per_gene.append(r2)
    
    overall_mse = np.mean(mse_per_gene)
    overall_r2 = np.mean(r2_per_gene)
    
    results[name] = {'overall_mse': overall_mse, 'overall_r2': overall_r2}
    
    print(f"  MSE: {overall_mse:.4f}")
    print(f"  R² Score: {overall_r2:.4f}")

# ============================================================================
# PART 4: CROSS-VALIDATION (FIXED)
# ============================================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-fold)")
print("=" * 60)

# For multi-output, we need to use MultiOutputRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(rf_model)

cv_scores = cross_val_score(multi_model, patient_features, patient_targets, 
                            cv=5, scoring='r2')

print(f"\n5-fold CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# PART 5: VISUALIZATION (Simplified)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Model comparison
model_names = list(results.keys())
r2_scores = [results[m]['overall_r2'] for m in model_names]
axes[0].bar(model_names, r2_scores, color=['green' if r>0.5 else 'orange' for r in r2_scores])
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Comparison')
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylim(0, 1)

# Plot 2: Gene expression over time
time_points = [7, 42, 98, 182]
genes = ['v0', 'v1', 'v2', 'v3', 'v4']
for i, gene in enumerate(genes):
    values = [ode_predictions[t][i] for t in time_points]
    axes[1].plot(time_points, values, marker='o', label=gene)
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Gene Expression')
axes[1].set_title('ODE Model Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: CV scores
axes[2].bar(range(1, 6), cv_scores)
axes[2].axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('R² Score')
axes[2].set_title('Cross-Validation Results')
axes[2].legend()
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('asthma_ml_results_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'asthma_ml_results_fixed.png'")

# ============================================================================
# PART 6: NEW PATIENT PREDICTION
# ============================================================================

print("\n" + "=" * 60)
print("NEW PATIENT PREDICTION")
print("=" * 60)

new_patient = np.array([[
    35, 24.5, 7, 0.75, 0.45, 0.62, 1, 1, 2.1, 65
]])

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
prediction = best_model.predict(new_patient)

print("\nPredicted Gene Expression at Day 182:")
for i, gene in enumerate(['v0', 'v1', 'v2', 'v3', 'v4']):
    print(f"  {gene}: {prediction[0, i]:.4f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Best Model: Random Forest (R² = {results['Random Forest']['overall_r2']:.3f})")
print(f"✅ Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"✅ ODE final cost: 4.27")