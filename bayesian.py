import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("heart.csv") 

drop_cols = [col for col in ['id', 'dataset'] if col in df.columns]
df.drop(columns=drop_cols, inplace=True)

label_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal']
for col in label_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

if 'fbs' in df.columns:
    df.fillna({'fbs': 0}, inplace=True)
    df['fbs'] = df['fbs'].astype(int) 

scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1].astype(float))

# === Fuzzy Logic System ===
chol = ctrl.Antecedent(np.arange(-2, 3, 0.1), 'chol')
bp = ctrl.Antecedent(np.arange(-2, 3, 0.1), 'bp')
age = ctrl.Antecedent(np.arange(-2, 3, 0.1), 'age')
risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')

# Generate membership functions automatically
for var in [chol, bp, age]:
    var.automf(3)  # Generates labels: 'poor', 'average', 'good'

# Define risk levels manually
risk['low'] = fuzz.trimf(risk.universe, [0, 0.2, 0.4])
risk['medium'] = fuzz.trimf(risk.universe, [0.3, 0.5, 0.7])
risk['high'] = fuzz.trimf(risk.universe, [0.6, 0.8, 1.0])

# Define fuzzy rules
rules = [
    ctrl.Rule(chol['good'] | bp['good'] | age['good'], risk['high']),
    ctrl.Rule(chol['average'] | bp['average'] | age['average'], risk['medium']),
    ctrl.Rule(chol['poor'] & bp['poor'] & age['poor'], risk['low'])
]

# Control system
risk_ctrl = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# Test with a sample patient
sample = df.iloc[0]  # First patient
risk_ctrl.input['chol'] = sample['chol']
risk_ctrl.input['bp'] = sample['trestbps']
risk_ctrl.input['age'] = sample['age']
risk_ctrl.compute()

# Output result
fuzzy_risk = risk_ctrl.output['risk']
print(f"\nFuzzy Logic Predicted Heart Disease Risk: {fuzzy_risk:.2f}")

plt.bar(['Fuzzy Logic Prediction'], [fuzzy_risk], color=['blue'])
plt.ylabel('Risk Score')
plt.title('Heart Disease Prediction using Fuzzy Logic')
plt.ylim(0, 1)
plt.show()

