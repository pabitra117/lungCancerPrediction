"""
Lung Cancer Prediction - Model Training Script (v2)
=====================================================
Uses ALL 15 original features from the dataset for training.
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import ADASYN
import joblib
import warnings

warnings.filterwarnings("ignore")

# -- Configuration --
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey lung cancer.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# -- Step 1: Load Dataset --
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# -- Step 2: Remove Duplicates --
df = df.drop_duplicates()
print(f"  After removing duplicates: {df.shape[0]} rows")

# -- Step 3: Label Encode All Columns --
le = preprocessing.LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
df['SMOKING'] = le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS'] = le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY'] = le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE'] = le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE'] = le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE '] = le.fit_transform(df['FATIGUE '])
df['ALLERGY '] = le.fit_transform(df['ALLERGY '])
df['WHEEZING'] = le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING'] = le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING'] = le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH'] = le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY'] = le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN'] = le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
print("  Label encoding complete")

# -- Step 4: Use ALL 15 features (no drops) --
# Original 15 features: GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY,
# PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING,
# ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
print(f"  Using ALL {X.shape[1]} features: {list(X.columns)}")

# -- Step 5: ADASYN Oversampling --
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)
print(f"  After ADASYN: {len(X)} samples")

# -- Step 6: Train/Test Split --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# -- Step 7: Create Output Directory --
os.makedirs(MODELS_DIR, exist_ok=True)

# -- Step 8: Train and Save All Models --
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

models = {
    "logistic_regression": LogisticRegression(random_state=0),
    "decision_tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "knn": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    "gaussian_nb": GaussianNB(),
    "multinomial_nb": MultinomialNB(),
    "svc": SVC(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(),
    "mlp": MLPClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
}

print("\n" + "=" * 60)
print("TRAINING AND SAVING MODELS (ALL 15 FEATURES)")
print("=" * 60)

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")

    # Save
    filepath = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, filepath)
    print(f"  Saved -> {filepath}")

# Also save the feature names for the frontend
feature_names = list(X.columns)
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

print("\n" + "=" * 60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print(f"Models saved in: {MODELS_DIR}")
print(f"Features used: {feature_names}")
print("=" * 60)
