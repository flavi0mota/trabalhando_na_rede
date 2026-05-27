import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

np.random.seed(42)
n_amtrs = 5000
n_features = 5

t = np.linspace(0, 4 * np.pi, n_amtrs)
sinal = np.sin(t) + 0.3 * np.random.randn(n_amtrs)
X = np.array([sinal[i:i + n_features] for i in range(n_amtrs - n_features)])
y = (X.mean(axis=1) > 0.2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"treino: {X_train.shape} | Teste: {X_test.shape}")
print(f"classe 0: {np.bincount(y_train)[0]} | Classe 1: {np.bincount(y_train)[1]}\n")

modelo_cv = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scrs = cross_val_score(modelo_cv, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Folds: {np.round(scrs, 4)}")
print(f"Media: {scrs.mean():.4f} +/- {scrs.std() * 2:.4f}\n")

modelo_cv.fit(X_train, y_train)
y_pred = modelo_cv.predict(X_test)

print(f"Acuracia final: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Classe 0', 'Classe 1']))

joblib.dump(modelo_cv, 'rf_model.pkl')