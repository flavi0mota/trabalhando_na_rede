import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)
n_amostras = 5000
n_features = 5

t = np.linspace(0, 4*np.pi, n_amostras)
sinal = np.sin(t) + 0.3 * np.random.randn(n_amostras)   # sinal com ruído

X = np.array([sinal[i:i+n_features] for i in range(n_amostras - n_features)])

y = (X.mean(axis=1) > 0.2).astype(int)

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
print(f"Distribuição das classes - Treino: {np.bincount(y_train)}, Teste: {np.bincount(y_test)}")

# Modelo base
rf_base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_base.fit(X_train, y_train)
print(f"Acurácia base (treino): {rf_base.score(X_train, y_train):.4f}")
print(f"Acurácia base (teste):  {rf_base.score(X_test, y_test):.4f}")

# Busca de hiperparâmetros (opcional, pode ser pesada)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                    cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Acurácia otimizada (teste): {accuracy_score(y_test, grid.predict(X_test)):.4f}")

# Salva o modelo otimizado
best_model = grid.best_estimator_
joblib.dump(best_model, 'rf_model.pkl')
print("Modelo otimizado salvo como rf_model.pkl")