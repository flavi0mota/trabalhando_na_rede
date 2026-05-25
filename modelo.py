import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
X = np.random.rand(2000, 5) * 100
y = (X[:, 0] + X[:, 1] > 100).astype(int)

# Mais árvores e profundidade limitada para evitar overfit do ruído
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

joblib.dump(model, 'rf_model.pkl')
print("Modelo salvo como rf_model.pkl")
print("Acurácia no treino:", model.score(X, y))
