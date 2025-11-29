# train_and_save_model.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# Leitura correta do arquivo
df = pd.read_csv('dataset.csv')

print(f"Dataset carregado: {df.shape[0]} terremotos")
print(f"Com tsunami: {df['tsunami'].sum()} | Sem tsunami: {df['tsunami'].value_counts()[0]}")

# Apenas as 5 colunas solicitadas
features = ['magnitude', 'sig', 'depth', 'latitude', 'longitude']
X = df[features]
y = df['tsunami']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelos
rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Previsões
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Resultados
print("\n" + "="*50)
print("RANDOM FOREST (MELHOR MODELO)")
print("="*50)
print(classification_report(y_test, y_pred_rf))
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)
print(classification_report(y_test, y_pred_lr))

# Gráficos
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest - Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Logistic Regression - Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')

plt.tight_layout()
plt.show()

# Salvar o melhor modelo
joblib.dump(rf, 'best_tsunami_model.pkl')
print("\nModelo Random Forest salvo como 'best_tsunami_model.pkl'")