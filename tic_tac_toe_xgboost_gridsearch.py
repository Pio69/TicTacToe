
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
data = pd.read_csv('tic_tac_toe_data.csv')

# Remover a última linha incorreta (parece ser uma linha de cabeçalho)
data = data[data['Tabuleiro'] != 'Tabuleiro']

# Convertendo a coluna 'Tabuleiro' para lista de floats
data['Tabuleiro'] = data['Tabuleiro'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Preparar os dados
X = pd.DataFrame(data['Tabuleiro'].to_list())  # Features (estado do tabuleiro)
y = data['Jogada'].apply(lambda x: 1 if x == 'X' else 0)  # Labels (1 = X, 0 = O)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo os hiperparâmetros para otimização
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.001, 0.01, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.05, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

# Usar GridSearchCV para encontrar a melhor combinação de hiperparâmetros
clf = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)

# Melhor estimador encontrado
best_clf = clf.best_estimator_
print(f"Melhor combinação de parâmetros: {clf.best_params_}")

# Fazer previsões com o melhor modelo
y_pred = best_clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (XGBoost com hiperparâmetros ajustados): {accuracy * 100:.2f}%')
