
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Usar GridSearchCV para encontrar a melhor combinação de hiperparâmetros
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
clf.fit(X_train, y_train)

# Melhor estimador encontrado
best_clf = clf.best_estimator_
print(f"Melhor combinação de parâmetros: {clf.best_params_}")

# Fazer previsões com o melhor modelo
y_pred = best_clf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
