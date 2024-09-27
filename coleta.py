import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Carregar os dados
data = pd.read_csv('tic_tac_toe_data.csv')

# Remover a última linha incorreta (caso seja uma linha de cabeçalho)
data = data[data['Tabuleiro'] != 'Tabuleiro']

# Convertendo a coluna 'Tabuleiro' para lista de floats
data['Tabuleiro'] = data['Tabuleiro'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Preparar os dados
X = pd.DataFrame(data['Tabuleiro'].to_list())  # Features (estado do tabuleiro)
y = data['Jogada'].apply(lambda x: 1 if x == 'X' else 0)  # Labels (1 = X, 0 = O)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o grid de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300],         # Testar diferentes números de árvores
    'max_depth': [10, 15, 20],               # Diferentes profundidades máximas
    'min_samples_split': [2, 5],             # Testar divisões com mais amostras
    'min_samples_leaf': [1, 2],              # Número mínimo de amostras por nó folha
    'bootstrap': [True, False]               # Testar com e sem amostragem com reposição
}

# Configurar o modelo Random Forest
rf = RandomForestClassifier(random_state=42)

# Usar GridSearchCV para buscar a melhor combinação de hiperparâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Ajustar o grid search aos dados
grid_search.fit(X_train, y_train)

# Melhor conjunto de hiperparâmetros encontrados
print(f'Melhores hiperparâmetros: {grid_search.best_params_}')

# Avaliar o modelo com os melhores hiperparâmetros
y_pred_best = grid_search.best_estimator_.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Acurácia com Grid Search: {accuracy_best * 100:.2f}%')

# Salvar o melhor modelo
model_filename_best = 'tic_tac_toe_random_forest_grid_search.pkl'
with open(model_filename_best, 'wb') as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)

print(f'Modelo com Grid Search salvo como {model_filename_best}')
