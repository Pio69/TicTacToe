import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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

# Criar e treinar o modelo XGBoost
clf = XGBClassifier(random_state=42, n_estimators=50, max_depth=5, learning_rate=0.1)
clf.fit(X_train, y_train)

# Avaliar o modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Salvar o modelo treinado como um arquivo pickle
model_filename = 'tic_tac_toe_xgboost_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(clf, model_file)

print(f'Modelo salvo como {model_filename}')
