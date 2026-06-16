import pandas as pd
import numpy as np

#carregando e categorizando os dados
dados = pd.read_csv('mushroom.csv', sep=';')

dados['stalk-root'] = dados['stalk-root'].replace('?', np.nan)

#indo pela moda 
moda = dados['stalk-root'].mode()[0]
dados['stalk-root'] = dados['stalk-root'].fillna(moda)


#separando os atributos da classe alvo
dados_atributos = dados.drop(columns=['mushroom_type'])
dados_classe = dados['mushroom_type']

# 3. Converta as letras em formato numérico 
dados_atributos = pd.get_dummies(dados_atributos)

# print("Dimensões dos atributos:", dados_atributos.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

#separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    dados_atributos, dados_classe, test_size=0.3, random_state=42
)

#hiperparametrização
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
criterion = ['gini', 'entropy']
min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
max_features = ['sqrt', 'log2']

# criar a grade de valores
rf_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'max_features': max_features
}

rf = RandomForestClassifier(random_state=42)

print("\nIniciando hiperparametrização (processando Grid)...")
rf_hyperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=10,
    cv=3,
    verbose=2, 
    n_jobs=-1,
    random_state=42
)

rf_hyperparameters.fit(X_train, y_train)
modelo_rf = rf_hyperparameters.best_estimator_

# print(rf_hyperparameters.best_params_)

modelo_rf.fit(X_train, y_train)
previsoes_rf = modelo_rf.predict(X_test)

#avaliação do Modelo
acuracia_rf = accuracy_score(y_test, previsoes_rf)
f1 = f1_score(y_test, previsoes_rf, average='macro')

print(f"\n- Resultados da Avaliação -")
print(f"Acurácia Global: {acuracia_rf:.4f}")
print(f"F1-Score: {f1:.4f}")

# 9. Acurácia por Classe (Adaptado para e / p)
matriz = confusion_matrix(y_test, previsoes_rf)
classes_possiveis = sorted(dados_classe.unique()) # Vai retornar ['e', 'p']

print("\nAcurácia por Classe:")
acertos_por_classe = matriz.diagonal()
total_por_classe = matriz.sum(axis=1)
acuracia_por_classe = acertos_por_classe / total_por_classe

for i, classe in enumerate(classes_possiveis):
    # Traduzindo a letra para ficar bonito no print
    nome_classe = "Comestível (e)" if classe == 'e' else "Venenoso (p)"
    print(f"Classe {nome_classe}: {acuracia_por_classe[i] * 100:.2f}%")


## ;)
import pickle
with open('modelo_rf_cogumelos.pkl', 'wb') as f:
    pickle.dump(modelo_rf, f)

with open('colunas_treino.pkl', 'wb') as f:
    pickle.dump(X_train.columns, f)

print("sucesso!")
