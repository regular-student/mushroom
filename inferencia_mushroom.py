import pandas as pd
import numpy as np
import pickle

#carregando o modelo
with open('modelo_rf_cogumelos.pkl', 'rb') as f:
    modelo_carregado = pickle.load(f)

with open('colunas_treino.pkl', 'rb') as f:
    colunas_treino = pickle.load(f)

print("Modelo carregado")

# dados de um cogumelo desconhecido
dados_desconhecidos = {
    'cap-shape': ['x'], 'cap-surface': ['s'], 'cap-color': ['n'],
    'bruises': ['t'], 'odor': ['p'], 'gill-attachment': ['f'],
    'gill-spacing': ['c'], 'gill-size': ['n'], 'gill-color': ['k'],
    'stalk-shape': ['e'], 'stalk-root': ['?'], # dado em falta
    'stalk-surface-above-ring': ['s'], 'stalk-surface-below-ring': ['s'],
    'stalk-color-above-ring': ['w'], 'stalk-color-below-ring': ['w'],
    'veil-type': ['p'], 'veil-color': ['w'], 'ring-number': ['o'],
    'ring-type': ['p'], 'spore-print-color': ['k'], 'population': ['s'], 
    'habitat': ['u']
}

#DataFrame
df_novo = pd.DataFrame(dados_desconhecidos)

# mesmo pré-processamento do treino
# Substituir '?' por nulo e preencher com 'b'
df_novo['stalk-root'] = df_novo['stalk-root'].replace('?', np.nan)
df_novo['stalk-root'] = df_novo['stalk-root'].fillna('b')

# Fazer o One-Hot Encoding
df_novo_dummies = pd.get_dummies(df_novo)

# As características que este cogumelo não tem serão preenchidas com 0.
df_novo_alinhado = df_novo_dummies.reindex(columns=colunas_treino, fill_value=0)

#inferencia
previsao = modelo_carregado.predict(df_novo_alinhado)

if previsao[0] == 'e':
    print("Classificação: O cogumelo é COMESTÍVEL (edible). Pode consumir.")
else:
    print("Classificação: O cogumelo é VENENOSO (poisonous). PERIGO!")
