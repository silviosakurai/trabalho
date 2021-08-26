# -- coding: utf-8 --

import os
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go

Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_dir = os.path.join(Base_dir, 'data')

str_connection = 'sqlite:///{path}'

my_files = []

files_names = [i for i in os.listdir(Data_dir) if i.endswith('.csv')]

str_connection = str_connection.format(path=os.path.join(Data_dir, 'olist.db'))

for i in files_names:
    df_tmp = pd.read_csv(os.path.join(Data_dir, i))
    print(df_tmp.head())
    db_name = 'fr_' + i.strip('.csv').replace('olist_', '').replace('_dataset','')
    df_tmp.to_sql(db_name, str_connection)


previsores = pd.read_csv(os.path.join(Data_dir, 'sales (2).csv'))


from sklearn.model_selection import train_test_split

X = previsores.drop(columns=['DATA', 'VENDAS'])
y = previsores.drop(columns=['DATA', 'FDS', 'DS', 'DATA_FESTIVA', 'VESPERA_DATA_FESTIVA', 'POS_DATA_FESTIVA', 'DATA_NAO_FESTIVA', 'FERIADO', 'NAO_FERIADO', 'SEMANA_PAGAMENTO', 'SEMANA_DE_NAO_PAGAMENTO', 'BAIXA_TEMPORADA', 'ALTA_TEMPORADA', 'QTD_CONCORRENTES', 'PRECIPITACAO', 'TEMPERATURA', 'UMIDADE', 'VENDAS_MEDIA_TRIM','VENDAS_MEDIA_MES','VENDAS_STD_TRIM','VENDAS_STD_MES'])
prev_train, prev_test, class_train, class_test = train_test_split(X, y, test_size=0.2)

#vendo a correlação entre as variáveis
a = previsores.corr()

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout

classificador = Sequential()

#os melhores testes foram com uma camada oculta apenas
classificador.add(Dense(units = 64, activation = 'tanh', kernel_initializer='normal', input_dim=20))

classificador.add(Dense(128,activation='relu'))

classificador.add(Dense(units=1,activation='relu'))

otimizador = keras.optimizers.RMSprop()

classificador.compile(otimizador, loss='mse', metrics=['mape'])

history = classificador.fit(prev_train, class_train, batch_size=50, epochs=1500)


#plot do grafico da curva de aprendizado no treino
fig = go.Figure()

fig.add_trace(go.Scattergl(y=history.history['mape'],
                    name='Valid'))

fig.update_layout(height=500, width=700,
                  xaxis_title='Epoch',
                  yaxis_title='MAE')

plot(fig, auto_open=True)

#calculando as previsões
previsoes = classificador.predict(prev_test)

from sklearn.metrics import mean_absolute_error

#utilizando o MAE para ver a precisão
precisao = mean_absolute_error(class_test, previsoes)

resultado = classificador.evaluate(prev_test, class_test)

#plotando o grafico mostrando os valores previstos e reais
fig = go.Figure()

fig.add_trace(go.Scattergl(y=previsoes[:,0],
                    name='PREVISTO'))
fig.add_trace(go.Scattergl(y=class_test['VENDAS'],
                    name='REAL'))
fig.update_layout(height=500, width=700,
                  xaxis_title='',
                  yaxis_title='', colorway=["green", "orange"])

plot(fig, auto_open=True)