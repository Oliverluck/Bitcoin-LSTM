import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
from binance.client import Client

# Definindo as chaves da API da Binance (você precisa ter uma conta na Binance e gerar suas próprias chaves de API)
api_key = 'sua_api_key'
api_secret = 'sua_api_secret'

# Criando uma instância do cliente da Binance
client = Client(api_key, api_secret, tld='us')

# Obtendo a data atual
current_date = datetime.now().strftime("%Y-%m-%d")

# Obtendo dados históricos do Bitcoin usando a API da Binance
klines = client.get_historical_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY, start_str='2010-01-01', end_str=current_date)

# Filtrando apenas a coluna de preço de fechamento
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['close']]

# Normalizando os dados entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# Dividindo os dados em treinamento e teste
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Função auxiliar para criar sequências de dados de entrada e saída
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Definindo o comprimento da sequência temporal
sequence_length = 30

# Criando sequências de treinamento e teste
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Definindo os hiperparâmetros manualmente (você pode alterá-los se quiser)
units = 150 # número de unidades nas camadas LSTM
dropout = 0.3 # taxa de dropout nas camadas Dropout

# Definindo o early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Criando o modelo final com os hiperparâmetros definidos manualmente
final_model = Sequential()
final_model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)))
final_model.add(Dropout(dropout))
final_model.add(LSTM(units=units))
final_model.add(Dropout(dropout))
final_model.add(Dense(units=1))
final_model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo final
final_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

# Fazendo previsões com o modelo final treinado
predictions = final_model.predict(X_test)

# Desnormalizando as previsões e os valores reais
predictions = scaler.inverse_transform(predictions)
y_test_denormalized = scaler.inverse_transform(y_test)

# Calculando o Erro Quadrático Médio (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_denormalized, predictions))
print("Erro Quadrático Médio (RMSE):", rmse)

# Calculando o Erro Percentual Absoluto Médio (MAPE)
mape = np.mean(np.abs((y_test_denormalized - predictions) / y_test_denormalized)) * 100
print("Erro Percentual Absoluto Médio (MAPE):", mape)

# Calculando o Coeficiente de Determinação (R²)
r2 = r2_score(y_test_denormalized, predictions)
print("Coeficiente de Determinação (R²):", r2)

# Obtendo os dados mais recentes para fazer uma previsão futura
last_data = scaled_data[-sequence_length:]
last_data = np.expand_dims(last_data, axis=0)

# Fazendo a previsão futura com o modelo final treinado
future_prediction = final_model.predict(last_data)
future_prediction = scaler.inverse_transform(future_prediction)
print('Próxima previsão:', future_prediction.flatten()[0])

# Plotando as curvas de previsão e valores reais
plt.plot(y_test_denormalized, label='Valores Reais')
plt.plot(predictions, label='Previsões')

# Exibindo a legenda
plt.legend()

# Exibindo o gráfico
plt.show()

# Imprimindo os 10 últimos dias com o resultado da previsão / real e a % de diferença entre um e outro
results = pd.DataFrame({'Real': y_test_denormalized.flatten()[-10:], 'Previsão': predictions.flatten()[-10:]})
results['Diferença (%)'] = np.abs((results['Real'] - results['Previsão']) / results['Real']) * 100
print(results)
