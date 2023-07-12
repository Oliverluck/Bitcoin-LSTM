# Previsão do preço do Bitcoin usando LSTM

Neste projeto, eu uso uma rede neural recorrente do tipo LSTM (Long Short-Term Memory) para prever o preço futuro do Bitcoin, a partir de dados históricos obtidos da API da Binance. O modelo é treinado e testado com dados normalizados entre 0 e 1, e as previsões são desnormalizadas para obter os valores reais em dólares. Eu também calculo algumas métricas de avaliação, como o Erro Quadrático Médio (RMSE), o Erro Percentual Absoluto Médio (MAPE) e o Coeficiente de Determinação (R²), para medir o desempenho do modelo. Por fim, eu faço uma previsão futura usando os dados mais recentes disponíveis.

## Requisitos

Para executar este projeto, você precisa ter instalado as seguintes bibliotecas:

- numpy
- pandas
- sklearn
- keras
- datetime
- matplotlib
- binance

Você pode instalar todas as dependências usando o comando:

`pip install -r requirements.txt`

Além disso, você precisa ter uma conta na Binance e gerar suas próprias chaves de API. Você pode fazer isso no site https://www.binance.com/pt-BR.

## Uso

Para executar este projeto, basta rodar o script `LSTM.py` no seu terminal ou no seu ambiente de desenvolvimento preferido. O script irá baixar os dados históricos do Bitcoin da API da Binance, treinar e testar o modelo LSTM, calcular as métricas de avaliação, fazer a previsão futura e mostrar um gráfico com as previsões e os valores reais.
