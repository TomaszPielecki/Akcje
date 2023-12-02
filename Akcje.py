import datetime as dt

import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
from numpy import array, reshape
from pandas import concat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras import metrics


def main():
    print("Algorytm Predykcji: Ceny akcji na giełdzie")
    print(f"Twoja predykcja dla akcji {stock_symbol},\nna dzień "
          f"{prediction_future_days} wynosi  {prediction}")


def get_user_input():
    while True:
        print(
            "Algorytm Predykcji: Podaj liczbę dni, dla których chcesz dokonać prognozy ceny akcji (lub wpisz 'exit' aby wyjść):")
        future_days = input()
        if future_days.lower() == 'exit':
            return None, None

        try:
            future_days = int(future_days)
            if future_days <= 0:
                raise ValueError("Liczba dni musi być większa od zera.")

            print("Podaj symbol akcji np. AAPL, GOOGL:")
            stock_symbol = input().upper()
            if not stock_symbol.isalpha():
                raise ValueError("Wprowadzony symbol akcji jest nieprawidłowy.")

            return future_days, stock_symbol
        except ValueError as ve:
            print(f"Błąd: {ve}. Spróbuj ponownie.")


try:
    future_days, stock_symbol = get_user_input()
    while future_days is not None:
        start_date = dt.datetime.now() - dt.timedelta(365 * 4)
        end_date = dt.datetime.now()

        data = yf.download(stock_symbol, start=start_date, end=end_date, interval='1d')
        if data.empty:
            raise ValueError(f"Brak danych dla akcji o symbolu {stock_symbol} w okresie czasu.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['High'].values.reshape(-1, 1))

        prediction_days = 60

        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data) - future_days):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x + future_days, 0])

        x_train, y_train = array(x_train), array(y_train)
        x_train = reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.MeanSquaredError()])

        model.fit(x_train, y_train, epochs=50, batch_size=32)

        test_start = dt.datetime.now() - dt.timedelta(365 * 4)
        test_end = dt.datetime.now()
        test_data = yf.download(stock_symbol, start=test_start, end=test_end)
        actual_price = test_data['Close'].values

        total_dataset = concat((data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        scaler.fit(model_inputs)
        model_inputs = scaler.transform(model_inputs)

        x_test = []

        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = array(x_test)
        x_test = reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction_prices = model.predict(x_test)
        prediction_prices = scaler.inverse_transform(prediction_prices)

        real_data = [model_inputs[len(model_inputs) + future_days - prediction_days:len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[1], real_data.shape[0], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        prediction_future_days = dt.datetime.now() + dt.timedelta(future_days)

        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, actual_price, color='black', label=f'Rzeczywiste ceny {stock_symbol}')
        plt.plot(test_data.index, prediction_prices, color='green', label=f'Przewidywane ceny {stock_symbol}')
        plt.xlabel(f'Time\n value = {prediction}, Days now {prediction_future_days}')
        plt.ylabel('Cena')
        plt.title(f'Prognoza cen akcji {stock_symbol} na {future_days} dni')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        future_days, stock_symbol = get_user_input()

    print("Koniec programu.")
    main()

except ValueError as ve:
    print(f"Wystąpił błąd: {ve}")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")
