# -*- coding: utf-8 -*-
#!venv/bin/python
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram import types
import requests
from bs4 import BeautifulSoup
import investpy
import numpy as np
import socket
from datetime import datetime
from datetime import date
import datetime as dt
import yfinance as yf
import yahoofinancials
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas_datareader import data as pdr
import re
from aiogram.utils.markdown import *
import time
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

plt.style.use('fivethirtyeight')
# Объект бота
bot = Bot(token="2125750419:AAEfypdZsvCNWE2_a7NSv7Cz2hkru7e5Vxk")
# Диспетчер для бота
dp = Dispatcher(bot)
# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

# Хэндлер на команду /test1
@dp.message_handler(commands="start")
async def cmd_test1(message: types.Message):
    UserAgent().chrome
    page_link = 'https://ru.investing.com/currencies/usd-rub'
    response = requests.get(page_link, headers={'User-Agent': UserAgent().chrome})
    response
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')
    pars = soup.find('span', class_='arial_26 inlineblock pid-2186-last').get_text()

    UserAgent().chrome
    page_link = 'https://ru.investing.com/currencies/eur-rub'
    response1 = requests.get(page_link, headers={'User-Agent': UserAgent().chrome})
    response1
    html1 = response1.content
    soup1 = BeautifulSoup(html1, 'html.parser')
    pars1 = soup1.find('span', class_='arial_26 inlineblock pid-1691-last').get_text()

    keyboard = types.ReplyKeyboardMarkup()
    button_1 = types.KeyboardButton(text="Анализ акции")
    keyboard.add(button_1)
    button_2 = "Инструкция"
    keyboard.add(button_2)
    button_3 = "Риск акции"
    keyboard.add(button_3)
    button_4 = "Прогноз акции"
    keyboard.add(button_4)
    button_5 = "Потенциальная акция"
    keyboard.add(button_5)
    button_6 = "Оптимизация портфеля"
    keyboard.add(button_6)
    button_7 = "Анализ портфеля"
    keyboard.add(button_7)
    await message.answer(f"---🎉AFI Telegram Bot🎉---\nАналитика и прогнозирование фондового рынка.\nКурс USD/RUB EUR/RUB: \n{pars} \n{pars1} \nВведите ТИКЕР (можно найти на investing.com) ", reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "Инструкция")
async def without_puree(message: types.Message):
    with open('instr.jpg', 'rb') as photo:
        await message.reply_photo(photo, caption='---Инструкция--- \n1.Зайдите на сайт investing.com, выберите интересующую вас акцию(акция должна быть биржи NASDAQ) и скопируйте тикер. \n2.Введите "/start", укажите тикер, выберите кнопку и программа начнёт свою работу.')
@dp.message_handler(lambda message: message.text.isupper())
async def get_name(message: types.Message):
    global action
    action = message.text
    if (action.isdigit() == False) and (action.isupper() == True):
        await message.answer('Тикер установлен')
    else:
        await message.reply('Некоректный тикер! Повторите попытку!')
@dp.message_handler(lambda message: message.text =="Анализ акции")
async def get_name(message: types.Message):
        try:
            def Stock_SMA(stock, country):
                ''' stock - stock exchange abbreviation; country - the name of the country'''
                # Read data
                current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(
                    date.today().year)
                try:
                    df = investpy.get_stock_historical_data(stock=stock, country=country,
                                                            from_date='01/01/2019',
                                                            to_date=current_date)
                except:
                    df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
                # Count SMA30 / SMA90
                SMA30 = pd.DataFrame()
                SMA30['Close Price'] = df['Close'].rolling(window=30).mean()
                SMA90 = pd.DataFrame()
                SMA90['Close Price'] = df['Close'].rolling(window=90).mean()
                data = pd.DataFrame()
                data['Stock'] = df['Close']
                data['SMA30'] = SMA30['Close Price']
                data['SMA90'] = SMA90['Close Price']

                # Визуализируем
                plt.figure(figsize=(12.6, 4.6))
                plt.plot(data['Stock'], label=stock, alpha=0.35)
                plt.plot(SMA30['Close Price'], label='SMA30', alpha=0.35)
                plt.plot(SMA90['Close Price'], label='SMA90', alpha=0.35)
                plt.title(stock + ' История (SMA)')
                plt.xlabel('01/01/2019 - ' + current_date)
                plt.ylabel('Цена закрытия')
                plt.legend(loc='upper left')
                plt.savefig('img1.jpg')

            def Stock_EMA(stock, country):
                ''' stock - stock exchange abbreviation; country - the name of the country'''
                # Read data
                current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(
                    date.today().year)
                try:
                    df = investpy.get_stock_historical_data(stock=stock, country=country,
                                                            from_date='01/01/2019',
                                                            to_date=current_date)
                except:
                    df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
                # Count EMA20 / EMA60
                EMA20 = pd.DataFrame()
                EMA20['Close Price'] = df['Close'].ewm(span=20).mean()
                EMA60 = pd.DataFrame()
                EMA60['Close Price'] = df['Close'].ewm(span=60).mean()
                data = pd.DataFrame()
                data['Stock'] = df['Close']
                data['EMA20'] = EMA20['Close Price']
                data['EMA60'] = EMA60['Close Price']

                # Визуализируем
                plt.figure(figsize=(12.6, 4.6))
                plt.plot(data['Stock'], label=stock, alpha=0.35)
                plt.plot(EMA20['Close Price'], label='EMA30', alpha=0.35)
                plt.plot(EMA60['Close Price'], label='EMA60', alpha=0.35)
                plt.title(stock + ' История (EMA)')
                plt.xlabel('01/01/2019 - ' + current_date)
                plt.ylabel('Цена закрытия')
                plt.legend(loc='upper left')
                plt.savefig('img2.jpg')

            def Upper_levels(stock, country):
                current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(
                    date.today().year)
                try:
                    df = investpy.get_stock_historical_data(stock=stock, country=country,
                                                            from_date='01/01/2019',
                                                            to_date=current_date)
                except:
                    df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)

                pivots = []
                dates = []
                counter = 0
                lastPivot = 0

                Range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                for i in df.index:
                    currentMax = max(Range, default=0)
                    value = round(df['High'][i], 2)

                    Range = Range[1:9]
                    Range.append(value)
                    dateRange = dateRange[1:9]
                    dateRange.append(i)

                    if currentMax == max(Range, default=0):
                        counter += 1
                    else:
                        counter = 0
                    if counter == 5:
                        lastPivot = currentMax
                        dateloc = Range.index(lastPivot)
                        lastDate = dateRange[dateloc]
                        pivots.append(lastPivot)
                        dates.append(lastDate)

                timeD = dt.timedelta(days=30)

                plt.figure(figsize=(12.6, 4.6))
                plt.title(stock + ' История (upper levels)')
                plt.xlabel('01/01/2019 - ' + current_date)
                plt.ylabel('Цена закрытия')
                plt.plot(df['High'], label=stock, alpha=0.35)
                for index in range(len(pivots)):
                    plt.plot_date([dates[index], dates[index] + timeD], [pivots[index], pivots[index]],
                                  linestyle='-',
                                  linewidth=2, marker=",")
                plt.legend(loc='upper left')
                plt.savefig('img3.jpg')

                print('Dates / Prices of pivot points:')
                for index in range(len(pivots)):
                    print(str(dates[index].date()) + ': ' + str(pivots[index]))

            def Low_levels(stock, country):
                current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(
                    date.today().year)
                try:
                    df = investpy.get_stock_historical_data(stock=stock, country=country,
                                                            from_date='01/01/2019',
                                                            to_date=current_date)
                except:
                    df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)

                pivots = []
                dates = []
                counter = 0
                lastPivot = 0

                Range = [999999] * 10
                dateRange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                for i in df.index:
                    currentMin = min(Range, default=0)
                    value = round(df['Low'][i], 2)

                    Range = Range[1:9]
                    Range.append(value)
                    dateRange = dateRange[1:9]
                    dateRange.append(i)

                    if currentMin == min(Range, default=0):
                        counter += 1
                    else:
                        counter = 0
                    if counter == 5:
                        lastPivot = currentMin
                        dateloc = Range.index(lastPivot)
                        lastDate = dateRange[dateloc]
                        pivots.append(lastPivot)
                        dates.append(lastDate)

                timeD = dt.timedelta(days=30)

                plt.figure(figsize=(12.6, 4.6))
                plt.title(stock + ' История (low levels)')
                plt.xlabel('01/01/2019 - ' + current_date)
                plt.ylabel('Цена закрытия')
                plt.plot(df['Low'], label=stock, alpha=0.35)
                for index in range(len(pivots)):
                    plt.plot_date([dates[index], dates[index] + timeD], [pivots[index], pivots[index]],
                                  linestyle='-',
                                  linewidth=2, marker=",")
                plt.legend(loc='upper left')
                plt.savefig('img4.jpg')

                print('Dates / Prices of pivot points:')
                for index in range(len(pivots)):
                    print(str(dates[index].date()) + ': ' + str(pivots[index]))

            def Last_Month(stock, country):
                current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(
                    date.today().year)
                try:
                    df = investpy.get_stock_historical_data(stock=stock, country=country,
                                                            from_date='01/01/2019',
                                                            to_date=current_date)
                except:
                    df = yf.download(stock, start='2019-01-01', end=date.today(), progress=False)
                plt.figure(figsize=(12.6, 4.6))
                plt.plot(df['Close'][-30:], label=stock, alpha=0.35)
                plt.title(stock + ' История за последние 30 дней')
                plt.xlabel('Последние 30 дней')
                plt.ylabel('Цена закрытия')
                plt.legend(loc='upper left')
                plt.savefig('img5.jpg')
                print('Цена за последние пять дней ' + stock + ' =\n', np.array(df['Close'][-5:][0]), '$;\n',
                      np.array(df['Close'][-5:][1]),
                      '$;\n', np.array(df['Close'][-5:][2]), '$;\n', np.array(df['Close'][-5:][3]), '$;\n',
                      np.array(df['Close'][-5:][4]), '$;\n', file=open("output.txt", "a"))
                p_1 = abs(1 - df['Close'][-5:][1] / df['Close'][-5:][0])
                if df['Close'][-5:][1] >= df['Close'][-5:][0]:
                    pp_1 = '+' + str(round(p_1 * 100, 2)) + '%'
                else:
                    pp_1 = '-' + str(round(p_1 * 100, 2)) + '%'
                p_2 = abs(1 - df['Close'][-5:][2] / df['Close'][-5:][1])
                if df['Close'][-5:][2] >= df['Close'][-5:][1]:
                    pp_2 = '+' + str(round(p_2 * 100, 2)) + '%'
                else:
                    pp_2 = '-' + str(round(p_2 * 100, 2)) + '%'
                p_3 = abs(1 - df['Close'][-5:][3] / df['Close'][-5:][2])
                if df['Close'][-5:][3] >= df['Close'][-5:][2]:
                    pp_3 = '+' + str(round(p_3 * 100, 2)) + '%'
                else:
                    pp_3 = '-' + str(round(p_3 * 100, 2)) + '%'
                p_4 = abs(1 - df['Close'][-5:][4] / df['Close'][-5:][3])
                if df['Close'][-5:][4] >= df['Close'][-5:][3]:
                    pp_4 = '+' + str(round(p_4 * 100, 2)) + '%'
                else:
                    pp_4 = '-' + str(round(p_4 * 100, 2)) + '%'
                print('Процент +/- ' + stock + ' =', pp_1, ';', pp_2, ';', pp_3, ';', pp_4,
                      file=open("output.txt", "a"))

            stock = action

            country = 'NASDAQ'

            Stock_SMA(stock, country)
            Stock_EMA(stock, country)
            Upper_levels(stock, country)
            Low_levels(stock, country)
            Last_Month(stock, country)

            # print('img1.jpg')
            # print('img2.jpg')
            # print('img3.jpg')
            # print('img4.jpg')
            # print('img5.jpg')
            with open('img1.jpg', 'rb') as photo:
                await bot.send_photo(chat_id=message.chat.id, photo=photo)
            with open('img2.jpg', 'rb') as photo:
                await bot.send_photo(chat_id=message.chat.id, photo=photo)
            with open('img3.jpg', 'rb') as photo:
                await bot.send_photo(chat_id=message.chat.id, photo=photo)
            with open('img4.jpg', 'rb') as photo:
                await bot.send_photo(chat_id=message.chat.id, photo=photo)
            with open('img5.jpg', 'rb') as photo:
                await bot.send_photo(chat_id=message.chat.id, photo=photo)  # Отправка фото

            with open('output.txt', 'r') as f:
                output = f.read()
            await bot.send_message(chat_id=message.chat.id, text=output)
            path = "output.txt"
            os.remove(path)
        except:
            await message.reply('Некоректный тикер! Повторите попытку!')

@dp.message_handler(lambda message: message.text =="Риск акции")
async def get_name(message: types.Message):
    try:
        def getData(stocks, start, end):
            stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
            stockData = stockData['Close']
            returns = stockData.pct_change()
            meanReturns = returns.mean()
            covMatrix = returns.cov()
            return returns, meanReturns, covMatrix

        # Portfolio Performance
        def portfolioPerformance(weights, meanReturns, covMatrix, Time):
            returns = np.sum(meanReturns * weights) * Time
            std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
            return returns, std

        stockList = [action]
        stocks = [stock for stock in stockList]
        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=360)

        returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)
        returns = returns.dropna()

        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)

        returns['portfolio'] = returns.dot(weights)

        def historicalVaR(returns, alpha=5):
            """
            Read in a pandas dataframe of returns / a pandas series of returns
            Output the percentile of the distribution at the given alpha confidence level
            """
            if isinstance(returns, pd.Series):
                return np.percentile(returns, alpha)

            # A passed user-defined-function will be passed a Series for evaluation.
            elif isinstance(returns, pd.DataFrame):
                return returns.aggregate(historicalVaR, alpha=alpha)

            else:
                raise TypeError("Expected returns to be dataframe or series")

            def historicalCVaR(returns, alpha=5):
                """
                Read in a pandas dataframe of returns / a pandas series of returns
                Output the CVaR for dataframe / series
                """
                if isinstance(returns, pd.Series):
                    belowVaR = returns <= historicalVaR(returns, alpha=alpha)
                    return returns[belowVaR].mean()

                # A passed user-defined-function will be passed a Series for evaluation.
                elif isinstance(returns, pd.DataFrame):
                    return returns.aggregate(historicalCVaR, alpha=alpha)

                else:
                    raise TypeError("Expected returns to be dataframe or series")

        Time = 1

        VaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
        CVaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
        pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

        InitialInvestment = 1000  # ПЕРЕМЕННАЯ
        print('Ожидаемая доходность портфеля в день за 1000$: ', round(InitialInvestment * pRet, 2), '$',
              file=open("risk.txt", "a"))
        print('Значение риска 95 CI: ', round(InitialInvestment * VaR, 2), '%', file=open("risk.txt", "a"))
        print('Условный 95CI: ', round(InitialInvestment * CVaR, 2), '%', file=open("risk.txt", "a"))
        with open('risk.txt', 'r') as f:
            risk = f.read()
        await bot.send_message(chat_id=message.chat.id, text=risk)
        path = "risk.txt"
        os.remove(path)
    except:
        await message.reply('Некоректный тикер! Повторите попытку!')

@dp.message_handler(lambda message: message.text =="Прогноз акции")
async def get_name(message: types.Message):
    try:
        await message.answer('Выполняется анализирование с помощью ИИ')
        await message.answer('Примерное время выполнения 3 минуты')

        AAPL = yf.download(action,
                           start='2020-01-01',
                           end='2022-5-20',
                           progress=False)
        plt.figure(figsize=(16, 8))
        plt.title('Close Price History')
        plt.plot(AAPL['Close'])
        plt.xlabel('Date')
        plt.ylabel('Close price USD')
        plt.savefig('img_ii1.jpg')

        # Создаем новый датафрейм только с колонкой "Close"
        data = AAPL.filter(['Close'])
        # преобразовываем в нумпаевский массив
        dataset = data.values
        # Вытаскиваем количество строк в дате для обучения модели (LSTM)
        training_data_len = math.ceil(len(dataset) * .8)

        # Scale the data (масштабируем)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Создаем датасет для обучения
        train_data = scaled_data[0:training_data_len]
        # разбиваем на x underscore train и y underscore train
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i])
            y_train.append(train_data[i])

        # Конвертируем x_train и y_train в нумпаевский массив
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        # Строим нейронку
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Компилируем модель
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Тренируем модель
        model.fit(x_train, y_train, batch_size=1, epochs=10)

        # Создаем тестовый датасет
        test_data = scaled_data[training_data_len - 60:]
        # по аналогии создаем x_test и y_test
        x_test = []
        y_test = dataset[training_data_len:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i])

        # опять преобразуем в нумпаевский массив
        x_test = np.array(x_test)

        # опять делаем reshape
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        # Получаем модель предсказывающую значения
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Получим mean squared error (RMSE) - метод наименьших квадратов
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

        # Строим график
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Визуализируем
        plt.figure(figsize=(16, 8))
        plt.title('Модель LSTM')
        plt.xlabel('Дата', fontsize=18)
        plt.ylabel('Цена закрытия', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
        plt.savefig('img_ii1.jpg')
        with open('img_ii1.jpg', 'rb') as photo:
            await bot.send_photo(chat_id=message.chat.id, photo=photo)
        print('Прогноз с использованием нейроных сетей', file=open("ii.txt", "a"))
        with open('ii.txt', 'r') as f:
            ii = f.read()
        await bot.send_message(chat_id=message.chat.id, text=ii)
        path = "ii.txt"
        os.remove(path)

    except:
        await message.reply('Некоректный тикер! Повторите попытку!')


@dp.message_handler(lambda message: message.text =="Потенциальная акция")
async def get_name(message: types.Message):
    try:
        tocks = investpy.stocks.get_stocks(country='russia')['symbol']

        counter = 1
        index = 1
        good_stocks = []
        current_date = str(date.today().day) + '/' + str(date.today().month) + '/' + str(date.today().year)
        a = date.today().month
        if a == 1:
            from_date = str(date.today().day) + '/' + str(12) + '/' + str(date.today().year - 1)
        else:
            from_date = str(date.today().day) + '/' + str(date.today().month - 1) + '/' + str(date.today().year)
        for stock in tocks:
            if counter == 1:
                counter = 0
            try:
                df = investpy.get_stock_historical_data(stock=stock, country='russia', from_date=from_date,
                                                        to_date=current_date)
                technical_indicators = investpy.technical.technical_indicators(stock, 'russia', 'stock',
                                                                               interval='daily')
                country = 'russia'
            except:
                continue
            tech_sell = len(technical_indicators[technical_indicators['signal'] == 'sell'])
            tech_buy = len(technical_indicators[technical_indicators['signal'] == 'buy'])
            moving_averages = investpy.technical.moving_averages(stock, country, 'stock', interval='daily')
            moving_sma_sell = len(moving_averages[moving_averages['sma_signal'] == 'sell'])
            moving_sma_buy = len(moving_averages[moving_averages['sma_signal'] == 'buy'])

            moving_ema_sell = len(moving_averages[moving_averages['ema_signal'] == 'sell'])
            moving_ema_buy = len(moving_averages[moving_averages['ema_signal'] == 'buy'])
            if tech_buy < 9 or tech_sell > 2 or moving_sma_buy < 5 or moving_ema_buy < 5:
                continue
            sma_20 = moving_averages['sma_signal'][2]
            sma_100 = moving_averages['sma_signal'][4]
            ema_20 = moving_averages['ema_signal'][2]
            ema_100 = moving_averages['ema_signal'][4]
            print('Фонд =', stock, file=open("out.txt", "a"))
            print('Технические индикаторы продажи: для покупки =', tech_buy, ', 12; ', 'продавать =', tech_sell,
                  ', 12;',
                  file=open("out.txt", "a"))
            print('SMA скользящие средние: покупать =', moving_sma_buy, ', 6; ', 'продавать =', moving_sma_sell,
                  ', 6;',
                  file=open("out.txt", "a"))
            print('EMA скользящие средние: покупать =', moving_ema_buy, ', 6; ', 'продавать =', moving_ema_sell,
                  ', 6;',
                  file=open("out.txt", "a"))
            print('SMA_20 =', sma_20, ';', 'SMA_100 =', sma_100, ';', 'EMA_20 =', ema_20, ';', 'EMA_100 =',
                  ema_100,
                  file=open("out.txt", "a"))
            print('Цены за последние пять дней ' + stock + ' =', np.array(df['Close'][-5:][0]), ';',
                  np.array(df['Close'][-5:][1]),
                  ';', np.array(df['Close'][-5:][2]), ';', np.array(df['Close'][-5:][3]), ';',
                  np.array(df['Close'][-5:][4]), file=open("output.txt", "a"))
            p_1 = abs(1 - df['Close'][-5:][1] / df['Close'][-5:][0])
            if df['Close'][-5:][1] >= df['Close'][-5:][0]:
                pp_1 = '+' + str(round(p_1 * 100, 2)) + '%'
            else:
                pp_1 = '-' + str(round(p_1 * 100, 2)) + '%'
            p_2 = abs(1 - df['Close'][-5:][2] / df['Close'][-5:][1])
            if df['Close'][-5:][2] >= df['Close'][-5:][1]:
                pp_2 = '+' + str(round(p_2 * 100, 2)) + '%'
            else:
                pp_2 = '-' + str(round(p_2 * 100, 2)) + '%'
            p_3 = abs(1 - df['Close'][-5:][3] / df['Close'][-5:][2])
            if df['Close'][-5:][3] >= df['Close'][-5:][2]:
                pp_3 = '+' + str(round(p_3 * 100, 2)) + '%'
            else:
                pp_3 = '-' + str(round(p_3 * 100, 2)) + '%'
            p_4 = abs(1 - df['Close'][-5:][4] / df['Close'][-5:][3])
            if df['Close'][-5:][4] >= df['Close'][-5:][3]:
                pp_4 = '+' + str(round(p_4 * 100, 2)) + '%'
            else:
                pp_4 = '-' + str(round(p_4 * 100, 2)) + '%'
            print('Процент +/- of ' + stock + ' =', pp_1, ';', pp_2, ';', pp_3, ';', pp_4,
                  file=open("out.txt", "a"))
            print()
            with open('out.txt', 'r') as f:
                out = f.read()
            await bot.send_message(chat_id=message.chat.id, text=out)
            path = "out.txt"
            os.remove(path)

            good_stocks.append(stock)
            break
    except:
        await message.reply('Некоректный тикер! Повторите попытку!')

@dp.message_handler(lambda message: message.text =="Оптимизация портфеля")
async def get_name(message: types.Message):
    await message.answer('Введите тикеры через пробел и отправьте сообщение "Выполнить".')


@dp.message_handler(lambda message: message.text =="Выполнить")
async def get_name(message: types.Message):
    try:
        lst = action.replace('.', '').split()
        data = yf.download(lst, period='3mo')
        closeData = data.Close
        closeData
        dCloseData = closeData.pct_change()
        dCloseData
        dohMean = dCloseData.mean()
        dohMean
        cov = dCloseData.cov()
        cov
        cnt = len(dCloseData.columns)

        def randPortf():
            res = np.exp(np.random.randn(cnt))
            res = res / res.sum()
            return res

        r = randPortf()
        print(r)
        print(r.sum())

        def dohPortf(r):
            return np.matmul(dohMean.values, r)

        r = randPortf()
        print(r)
        d = dohPortf(r)
        print(d)

        def riskPortf(r):
            return np.sqrt(np.matmul(np.matmul(r, cov.values), r))

        r = randPortf()
        print(r)
        rs = riskPortf(r)
        print(rs)

        N = 1000

        risk = np.zeros(N)
        doh = np.zeros(N)
        portf = np.zeros((N, cnt))

        for n in range(N):
            r = randPortf()

            portf[n, :] = r
            risk[n] = riskPortf(r)
            doh[n] = dohPortf(r)

        min_risk = np.argmin(risk)

        maxSharpKoef = np.argmax(doh / risk)

        r_mean = np.ones(cnt) / cnt
        risk_mean = riskPortf(r_mean)
        doh_mean = dohPortf(r_mean)

        import pandas as pd

        print('---------- Минимальный риск ----------', file=open("out.txt", "a"))
        print()
        print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.), file=open("out.txt", "a"))
        print("доходность = %1.2f%%" % (float(doh[min_risk]) * 100.), file=open("out.txt", "a"))
        print()
        print(pd.DataFrame([portf[min_risk] * 100], columns=dCloseData.columns, index=['доли, %']).T,
              file=open("out.txt", "a"))
        print()

        print('---------- Макс. коэфф. Шарпа ----------', file=open("out.txt", "a"))
        print()
        print("риск = %1.2f%%" % (float(risk[maxSharpKoef]) * 100.), file=open("out.txt", "a"))
        print("доходность = %1.2f%%" % (float(doh[maxSharpKoef]) * 100.), file=open("out.txt", "a"))
        print()
        print(pd.DataFrame([portf[maxSharpKoef] * 100], columns=dCloseData.columns, index=['доли, %']).T,
              file=open("out.txt", "a"))
        print()

        print('---------- Средний портфель ----------', file=open("out.txt", "a"))
        print()
        print("риск = %1.2f%%" % (float(risk_mean) * 100.), file=open("out.txt", "a"))
        print("доходность = %1.2f%%" % (float(doh_mean) * 100.), file=open("out.txt", "a"))
        print()
        print(pd.DataFrame([r_mean * 100], columns=dCloseData.columns, index=['доли, %']).T, file=open("out.txt", "a"))
        print()
        with open('out.txt', 'r') as f:
            out = f.read()
        await bot.send_message(chat_id=message.chat.id, text=out)
        path = "out.txt"
        os.remove(path)
    except:
        await message.reply('Некоректный тикер! Повторите попытку!')

@dp.message_handler(lambda message: message.text =="Анализ портфеля")
async def get_name(message: types.Message):
    await message.answer('Введите тикеры и кол-во купленных акций через пробел и отправьте сообщение "Анализ". Пример: "AAPL 3 OZON 5"')

@dp.message_handler(lambda message: message.text =="Анализ")
async def get_name(message: types.Message):
    try:
        await message.answer('Подождите немного!')
        lst = action.replace('.', '').split()
        f = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        q = list()
        cen = list()
        cen5 = list()
        cen1 = list()
        divlist = []
        divlist1 = []
        for key in f:
            tickers = [key]
            for ticker in tickers:
                ticker_yahoo = yf.Ticker(ticker)
                data = ticker_yahoo.history()
                last_quote = (data.tail(1)['Close'].iloc[0])
                price = ticker_yahoo.info.get("currentPrice") * int(f.get(key))
                price = round(price, 2)
                f[key] = price
                naming = ticker_yahoo.info.get("shortName")
                cen.append(key)
                cen.append(str(ticker_yahoo.info.get("shortName")))

                cen1.append(key)
                cen1.append(ticker_yahoo.info.get("currentPrice"))

                cen5.append(key)
                cen5.append(str(price))

                q.append(str(naming))
                q.append(str(price))

                divlist.append(key)
                divlist.append(str(ticker_yahoo.info.get('lastDividendValue')))

                divlist1.append(key)
                divlist1.append(ticker_yahoo.info.get("shortName"))

        cena5 = {cen5[i]: cen5[i + 1] for i in range(0, len(cen5), 2)}
        cena1 = {cen1[i]: cen1[i + 1] for i in range(0, len(cen1), 2)}
        cena = {cen[i]: cen[i + 1] for i in range(0, len(cen), 2)}
        div = {divlist[i]: divlist[i + 1] for i in range(0, len(divlist), 2)}
        div1 = {divlist1[i]: divlist1[i + 1] for i in range(0, len(divlist1), 2)}


        print("Стоимость акции:", file=open("price1.txt", "a"))
        for key in cena:
            lis1 = list(cena.keys())
            print(" " + cena.get(key) + " - " + str(cena1.get(key)) + "$", file=open("price1.txt", "a"))
        print(" ", file=open("price1.txt", "a"))
        print("Стоимость всех акций:", file=open("price2.txt", "a"))
        for key in cena:
            lis1 = list(cena.keys())
            print(" " + cena.get(key) + " - " + str(cena5.get(key)) + "$", file=open("price2.txt", "a"))
        print(" ", file=open("price2.txt", "a"))
        print("Дивидентовый доход:", file=open("div.txt", "a"))
        for key in div:
            lis1 = list(div.keys())
            print(" " + div1.get(key) + " - " + str(div.get(key)) + "$", file=open("div.txt", "a"))
        lis_sum = list(cena5.values())
        arr = np.array(lis_sum)
        arr = arr.astype('float64')
        print(" ", file=open("div.txt", "a"))
        print("Стоимость портфеля:", file=open("price.txt", "a"))
        print(" ", round(arr.sum(), 2), "$", file=open("price.txt", "a"))
        with open('price1.txt', 'r') as f:
                price1 = f.read()
        with open('price2.txt', 'r') as f:
                price2 = f.read()
        with open('div.txt', 'r') as f:
                div = f.read()
        with open('price.txt', 'r') as f:
                price = f.read()
        output1 = "✅" + price1 + "✅" + price2 + "✅" + div + "✅" + price
        await bot.send_message(chat_id=message.chat.id, text=output1)
        path = "price1.txt"
        os.remove(path)
        path = "price2.txt"
        os.remove(path)
        path = "div.txt"
        os.remove(path)
        path = "price.txt"
        os.remove(path)
    except:
        await message.reply('Некоректный тикер! Повторите попытку!')
if __name__ == "__main__":
    # Запуск бота
    executor.start_polling(dp, skip_updates=True)



