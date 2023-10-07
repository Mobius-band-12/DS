import pandas as pd
import numpy as np
from prophet import Prophet
import pickle

with open("model1.pkl", 'rb') as file:  
    model1 = pickle.load(file)

with open("model2.pkl", 'rb') as file:  
    model2 = pickle.load(file)

def forecast(item_info: dict, store_info: str, stats_by_store_item: dict, stats_by_store: dict, stats_by_item: dict, dates: list, h=14) -> list:
    """
    Функция для предсказания продаж. Определяет, какой моделью предсказывать пару товар-магазин, полученную на вход
    :params item_info: айди товара, его группа, категория, подкатегория
    :params store_info: айди магазина
    :stats_by_store_item: продажи данного товара в данном магазине в последние три недели перед прогнозом
    :stats_by_store: суммарные продажи в данном магазине в последние три недели перед прогнозом
    :stats_by_item: суммарные продажи данного товара во всех магазинах, в рублях и штуках, за все доступные дни
    :dates: список дат для прогноза
    :h: количество дней для прогноза (len(dates) == h)

    """
    store_item = store_info + '-' + item_info['sku']
    model1_list = pd.read_csv('model1_list.csv')['store_item'].to_list()

    if store_item in model1_list:
        forecast = get_forecast_from_model1(store_item, item_info['sku'], store_info, stats_by_store_item, stats_by_item, dates, h)
    else: forecast = get_forecast_from_model2(item_info, store_info, stats_by_store, stats_by_item, dates, h)

    return forecast

def get_forecast_from_model1(store_item, item_info, store_info, stats_by_store_item, stats_by_item, dates, h):
    """
    Функция для предсказания первой моделью. Первая модель работает с наиболее частотными парами товар-магазин, характеризующимися стабильным спросом
    :store_item: пара товар-магазин
    :item_info: айди товара
    :store_info: айди магазина
    :stats_by_store_item: продажи данного товара в данном магазине в последние три недели перед прогнозом
    :stats_by_item: суммарные продажи данного товара во всех магазинах, в рублях и штуках, за все доступные дни
    :dates: список дат для прогноза
    :h: количество дней для прогноза (len(dates) == h)

    """
    targets = []

    #predict price with Prophet
    future_prices = get_prophet_prices(stats_by_item, h=h)

    for i in range(h):
        #create lower row - that we're to fill with features
        lower_row = pd.DataFrame({'store_item': store_item, 'pr_sku_id': item_info, 'st_id': store_info, 'date': pd.to_datetime(dates[i])}, index=[0])

        #if line is first in forecast period - take lags from the last line of train. otherwise from previous predicted line
        if i == 0:
            sales_history = sorted(list(stats_by_store_item.items()), key=lambda x: x[0])
            last_train_date = pd.to_datetime(sales_history[-1][0])            
            upper_row = pd.DataFrame({'date': last_train_date,
                                      'store_item': store_item,
                                      'sold': sales_history[-1][1]}, index=[0])
            for n in range(1, 21):
                upper_row[['sold_lag_' + str(n)]] = sales_history[-1-n][1]
            upper_row['sold_lag_21'] = np.nan   #not required
                
        else:
            upper_row = last_preprocessed

        #glue two lines together, the full and the empty that needs features to be added
        pred_with_history = pd.concat([upper_row, lower_row])
        last_preprocessed = model1_preprocess_test(pred_with_history, future_prices[i])

        #drop columns so that features corresponded to those model had been trained on
        X_test = last_preprocessed.drop(['sold', 'date', 'pr_sku_id', 'st_id'], axis=1)
        
        #fix categorical
        cat_features=['store_item', 'pr_sales_type_id', 'dow', 'season']
        for col in cat_features:
            X_test[col] = pd.Categorical(X_test[col])

        pred = model1.predict(X_test)

        #prediction becomes "fact" of the day we forecast to become lag_1 for the next day. also it goes to forecast table as target
        last_preprocessed['sold'] = round(pred.item())
        targets.append(round(pred.item()))

    return targets

def model1_preprocess_test(pred_with_history, pred_price):
    """
    Функция для предобработки строк, используемых для прогноза. Добавляет все необходимые фичи, на которых обучалась модель
    :pred_with_history: датафрейм из двух строк, где верхняя заполнена значениями, а нижняя требует заполнения
    :pred_price: цена, предсказанная на заданный день

    """

    #index needs to be reset, as for non-first lines in a forecast they are equal
    pred_with_history = pred_with_history.reset_index(drop=True)

    #add lags
    pred_with_history[['sold_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['sold_lag_' + str(n) for n in range(1, 22)]].fillna(method='ffill')
    pred_with_history[['sold_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['sold_lag_' + str(n) for n in range(1, 22)]].shift(1, axis=1)
    pred_with_history.loc[pred_with_history.index[-1], 'sold_lag_1'] = pred_with_history.loc[pred_with_history.index[0], 'sold']
        
    #no promo by default
    pred_with_history['pr_sales_type_id'] = 0
    pred_with_history['discount'] = 0    
    
    #add features of day-of-week, day-of-month, week, month, season
    pred_with_history['dow'] = pred_with_history['date'].dt.dayofweek
    pred_with_history['day'] = pred_with_history['date'].dt.day
    pred_with_history['week'] = pred_with_history['date'].dt.isocalendar().week.astype('int32')
    pred_with_history['month'] = pred_with_history['date'].dt.month
    pred_with_history['season'] = pred_with_history['date'].dt.quarter    
    
    #add Prophet-predicted price    
    pred_with_history['avg_daily_price'] = pred_price    
    
    #add moving averages
    for n in range(2, 22):
        pred_with_history['ma_' + str(n)] = pred_with_history[['sold_lag_' + str(x) for x in range(1, n+1)]].sum(axis=1) / n

    #drop upper row
    pred_with_history = pred_with_history.drop(pred_with_history.index[0], axis=0)    

    return(pred_with_history)

def get_forecast_from_model2(item_info, store_info, stats_by_store, stats_by_item, dates, h):
    """
    Функция для предсказания второй моделью. Вторая модель работает с мало продаваемыми товарами, характеризующимися нестабильным спросом
    :item_info: айди товара
    :store_info: айди магазина
    :stats_by_store: суммарные продажи в данном магазине в последние три недели перед прогнозом
    :stats_by_item: суммарные продажи данного товара во всех магазинах, в рублях и штуках, за все доступные дни
    :dates: список дат для прогноза
    :h: количество дней для прогноза (len(dates) == h)

    """
    targets = []

    #get dataframes with historic sales
    # sales = pd.read_csv('model2_sales.csv', parse_dates=['date'])
    # byproduct = pd.read_csv('model2_byproduct.csv', parse_dates=['date'])

    #predict price with Prophet
    future_prices = get_prophet_prices(stats_by_item, h=h)
    
    for i in range(h):
        #create lower row - that we're to fill with features
        lower_row = pd.DataFrame({'pr_sku_id': item_info['sku'], 'st_id': store_info, 'date': dates[i]}, index=[0])

        #if line is first in forecast period - take lags from the last line of train. otherwise from previous predicted line
        #first match sku to get sku lags
        if i == 0:
            sales_history = sorted(list(stats_by_item.items()), key=lambda x: x[0])[-21:]
            last_train_date = pd.to_datetime(sales_history[-1][0])            
            upper_row = pd.DataFrame({'pr_sku_id': item_info['sku'],
                                      'st_id': store_info,
                                      'date': last_train_date,                                      
                                      'sku': sales_history[-1][1][0]}, index=[0])
            for n in range(1, 21):
                upper_row[['sku_lag_' + str(n)]] = sales_history[-1-n][1][0]
            upper_row['sku_lag_21'] = np.nan   #not required
        else:
            upper_row = last_preprocessed

        #glue two lines together, the full and the empty that needs features to be added
        pred_with_history = pd.concat([upper_row, lower_row])        
        sku_preprocessed = model2_preprocess_test_by_sku(pred_with_history, future_prices[i], item_info)

        #now match store to get store lags
        if i == 0:
            sales_history = sorted(list(stats_by_store.items()), key=lambda x: x[0])
            last_train_date = pd.to_datetime(sales_history[-1][0])            
            upper_row = pd.DataFrame({'pr_sku_id': item_info['sku'],
                                      'st_id': store_info,
                                      'date': last_train_date,
                                      'store': sales_history[-1][1]}, index=[0])
            upper_row['sold'] = np.nan   #not required
            for n in range(1, 22):
                upper_row[['sku_lag_' + str(n)]] = np.nan   #not required
            for n in range(1, 21):
                upper_row[['store_lag_' + str(n)]] = sales_history[-1-n][1]
            upper_row['store_lag_21'] = np.nan   #not required
            
        else:
            upper_row = last_preprocessed

        #glue two lines together, the full and the empty that needs features to be added
        pred_with_history = pd.concat([upper_row, sku_preprocessed])  
        last_preprocessed = model2_preprocess_test_by_store(pred_with_history)

        #drop columns so that features corresponded to those model had been trained on
        X_test = last_preprocessed.drop(['date', 'sold', 'sku', 'store'], axis=1)

        #fix categorical
        cat_features=['pr_sku_id', 'st_id', 'season', 'dow', 'pr_subcat_id', 'pr_cat_id', 'pr_group_id', 'pr_sales_type_id']
        for col in cat_features:
            X_test[col] = pd.Categorical(X_test[col])
            
        pred = model2.predict(X_test)

        #prediction becomes "fact" of the day we forecast to become lag_1 for the next day. also it goes to forecast table as target
        last_preprocessed['sold'] = round(pred.item())
        targets.append(round(pred.item()))
    
    return targets

def model2_preprocess_test_by_sku(pred_with_history, pred_price, item_info):
    """
    Функция для предобработки строк, используемых для прогноза. Добавляет все необходимые фичи, на которых обучалась модель, относящиеся к товару
    :pred_with_history: датафрейм из двух строк, где верхняя заполнена значениями, а нижняя требует заполнения
    :pred_price: цена, предсказанная на заданный день
    :item_info: словарь с инфо о товаре. Из него извлекаются категория, подкатегория, группа

    """
    #index needs to be reset, as for non-first lines in a forecast they are equal
    pred_with_history = pred_with_history.reset_index(drop=True)

    #add features of day-of-week, day-of-month, week, month, season
    pred_with_history['dow'] = pred_with_history['date'].dt.dayofweek
    pred_with_history['day'] = pred_with_history['date'].dt.day
    pred_with_history['week'] = pred_with_history['date'].dt.isocalendar().week.astype('int32')
    pred_with_history['month'] = pred_with_history['date'].dt.month
    pred_with_history['season'] = pred_with_history['date'].dt.quarter 
    

    #add Prophet-predicted price    
    pred_with_history['avg_daily_price'] = pred_price
    
    #add items catalogue features
    pred_with_history['pr_subcat_id'] = item_info['subcategory']
    pred_with_history['pr_cat_id'] = item_info['category']
    pred_with_history['pr_group_id'] = item_info['group']

    #add lags - sku
    pred_with_history[['sku_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['sku_lag_' + str(n) for n in range(1, 22)]].fillna(method='ffill')
    pred_with_history[['sku_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['sku_lag_' + str(n) for n in range(1, 22)]].shift(1, axis=1)
    pred_with_history.loc[pred_with_history.index[-1], 'sku_lag_1'] = pred_with_history.loc[pred_with_history.index[0], 'sku']       
    
    #no promo by default
    pred_with_history['pr_sales_type_id'] = 0

    #drop upper row
    pred_with_history = pred_with_history.drop(pred_with_history.index[0], axis=0)    

    return(pred_with_history)

def model2_preprocess_test_by_store(pred_with_history):
    """
    Функция для предобработки строк, используемых для прогноза. Добавляет все необходимые фичи, на которых обучалась модель, относящиеся к магазину
    :pred_with_history: датафрейм из двух строк, где верхняя заполнена значениями, а нижняя требует заполнения

    """
    #index needs to be reset, as for non-first lines in a forecast they are equal
    pred_with_history = pred_with_history.reset_index(drop=True)
    
    #add lags - store
    pred_with_history[['store_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['store_lag_' + str(n) for n in range(1, 22)]].fillna(method='ffill')
    pred_with_history[['store_lag_' + str(n) for n in range(1, 22)]] = pred_with_history[['store_lag_' + str(n) for n in range(1, 22)]].shift(1, axis=1)
    pred_with_history.loc[pred_with_history.index[-1], 'store_lag_1'] = pred_with_history.loc[pred_with_history.index[0], 'store']
    
    #drop upper row
    pred_with_history = pred_with_history.drop(pred_with_history.index[0], axis=0)   

    return(pred_with_history)


def get_prophet_prices(stats_by_item, h):
    """
    Функция для предсказания цены Профетом. Результат используется моделями как признак вместо реальной средней цены на заданную дату
    :stats_by_item: исторические данные о продажах товара для обучения модели
    :h: количество дней для прогноза

    """
    rows = list(stats_by_item.items())
    sales = pd.DataFrame({'ds': [el[0] for el in rows], 'units': [el[1][0] for el in rows],
                          'rubs': [el[1][1] for el in rows]})
    sales['y'] = sales['rubs'] / sales['units']
    avg_price_ts_df = sales[['ds', 'y']]
    avg_price_ts_df = avg_price_ts_df.sort_values(by='ds')
    
    prophet = Prophet()
    prophet.fit(avg_price_ts_df)
    future = prophet.make_future_dataframe(periods=h, freq='D')
    prices_forecast = prophet.predict(future)
    future_prices = prices_forecast['yhat'].tail(h).to_list()
    return future_prices