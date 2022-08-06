import pandas as pd
import yfinance as yf
import talib as ta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


class Company:

    def __init__(self, company):
        self.company = company
        self.stock_hist = self.download_stock_information(company)
        self.model, self.f1_score, self.predict_prob = self.learn_binary_model("Random Forest")

        self.up_prob = self.predict_prob[-1][1]

    def download_stock_information(self, ticker):
        # Download ticker information from yfinance 
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period="max").drop(columns=['Stock Splits']) # Drop unecessary columns

        # Add feature vectors  
        stock_hist['Max Price (Last 7)'] = stock_hist['Close'].rolling(7).max()
        stock_hist['Min Price (Last 7)'] = stock_hist['Close'].rolling(7).min()
        stock_hist['SD Price (Last 7)'] = stock_hist['Close'].rolling(7).std()

        def calculate_average(df, days):    # Calculate the moving average
            for ma in days:
                column_name = "MA for %s days" %(str(ma))
                df.loc[:,column_name]=pd.DataFrame.rolling(df['Close'],ma).mean()
        days = [3, 10, 14, 21, 50, 100]
        calculate_average(stock_hist, days)

        # Add TA-lib trading indicators 

        # Relative Strength - price oscillator that ranges between 0 and 100. Divergence with the price is used to hint for potential reversals 
        stock_hist['Relative Strength'] = ta.RSI(stock_hist['Close'])
    
        # Exponential Moving Average - a derivative of typical moving average that places a greater weight and significance on the most recent data points 
        stock_hist['Exponential MA'] = ta.EMA(stock_hist['Close'])

        # Bollinger Bands - envelopes plotted at a standard deviation level above and below a simple moving average of the price (a measure of volatility swings)
        stock_hist['BOL_Up'], stock_hist['BOL_Mid'], stock_hist['BOL_Low'] = ta.BBANDS(stock_hist['Close'],timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # On Balance Volume - a momentum indicator that uses volue flow to predict changes in stock price
        stock_hist['OBV'] = ta.OBV(stock_hist['Close'], stock_hist['Volume'])

        # Momementum - compares the current price with the previous price from a given number of periods ago to determine the rate at which the price is moving
        stock_hist['Momentum (10)'] = ta.MOM(stock_hist['Close'], timeperiod=10)
        stock_hist['Momentum (30)'] = ta.MOM(stock_hist['Close'], timeperiod=30)

        # Rate of change - a momentum-based indicator that measures the percentage change in price between the current price and the price of a certain number of periods ago 
        stock_hist['Rate of Change'] = ta.ROC(stock_hist['Close'])

        # Stochastic - another momentum based indicator that compares the closing price to a range of its prices over a certain period of time 
        stock_hist['slowk'], stock_hist['slowd'] = ta.STOCH(stock_hist['High'], stock_hist['Low'], stock_hist['Close'])

        # Trying out the some of the candlestick patterns within TA-Lib -
        # These are are variety of well known candlestick patterns that traders use to help there predicted movements 
        stock_hist['CDL2CROWS'] = ta.CDL2CROWS(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLBELTHOLD'] = ta.CDLBELTHOLD(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLHAMMER'] = ta.CDLHAMMER(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLLONGLINE'] = ta.CDLLONGLINE(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])
        stock_hist['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(stock_hist['Open'], stock_hist['High'], stock_hist['Low'], stock_hist['Close'])

        stock_hist['Daily Return'] = ((stock_hist['Close']-stock_hist['Open'])/stock_hist['Open'])*100
        
        # Addd our prediction value (in this case just binary classification)
        stock_hist['1-Day Price Change'] = stock_hist['Close'].diff()

        conditions = [
            (stock_hist['1-Day Price Change'] <= 0),
            (stock_hist['1-Day Price Change'] > 0)        
        ]
        values = [-1, 1]
        stock_hist['Appreciation or Depreciation'] = np.select(conditions, values)
        
        # Add the shift 
        stock_hist['Next Day\'s Close'] = stock_hist['Close'].shift(-1)
        stock_hist['Next Day\'s Appreciation or Depreciation'] = stock_hist['Appreciation or Depreciation'].shift(-1)

        # Too handle the fact that we have Nan for some of the moving average values up to day 100 we determine a cut-off
        stock_hist = stock_hist.drop(index=stock_hist.index[:101], 
                axis=0, 
                inplace=False)

        # Because of this shift we want to drop the last row 
        # stock_hist.drop(stock_hist.tail(1).index,inplace=True)
        stock_hist.reset_index(drop=False, inplace=True) # reset index
        
        return stock_hist


    # extract training and test data based on index
    def extract_train_and_test(self, X, y, train_index, test_index):
        X_train = X[train_index[0]:train_index[-1]+1]
        y_train = y[train_index[0]:train_index[-1]+1]
        
        X_test = X[test_index[0]:test_index[-1]+1]
        y_test = y[test_index[0]:test_index[-1]+1]

        return X_train, y_train, X_test, y_test

    # splits data into train and test data
    def split_train_and_test_data(self, X, y):
        
        # Split our data into times series based information
        tscv = TimeSeriesSplit(n_splits=4)
        train_index_list = []
        test_index_list = []
        for train_index, test_index in tscv.split(X):
            train_index_list.append(train_index)
            test_index_list.append(test_index)

        self.test_index_list = test_index_list[-1]
        
        X_train, y_train, X_test, y_test = self.extract_train_and_test(X, y, train_index_list[-1], test_index_list[-1])
        
        y_train, y_test = y_train.to_numpy().astype(int), y_test.to_numpy().astype(int)

        return X_train, y_train, X_test, y_test, test_index_list



    # returns model, f1 score, and probability estimates
    def learn_binary_model(self, model_name):
        
        # Select features as well as what we are predicting 
        X = self.stock_hist[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Daily Return',
                'Max Price (Last 7)', 'Min Price (Last 7)','SD Price (Last 7)',
                'MA for 3 days', 'MA for 10 days','MA for 14 days', 'MA for 21 days',
                'MA for 50 days', 'MA for 100 days', 'Relative Strength','Exponential MA', 
                'BOL_Up', 'BOL_Mid', 'BOL_Low', 'OBV','Momentum (10)', 'Momentum (30)',
                'Rate of Change', 'slowk', 'slowd', 'CDL2CROWS', 'CDL3OUTSIDE',
                'CDLBELTHOLD', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHAMMER',
                'CDLLONGLINE', 'CDLMORNINGDOJISTAR', 'CDLXSIDEGAP3METHODS']]
        y = self.stock_hist['Next Day\'s Appreciation or Depreciation']
        X_train, y_train, X_test, y_test, test_index_list = self.split_train_and_test_data(X, y)
        
        model = None
        if model_name == "Dummy":
            model = DummyClassifier(strategy='most_frequent', random_state=43)

        elif model_name == "Random Forest":
            model = RandomForestClassifier(max_depth=3, random_state=0)
        
        model.fit(X_train, y_train)
        return model, self.get_f1_score(model, X_test, y_test), model.predict_proba(X_test)


    def get_f1_score(self, model, X_test, y_test):
        predict = model.predict(X_test)
        return f1_score(y_test, predict, average='macro')
