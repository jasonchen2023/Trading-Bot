# Design

There are two classes defined for the trading system. One is `Company`, which stores the company's data and calculates the probability of the company's stock rising using machine learning. The other class is `TradingModel`, which runs the algorithm to select the best stocks and actually trades the stock.


### company.py

This file defines the `Company` class, which is used to store, process, and analyze data for a company. 

The program first downloads the data of a company using yfinance, creates the features using TA-Lib, and stores all of the data in a Pandas dataframe. Then, it runs the machine learning model to predict the probabilities of the stock rising for all of days in our testing data set. The probability of the stock rising in the most recent day (today) is stored in a variable and later used by the trading bot.

The `Company` class contains the following functions:
```c
download_stock_information() – downloads company data from yfinance, creates features, and stores data in Pandas dataframe
split_train_and_test_data() – splits data using time-based splitting
extract_train_and_test() – helper function to extract train and test data from last time-series split
learn_binary_model() – runs ML model using our feature and output variables. Returns model, f1 score, and probability estimates
```

### trading_model.py

This file defines the `TradingModel` class, which is basically the trading bot. The trading bot creates company objects for each company. With the data for each company, the trading bot runs its algorithm to determine the best stocks to buy and purchases the stocks using the Alpaca API.

The class contains the following functions:
```c
get_companies_probs() – returns a dictionary with company as key, probability of stock rising as value. Also creates company objects and stores them in list
get_highest_ranked_stocks() – returns highest ranked stocks in a list based on probability of rising for the day
get_stocks_above_threshold() – returns list of stocks whose probability of stock rising is above threshold
sell_all_stocks() – sell all stocks in portfolio
buy_stocks() – buy best stocks as determined by algorithm
simulate() – simulate various trading strategies for a given period of time
simulate_buy_and_hold() – helper function to calculate buy and hold returns for simulation
simulate_trading_when_prob_above_threshold() – helper function to calculate returns for simulating trading stocks above threshold
simulate_trading_highest_ranked_stocks() – helper function to calculate returns for trading highest ranked stocks
```

### trade.py

A command line interface that allows the user to run the trading bot. The user selects the trading strategy they want to use and passes in the necessary information (number of companies to trade if trading highest ranked stocks, threshold if trading stocks with probabilities above certain threshold). The program automatically sells all stocks in the portfolio and buys the best stock based on the algorithm.