# Trading-Bot

This trading bot uses machine learning to predict stock movements. It determines the best stocks to buy and trades the stocks using the Alpaca API.

### Usage

- Enter your Alpaca API key and secret key in the credential file. You can create an Alpaca account [here](https://alpaca.markets)
- Download libraries below
- To run trading bot: `python3 trade.py [path_to_csv_file]`

### Tools and Libraries

- [yfinance](https://pypi.org/project/yfinance/) to download market data from Yahoo! Finance's API
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) to perform technical analysis of financial market data and create features
- [scikit-learn](https://scikit-learn.org/stable/) to build my machine learning models
- [Pandas](https://pandas.pydata.org) to read, manipulate, and store data
- [NumPy](https://numpy.org) to work with data and create features
- [Alpaca API](https://alpaca.markets) to trade stocks

### ML models

I trained several models including random forest, Logistic Regression, and Naive Bayes using the dataset for Apple. Random forest had the highest [F1-score](https://deepai.org/machine-learning-glossary-and-terms/f-score).

| Model      | F1 Score |
| ----------- | ----------- |
| Random Forest     | 0.3488      |
| Logistic Regression   | 0.3471  |
| Naive Bayes     | 0.3471       | 
| f1 Dummy   | 0.3189        |


### Features

I had 36 features for my model. Some of the main ones include open price, close price, high, low, volume, dividends, daily return, moving average, etc.

### Trading strategies

I use the **random forest** classifier to predict the probability of a stock appreciating today. My trading model determines the probabilities for all the stocks entered in the csv file.

With the probabilities of rising for all companies, there are two available strategies for trading:

1. **Trade top X stocks with highest probabilities**
2. **Trade all stocks with probabilities above a certain threshold**

 Trade top X stocks with highest probabilities – The user defines the number of stocks to trade (X) when they instantiate the model. This strategy trades the top X companies with highest probabilities.

Trade all stocks with probabilities above a certain threshold – The user defines a threshold between 0 and 1. The trading bot buys all stocks whose probabilities of rising is above the threshold.

In both strategies, all stocks that are bought will be weighted the same. That is, the nominal value bought will be the total cash available / number of stocks to buy.

I run the algorithm in the morning when the market opens. The trading_model object automatically creates a list of stocks to buy based on the strategy the user enters. The user can buy the stocks by calling `buy_stocks()`. Before buying the stocks, I sell all stocks in my portfolio from the previous day through `sell_all_stocks()` to maximize the cash available.

### Trading

I used the [Alpaca API](https://alpaca.markets) to trade stocks. Along with real trading, Alpaca also supports paper trading, which trades with "fake money" and is great for testing purposes.

### Simulation

Along with the ability to trade real stocks through Alpaca, my trading bot also includes a simulation feature. The simulation calculates the returns if an investor had used my model to trade stocks in the previous X number of days. The simulation can be run by calling the `simulate` function, passing in the number of days to evaluate, number of stocks to trade, and probability threshold:
- Number of days to simulate trading
    - E.g. If a user enters 300, the program will simulate trading for the previous 300 days 
- Number of stocks to trade (for strategy 1)
    - E.g. If a user enters 10, the program will trade the top 10 stocks with highest probabilies of rising
- Probability threshold at which a stock should be traded (for strategy 2)
    - Integer between 0 and 1
    - E.g. If the user enter 0.45, program will trade all stocks with probabilities above 0.45

The simulation automatically calculates and compares the returns for the two strategies listed above, as well as a buy and hold strategy where the investor buys on the first day and sells on the last day. Thus, one could compare the returns for different strategies for the same period of time.

### Next steps

This project is still a work in progress. Some things I'd like to improve/explore:
- **User interface**
    - Having a more advanced command line interface or graphical user interface that allows the user to trade stocks
- **Advanced trading strategies**
    - Right now, I do not weigh the stocks that are traded, so one idea is to assign stocks with higher rising probabilities greater weight. Another idea is to explore shorting stocks with high probabilities of depreciating.
- **Incorporate deep learning models**
    - Deep learning is a subset of machine learning that uses [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network). I'd like to explore if deep learning models are more accurate than the ML models I used.
- **Deploy program onto AWS**
    - Using AWS resources such as Lambda and Eventbridge rules to automatically run the algorithm every morning.
