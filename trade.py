from trading_model import TradingModel
import pandas as pd
import sys
import time

def load_company_names(file):
    df = pd.read_csv(file)
    companies = []
    for index,row in df.iterrows():
        if row['Ticker'] not in companies:
            companies.append(row['Ticker'])
    return companies

# load tickers into list
FILE = sys.argv[1]
company_names = load_company_names(FILE)
num_companies = len(company_names)

# query trading strategy
print("\n", "There are two trading strategies:")
print("1. Trade stocks with highest probabilities of appreciating")
print("2. Trade all stocks whose probability of appreciating is above a certain threshold \n")

strategy = None
bot = None
while strategy != "1" and strategy != "2":
    strategy = input("Please enter \"1\" for strategy 1, \"2\" for strategy 2: ")

# create model for strategy 1
if strategy == "1":
    
    print("\n", "Companies list size is: ", num_companies)
    num_companies_to_trade = int(input("Please enter number of companies to trade: "))

    while num_companies_to_trade > len(company_names):
        print("Cannot trade more than companies list size: ", num_companies)
        num_companies_to_trade = int(input("Please re-enter number of companies to trade: "))    
    
    bot = TradingModel(company_names, 0, num_companies_to_trade, "1")    # create trading model object

    print("\n", "Top ", num_companies_to_trade, " companies with highest probability are: ", bot.stocks_to_buy, "\n")

# create model for strategy 2
elif strategy == "2":
    
    threshold = float(input("Please enter a threshold between 0 and 1: "))
    while threshold < 0 or threshold > 1:
        threshold = float(input("Please enter a probability threshold between 0 and 1: "))
    
    bot = TradingModel(company_names, threshold, 0, "2")    # create trading model object

    print("\n", "Stocks with probability above ", threshold, " threshold are: ", bot.stocks_to_buy, "\n")


# Sell all stocks
time.sleep(0.5)
print("\n Selling all stocks in portfolio...\n")
time.sleep(0.5)
bot.sell_all_stocks()


# Buy stocks
print("Equity: $" + str(bot.equity), "\n")
time.sleep(0.5)
print("\n Buying stocks:", bot.stocks_to_buy, "\n")
time.sleep(0.5)
bot.buy_stocks()

print("\n Order executed \n")



