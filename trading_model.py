import pandas as pd
from company import Company
import os
import alpaca_trade_api as tradeapi
from credentials import *

os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"

class TradingModel:

    def __init__(self, stocks, threshold, number_of_stocks_to_rank, trading_strategy):
        self.stocks = stocks
        self.companies = {}
        self.company_up_probs = self.get_companies_probs(-1)
        self.number_of_stocks_to_rank = number_of_stocks_to_rank

        self.api = tradeapi.REST(API_KEY, SECRET_KEY)
        self.equity = self.api.get_account().equity

        self.threshold = threshold
        
        self.stocks_to_buy = []
        if trading_strategy == "1":
            self.stocks_to_buy = self.get_highest_ranked_stocks(self.company_up_probs, self.number_of_stocks_to_rank)
        elif trading_strategy == "2":
            self.stocks_to_buy = self.get_stocks_above_threshold(self.company_up_probs, self.threshold)


    # returns dictionary with company as key, probability of stock rising as value. Also creates company objects and stores in list
    def get_companies_probs(self, index):
        company_probs = {}
        for ticker in self.stocks:
            try:
                globals()[ticker] = Company(ticker)
                company = globals()[ticker]                

                self.companies[ticker] = company  # add company object to list
                company_probs[ticker] = round(company.up_prob, 4)   # add company up prob to dictionary
                print(ticker, "prob: ", company_probs[ticker])

            except:
                print("Error with ", ticker) 

        return company_probs


    # return highest ranked stocks in a list based on probability of rising for the day
    def get_highest_ranked_stocks(self, company_probs, num_stocks_to_rank):
        stocks_sorted_by_prob = sorted(company_probs.items(), key=lambda item: item[1])
        best_stocks = []
        
        for i in range(num_stocks_to_rank):
            best_stocks.append(stocks_sorted_by_prob[len(stocks_sorted_by_prob)-1-i][0])
        
        return best_stocks


    # return list of companies whose probability of stock rising is above threshold
    def get_stocks_above_threshold(self, company_probs, threshold):
        companies_above_threshold = []

        for company in company_probs:
            if company_probs[company] > threshold:
                companies_above_threshold.append(company)
        
        return companies_above_threshold


    # sell all stocks in portfolio
    def sell_all_stocks(self):
        positions = self.api.list_positions()

        for position in positions:
            try:
                self.api.submit_order(position.symbol, position.qty, "sell", "market", "gtc")
                print("sold", position.qty, "shares of", position.symbol)
            except:
                print("Error selling", position.symbol)    
        
        self.equity = self.api.get_account().equity


    # purchase equal notional amounts of stocks in list
    def buy_stocks(self):
        cash = float(self.api.get_account().cash)
        order_amt = cash / len(self.stocks_to_buy)
        
        for stock in self.stocks_to_buy:
            try:
                self.api.submit_order(symbol=stock, notional=order_amt, side="buy", type="market", time_in_force="day")
                print("bought $" + str(order_amt), "of", stock)
            except:
                print("Error buying ", stock)


    # calculate buy and hold return for the day, also returning dictionary with probability of stock rising for each company
    def simulate_buy_and_hold(self, num_days, day):
        b_h_day_return = 0
        company_probs = {}
        companies_count = 0
        
        for ticker in self.companies:
            company = self.companies[ticker]
            
            if len(company.test_index_list) < num_days: # ensure we have enough days of data for company to simulate
                continue

            prob = company.predict_prob[-day-1][1]
            company_probs[ticker] = prob
 
            percentage_change = self.compute_return(self.companies[ticker], day) # compute percentage change for stock
            b_h_day_return += percentage_change
            companies_count += 1

        b_h_day_return /= companies_count
        return b_h_day_return, company_probs

    # calculate daily return for stocks whose probabilities of rising are above threshold
    def simulate_trading_when_prob_above_threshold(self, num_days, day, company_probs, threshold):
        day_return, count = 0,0
        for stock in company_probs: 
            company = self.companies[stock]
            
            if len(company.test_index_list) < num_days: # ensure we have enough days of data to simulate
                continue
            
            if company_probs[stock] >= threshold:    
                percentage_change = self.compute_return(self.companies[stock], day)
                day_return += percentage_change

                count += 1

        day_return /= count
        return day_return

    # calculate daily return for trading only the X highest-probability stocks
    def simulate_trading_highest_ranked_stocks(self, num_days, day, company_probs, num_stocks_to_rank):
        day_return = 0
        companies_count = 0
        
        top_ranked_stocks = self.get_highest_ranked_stocks(company_probs, num_stocks_to_rank)

        for stock in top_ranked_stocks:
            company = self.companies[stock]
            
            # ensure we have enough days of data for the company to include company in simulation
            if len(company.test_index_list) < num_days: 
                continue

            companies_count += 1
            percentage_change = self.compute_return(company, day)
            day_return += percentage_change
        
        day_return /= companies_count
        return day_return

    # input: company object, number of days from last day
    # output: percentage change
    def compute_return(self, company, day):
        df = company.stock_hist
        index = company.test_index_list[-1*day]
        prev_close = df.iloc[index-1]["Close"]
        close = df.iloc[index]["Close"]

        percentage_change = (close-prev_close)/prev_close
        return percentage_change

    
    # simulate various trading strategies for the past X number of days 
    # strategies include buy and hold, trading when probability of rising is above threshold, and trading top X highest-probability stocks
    # input: number of trading days to simulate
    # output: percentage returns for each strategy in the time period
    def simulate(self, num_days, num_stocks_to_ranks, threshold):

        buy_and_hold_return, trading_threshold_return, best_stocks_return = 1, 1, 1

        # compute returns for all trading days in test set
        for day in range(num_days, 1, -1):

            # compute daily return for different strategies
            b_h_day_return, company_probs = self.simulate_buy_and_hold(num_days, day)
            highest_prob_day_return = self.simulate_trading_highest_ranked_stocks(num_days, day, company_probs, num_stocks_to_ranks)
            trading_threshold_day_return = self.simulate_trading_when_prob_above_threshold(num_days, day, company_probs, threshold)

            # update total return
            buy_and_hold_return *= (1+b_h_day_return)
            trading_threshold_return *= (1+trading_threshold_day_return)
            best_stocks_return *= (1+highest_prob_day_return)
        
        # convert to percentage returns
        trading_threshold_percent_return = 100 * (trading_threshold_return-1)
        best_stock_percent_return = 100 * (best_stocks_return-1)
        bh_percent_return = 100 * (buy_and_hold_return-1)
        
        print("Return for trading stocks above threshold: " + str(round(trading_threshold_percent_return, 2)) + "%")
        print("Return for trading top ranked stocks: " + str(round(best_stock_percent_return, 2)) + "%")
        print("Return for buy and hold strategy: " + str(round(bh_percent_return, 2)) + "%")
        
        return trading_threshold_percent_return, best_stock_percent_return, bh_percent_return 
