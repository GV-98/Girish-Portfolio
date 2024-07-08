"""

BUS 888 G200: Machine Learning, taught by Chaitanya 'CK' Kaligotla

Group Project Deliverable:

'Finding Optimal Trading Strategies:
Clustering + Neural Network Algorithms'

Submitted by:
    
    Student Name          ID #
  -------------------------------------  
    Girish Venkateswaran  301604831
    Muskan Malik          301585994
    Tarush Narang         301595982
    Kusum Matlani         301593345
  -------------------------------------  
  
Sources used:

    1. Professor CK's in-class code and learning materials
    2. ChatGPT for concept understanding and examples
    3. Wall Street Journal/Bloomberg for 1-Year Treasury Rate (Risk-Free Rate)

Honor Pledge -

“ I pledge on my honor that I have neither received nor given unauthorized n/
assistance on this deliverable.”

Thesis of the group project:
    
    "
    We believe, as analysts in the SIAS Fund Global Equities Asset Class, that the United States n/
    hosts a large proportion of the companies we diversify the SIAS Portfolio in. The S&P 500 is a n/
    basket of some of the best performers in the U.S market with sizeable market caps. These innovative n/
    companies often perform well individually but seem to perform stellarly as a grouped and weighted n/
    market portfolio (SPY). 
    
    Our goal, as a group, is to pick out one optimally and sub-optimally performing stock in the 
    S&P 500 and create a trading strategy for the pair in tandem and, hopefully, try to beat 
    the SPY itself. 
    
    We aim to use a rudimentary short strategy for the poor performer, and compare this approach
    with a mean reversion and an all-in trading approach.
    For the good performer, we prefer to use a simple buy-and-hold strategy and compare this to 
    all-in and mean reversion strategies. 
    
    For both stocks and their strategies, we will be using a neural network with 3 hidden layers
    and backtest using 90 day time periods (we are using days as the chosen interval of time).
    
    While our code does not strive to be perfect, we shall try our best to help bring our thesis 
    into fruition. 
    
    "

"""

"""
    
    
    Part 1: Using unsupervised learning (K-Means Clustering) to pick and consistently 
    select ever-changing stocks as our best and worst-performers. 
    Going back 2 years in the past for data collection.
    A K-Means Cluster Algo would, over time, pick different best and worst stocks as time
    progresses on.
    

"""   

#Importing all the libraries we need

import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

from pylab import plot,show
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense


import warnings
warnings.filterwarnings('ignore')

#Utilizing Wikipedia to scrape S&P 500 ticker names straight from the source

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

#Reading in the data and scraping ticker names
data_table = pd.read_html(sp500_url)
tickers = data_table[0]['Symbol'].values.tolist()
tickers = [s.replace('\n', '') for s in tickers]
tickers = [s.replace('.', '-') for s in tickers]
tickers = [s.replace(' ', '') for s in tickers]
print(tickers)

#Testing Yahoo Call for our learning purposes
#p = yf.Ticker('NVDA')
#p2 = p.history(start='2022-07-01',end='2024-07-01', auto_adjust=True, actions=True)['Close']
#print(p2.head())   

                    #Using auto_adjust=True, actions'True because yfinance does not use n/
                    #adjusted close prices

# Now we shall download our prices data for the S&P 500 tickers for 2 years, upto July 1 2024
prices_list = []
for ticker in tickers:
    try:
        data = yf.Ticker(ticker)
        prices = data.history(start='2022-07-01',end='2024-07-01', auto_adjust=True, actions=True)['Close']
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        prices_list.append(prices)
    except:
        pass
    prices_df = pd.concat(prices_list,axis=1)
prices_df.sort_index(inplace=True)

prices_df

# Create an empty dataframe for raw returns
returns = pd.DataFrame()

# Calculate annualized average daily returns for each ticker for each trading day
returns['Returns'] = prices_df.pct_change().mean() * 252
  # Note:
    # prices_df.pct_change(): Calculates the percentage change between consecutive prices in the prices_df DataFrame, giving you daily returns.
    #.mean(): Calculates the average of the daily returns.
    # * 252: Annualizes the average daily return by multiplying it by the approximate number of trading days in a year.

# Calculate annualized average volatility of daily return for each ticker for each trading day
returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)

returns

returns.loc['NVDA'] #Another sanity check!

"""We wish to use K-Means Clustering to segregate the stocks based on shared features"""

# Format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
print(X)

"""Identifying ideal number of clusters"""

sse = [] # also called distortions
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 20), sse)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

print(sse)

"""Based on the elbow method graph, we will be utilizing K = 7 clusters"""

K = 7  

kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X)

"""### Compute Centroids and prepare output"""

centroids = kmeans.cluster_centers_

# Assign each sample to a cluster
idx,_ = vq(data,centroids)

# Create a dataframe with the tickers and the clusters that's belong to
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
details_df = pd.DataFrame(details)

# Renaming columns now
details_df.columns = ['Ticker','Cluster']

# Create another dataframe with the tickers and data from each stock
clusters_df = returns.reset_index()

# Bring the clusters information from the dataframe 'details_df'
clusters_df['Cluster'] = details_df['Cluster']

# Create a DataFrame for centroids
centroids_df = pd.DataFrame(centroids, columns=['Returns', 'Volatility'])
centroids_df['Cluster'] = range(K)  # Assign a cluster label to each centroid

# Rename columns
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']

# See the table output of clustering tickers
clusters_df
# To see a specific cluster use:clusters_df[clusters_df['Cluster']==1]

"""Plotting our clusters"""

# Plot the clusters created using Plotly
fig = px.scatter(clusters_df, x="Returns", y="Volatility", color="Cluster", hover_data=["Ticker"])
#Add centroids to the scatter plot
fig.add_trace(go.Scatter(
    x=centroids_df['Returns'],
    y=centroids_df['Volatility'],
    mode='markers+text',
    marker=dict(color='black', size=15, symbol='x'),
    text=centroids_df['Cluster'],
    textposition='top center',
    name='Centroids'
))
#fig.update(layout_coloraxis_showscale=False)
fig.update_layout(coloraxis_showscale=False, height=800) # Update the layout to set height
fig.show()

#As a side note, our code works well here but the visualization seems to work best on n/
#Google Colab

#Testing out contents of certain clusters below!

clusters_df[clusters_df['Cluster']==2]

clusters_df[clusters_df['Cluster']==6]

"""We will now compute cluster statistics and display as a table"""

# Group the DataFrame by 'Cluster' and aggregate the desired statistics
cluster_summary = clusters_df.groupby('Cluster').agg(
    Avg_Return=('Returns', 'mean'),
    Avg_Volatility=('Volatility', 'mean'),
    Num_Tickers=('Ticker', 'count')
).reset_index()

# Calculate the return to volatility ratio
cluster_summary['Return_Volatility_Ratio'] = cluster_summary['Avg_Return'] / cluster_summary['Avg_Volatility']

# Calculate Sharpe Ratio for Each Cluster
risk_free_rate = 0.05113
cluster_summary['Sharpe_Ratio'] = (cluster_summary['Avg_Return'] - risk_free_rate) / cluster_summary['Avg_Volatility']

# Print the list of tickers per cluster
for cluster in clusters_df['Cluster'].unique():
    print(f"\nTickers in Cluster {cluster}:")
    print(clusters_df[clusters_df['Cluster'] == cluster]['Ticker'].tolist())

# Print the summary table
cluster_summary

"""Now that we have our clusters, we wish to choose the stocks with the best sharpe n/
  ratios in the best AND worst clusters! Find a sub-optimal and optimal stock this way!"""

# Step 1: Calculate Sharpe Ratio for Each Stock
risk_free_rate = 0.05113
clusters_df['Sharpe_Ratio'] = (clusters_df['Returns'] - risk_free_rate) / clusters_df['Volatility']

# Step 2: Aggregate Sharpe Ratios by Cluster
cluster_sharpe_ratios = clusters_df.groupby('Cluster')['Sharpe_Ratio'].mean().reset_index()
cluster_sharpe_ratios.columns = ['Cluster', 'Avg_Sharpe_Ratio']

# Step 3: Identify Best and Worst Clusters
best_cluster = cluster_sharpe_ratios.loc[cluster_sharpe_ratios['Avg_Sharpe_Ratio'].idxmax()]['Cluster']
worst_cluster = cluster_sharpe_ratios.loc[cluster_sharpe_ratios['Avg_Sharpe_Ratio'].idxmin()]['Cluster']

# Step 4: Find Best Stock in Best Cluster
best_stock_in_best_cluster = clusters_df[clusters_df['Cluster'] == best_cluster].sort_values(by='Sharpe_Ratio', ascending=False).iloc[0]
best_stock_ticker = best_stock_in_best_cluster['Ticker']

"""
Variations Below:
    
# Step 5: Find Worst Stock in best Cluster
#worst_stock_in_best_cluster = clusters_df[clusters_df['Cluster'] == best_cluster].sort_values(by='Sharpe_Ratio').iloc[0]
#worst_stock_ticker = worst_stock_in_best_cluster['Ticker']

# Step 5: Find Worst Stock in Worst Cluster
#worst_stock_in_worst_cluster = clusters_df[clusters_df['Cluster'] == worst_cluster].sort_values(by='Sharpe_Ratio').iloc[0]
#worst_stock_ticker = worst_stock_in_worst_cluster['Ticker']
"""

# Step 5: Find Best Stock in Worst Cluster
best_stock_in_worst_cluster = clusters_df[clusters_df['Cluster'] == worst_cluster].sort_values(by='Sharpe_Ratio', ascending=False).iloc[0]
worst_stock_ticker = best_stock_in_worst_cluster['Ticker']

# Output the results
print(f"Best stock in the best cluster: {best_stock_ticker}")
print(f"Best stock in the worst cluster: {worst_stock_ticker}")

#Testing once again!

best_cluster

worst_cluster

best_stock_ticker

worst_stock_ticker

"""
    
    
    Part 2: Using neural network, training the model, and evaluating it using backtesting 
    of 90 days. We shall also evaluate our trading strategies for each stock this way.
    

"""   

# Constants
TICKER = best_stock_ticker #Use this for best stock trading strategies

#TICKER = worst_stock_ticker #Use this for worst stock trading strategies

START_DATE = '2022-07-01' #Using 2 years worth of data!
END_DATE = '2024-07-01'
LOOK_BACK = 90
EPOCHS = 64
BATCH_SIZE = 16

RISK_FREE_RATE = 0.05113 #Risk-Free Rate = 1-Year U.S Treasury Rate

INITIAL_BALANCE = 5000  # Initial balance in dollars
TRANSACTION_COST = 2  # Cost per transaction in dollars
MAX_BUY_LIMIT = 50  # Maximum number of stocks to buy per transaction

# Functions for data fetching and preparation
def fetch_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
    return yf.download(ticker, start=start_date, end=end_date)

def prepare_data_without_scaling(data, look_back):
    close_prices = data['Close'].values.reshape(-1, 1)
    X, y = [], []
    for i in range(look_back, len(close_prices)):
        X.append(close_prices[i - look_back:i, 0])
        y.append(close_prices[i, 0])
    return np.array(X), np.array(y)

#Splitting the data into a train-test split of 80-20
def split_data(X, y, ratio=0.8):
    train_size = int(len(X) * ratio)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

#Building our neural network model using ReLu activation functions
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#Using a rolling window backtesting method of 90 days to evaluate our model and strategies!
def rolling_prediction(model, X_test, look_back, steps_ahead=1):
    test_predictions = []
    for i in range(0, len(X_test) - steps_ahead + 1):
        current_batch = X_test[i].reshape(1, look_back)
        for j in range(steps_ahead):
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.roll(current_batch, -1)
            if j < steps_ahead - 1:
                current_batch[-1][-1] = current_pred
    return test_predictions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Use this segment of the code, left as a large comment block, for the worst stock ticker
#trading strategy coding and implementation.

#Make sure to set TICKER = worst_stock_ticker in the constants section above!
    
#shorting strategy, for worst stock trading
    
def trading_strategy1(y_test_original, test_predictions_original): 
        
    balance = INITIAL_BALANCE
    stock_count = 0
    account_values = [INITIAL_BALANCE]
    transaction_cost = TRANSACTION_COST
    short_sell_points, cover_points = [], []
    
    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]
        
        # Short sell if the price is expected to drop
        if predicted_next_day_price < today_price and stock_count == 0:
            stock_count += 1
            balance += today_price - transaction_cost
            short_sell_points.append((i, today_price))
        
        # Cover the short if the price is expected to rise
        elif predicted_next_day_price > today_price and stock_count > 0:
            stock_count -= 1
            balance -= today_price + transaction_cost
            cover_points.append((i, today_price))
        
        # Update account value
        account_value = balance + stock_count * today_price
        account_values.append(account_value)
    
    # Final cover if still shorting at the end
    if stock_count > 0:
        balance -= stock_count * y_test_original[-1][0] + transaction_cost
        stock_count = 0
    
    account_values.append(balance)
    
    return balance, short_sell_points, cover_points, account_values
    

#Below is the all-in buying/selling strategy commmon for both stocks

def trading_strategy2(y_test_original, test_predictions_original):
    balance = INITIAL_BALANCE
    transaction_cost = TRANSACTION_COST
    max_buy_limit = MAX_BUY_LIMIT
    stock_count = 0
    buy_points, sell_points, account_values = [], [], []
    
    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]
        account_value = balance + stock_count * today_price
        account_values.append(account_value)
        max_stocks_to_buy = min(max_buy_limit, balance // (today_price + transaction_cost))
        if predicted_next_day_price > today_price and max_stocks_to_buy > 0:
            stock_count += max_stocks_to_buy
            balance -= (today_price + transaction_cost) * max_stocks_to_buy
            buy_points.append((i, today_price, max_stocks_to_buy))
        elif predicted_next_day_price < today_price and stock_count > 0:
            balance += (today_price - transaction_cost) * stock_count
            sell_points.append((i, today_price, stock_count))
            stock_count = 0
    if stock_count > 0:
        balance += y_test_original[-1][0] * stock_count
        stock_count = 0
    account_values.append(balance)
    return balance, buy_points, sell_points, account_values

#Mean reversion strategy common to both stocks, below

def mean_reversion_strategy(y_test_original, test_predictions_original, threshold=0.05):
    balance = INITIAL_BALANCE
    stock_count = 0
    account_values = [INITIAL_BALANCE]
    buy_points, sell_points = [], []
    mean_price = np.mean(y_test_original)

    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]

        account_value = balance + stock_count * today_price
        account_values.append(account_value)

        if predicted_next_day_price < mean_price * (1 - threshold) and stock_count == 0:
            stock_count += 1
            balance -= today_price
            buy_points.append((i, today_price))
        elif predicted_next_day_price > mean_price * (1 + threshold) and stock_count > 0:
            stock_count -= 1
            balance += today_price
            sell_points.append((i, today_price))

    if stock_count >= 1:
        balance += stock_count * y_test_original[-1][0]
        stock_count = 0
    account_values.append(balance)

    return balance, buy_points, sell_points, account_values

#This function outputs our desired statistics for each strategy for each stock combination
def calculate_performance(account_values, risk_free_rate=0.05113):
    total_return = (account_values[-1] - account_values[0]) / account_values[0] * 100
    num_years = (len(account_values) / 252)
    annualized_return = (account_values[-1] / account_values[0])**(1 / num_years) - 1
    cagr = (account_values[-1] / account_values[0])**(1 / num_years) - 1
    running_max = np.maximum.accumulate(account_values)
    drawdown = (account_values - running_max) / running_max
    max_drawdown = drawdown.min()
    daily_returns = np.diff(account_values) / account_values[:-1]
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }

#Now we need to bring in SPY data to compare our strategies with, as a benchmark

def fetch_benchmark_data(ticker="SPY", start_date=START_DATE, end_date=END_DATE):
    return yf.download(ticker, start=start_date, end=end_date)

def keep_last_values(array, num_values=133):
    return array[-num_values:] if len(array) >= num_values else array

def calculate_benchmark_performance(close_prices, initial_balance=INITIAL_BALANCE):
    account_values = [initial_balance * (price / close_prices[0]) for price in close_prices]
    return calculate_performance(account_values)

# Fetch and prepare data for the neural network model
data = fetch_stock_data()
X, y = prepare_data_without_scaling(data, LOOK_BACK)
X_train, X_test, y_train, y_test = split_data(X, y)

# Scaling the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Train the neural network model
model = build_model(LOOK_BACK)
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Test the model
test_predictions = rolling_prediction(model, X_test, LOOK_BACK)
test_predictions_original = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test)

# Implement trading strategies

#Use below line for worst stock (shorting)
balance1, short_sell_points1, cover_points1, account_values1 = trading_strategy1(y_test_original, test_predictions_original)
balance2, buy_points2, sell_points2, account_values2 = trading_strategy2(y_test_original, test_predictions_original)

#balance3, buy_points3, sell_points3, account_values3 = moving_average_crossover_strategy(y_test_original, test_predictions_original)

balance3, buy_points3, sell_points3, account_values3 = mean_reversion_strategy(y_test_original, test_predictions_original)

# Calculate performance for all strategies
account_values1_last = keep_last_values(account_values1)
account_values2_last = keep_last_values(account_values2)

#account_values3_last = keep_last_values(account_values3)

account_values3_last = keep_last_values(account_values3)

performance1 = calculate_performance(account_values1_last)
performance2 = calculate_performance(account_values2_last)

#performance3 = calculate_performance(account_values3_last)

performance3 = calculate_performance(account_values3_last)

# Fetch SPY data and calculate benchmark performance
spy_data = fetch_benchmark_data()
spy_close_prices = spy_data['Close'].values
spy_close_prices_split = keep_last_values(spy_close_prices, len(y_test_original))

benchmark_performance = calculate_benchmark_performance(spy_close_prices_split)
spy_account_values_last = keep_last_values([INITIAL_BALANCE * (price / spy_close_prices[0]) for price in spy_close_prices], 133)

# Extract dates for the test set
test_dates = data.index[-len(y_test):]
dates_last = test_dates[-133:]

# Print results
print("Strategy 1 Performance:")
for metric, value in performance1.items():
    print(f"{metric}: {value:.2f}%")

print("\nStrategy 2 Performance:")
for metric, value in performance2.items():
    print(f"{metric}: {value:.2f}%")
    
"""

#Just left here as a comment!
#print("Strategy 3 Performance:")
#for metric, value in performance3.items():
   # print(f"{metric}: {value:.2f}%")
    
"""

print("\nStrategy 3 Performance:")
for metric, value in performance3.items():
    print(f"{metric}: {value:.2f}%")

print("\nBenchmark (SPY) Performance:")
for metric, value in benchmark_performance.items():
    print(f"{metric}: {value:.2f}%")

# Compare the performance of all strategies with the benchmark
def compare_performance(strategy_performance, benchmark_performance):
    for metric in strategy_performance:
        if strategy_performance[metric] > benchmark_performance[metric]:
            print(f"Strategy outperformed the benchmark in terms of {metric}.")
        else:
            print(f"Strategy underperformed the benchmark in terms of {metric}.")

print("\nComparing Strategy 1 with SPY:")
compare_performance(performance1, benchmark_performance)

print("\nComparing Strategy 2 with SPY:")
compare_performance(performance2, benchmark_performance)

"""

# Compare the performance of all strategies with the benchmark
#print("\nComparing Strategy 3 with SPY:")
#compare_performance(performance3, benchmark_performance)

"""

print("\nComparing Strategy 3 with SPY:")
compare_performance(performance3, benchmark_performance)

## PLOTTING INDIVIDUAL STRATEGIES

def plot_trading_strategy(strategy_num, y_test_original, test_predictions_original, buy_points, sell_points, account_values):
    # Extract buy and sell points
    if buy_points:
        buy_indices, buy_prices, *_ = zip(*buy_points)
    else:
        buy_indices, buy_prices, *_ = [], []

    if sell_points:
        sell_indices, sell_prices, *_ = zip(*sell_points)
    else:
        sell_indices, sell_prices, *_ = [], []

    plt.figure(figsize=(15, 6))
    ax1 = plt.gca()  # get current axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax1.plot(y_test_original, label="Actual Prices", color="blue")
    ax1.plot(test_predictions_original, label="Predicted Prices", color="red", alpha=0.7)
    ax1.scatter(buy_indices, buy_prices, color="green", marker="^", alpha=1, label="Buy (Short) Points", s=100)
    ax1.scatter(sell_indices, sell_prices, color="red", marker="v", alpha=1, label="Sell (Cover) Points", s=100)
    ax1.set_title(f"Trading Strategy {strategy_num} on Test Data")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Stock Price")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(account_values, label="Account Value", color="green", linestyle='--')
    ax2.set_ylabel("Account Value", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.legend(loc='upper right')

    plt.show()

#Short_sell and cover_points for worst stock, buy and sell_points for best stock

strategies = [
    (1, y_test_original, test_predictions_original, short_sell_points1, cover_points1, account_values1),
    (2, y_test_original, test_predictions_original, buy_points2, sell_points2, account_values2),
    #(3, y_test_original, test_predictions_original, buy_points3, sell_points3, account_values3),
    (3, y_test_original, test_predictions_original, buy_points3, sell_points3, account_values3)
 ]

for strategy in strategies:
    plot_trading_strategy(*strategy)
    

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Use this segment of the code, left as a large comment block, for the best stock ticker
#trading strategy coding and implementation.

#Make sure to set TICKER = best_stock_ticker in the constants section above!

#Using the buy-and-hold strategy below for the best stock

def trading_strategy1(y_test_original, test_predictions_original):
    balance = INITIAL_BALANCE
    stock_count = 0
    account_values = [INITIAL_BALANCE]
    buy_points, sell_points = [],[]
    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]
        account_value = balance + stock_count * today_price
        account_values.append(account_value)
        if predicted_next_day_price > today_price and balance > today_price and stock_count == 0:
            stock_count += 1
            balance -= today_price
            buy_points.append((i, today_price))
        elif predicted_next_day_price < today_price and stock_count > 0:
            stock_count -= 1
            balance += today_price
            sell_points.append((i, today_price))
    if stock_count >= 1:
        balance += stock_count * y_test_original[-1][0]
        stock_count = 0
    account_values.append(balance)
    return balance, buy_points, sell_points, account_values

#Below is the all-in buying/selling strategy commmon for both stocks

def trading_strategy2(y_test_original, test_predictions_original):
    balance = INITIAL_BALANCE
    transaction_cost = TRANSACTION_COST
    max_buy_limit = MAX_BUY_LIMIT
    stock_count = 0
    buy_points, sell_points, account_values = [], [], []
    
    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]
        account_value = balance + stock_count * today_price
        account_values.append(account_value)
        max_stocks_to_buy = min(max_buy_limit, balance // (today_price + transaction_cost))
        if predicted_next_day_price > today_price and max_stocks_to_buy > 0:
            stock_count += max_stocks_to_buy
            balance -= (today_price + transaction_cost) * max_stocks_to_buy
            buy_points.append((i, today_price, max_stocks_to_buy))
        elif predicted_next_day_price < today_price and stock_count > 0:
            balance += (today_price - transaction_cost) * stock_count
            sell_points.append((i, today_price, stock_count))
            stock_count = 0
    if stock_count > 0:
        balance += y_test_original[-1][0] * stock_count
        stock_count = 0
    account_values.append(balance)
    return balance, buy_points, sell_points, account_values


#Mean reversion strategy common to both stocks, below

def mean_reversion_strategy(y_test_original, test_predictions_original, threshold=0.05):
    balance = INITIAL_BALANCE
    stock_count = 0
    account_values = [INITIAL_BALANCE]
    buy_points, sell_points = [], []
    mean_price = np.mean(y_test_original)

    for i in range(len(y_test_original) - 1):
        today_price = y_test_original[i][0]
        predicted_next_day_price = test_predictions_original[i + 1][0]

        account_value = balance + stock_count * today_price
        account_values.append(account_value)

        if predicted_next_day_price < mean_price * (1 - threshold) and stock_count == 0:
            stock_count += 1
            balance -= today_price
            buy_points.append((i, today_price))
        elif predicted_next_day_price > mean_price * (1 + threshold) and stock_count > 0:
            stock_count -= 1
            balance += today_price
            sell_points.append((i, today_price))

    if stock_count >= 1:
        balance += stock_count * y_test_original[-1][0]
        stock_count = 0
    account_values.append(balance)

    return balance, buy_points, sell_points, account_values

#This function outputs our desired statistics for each strategy for each stock combination
def calculate_performance(account_values, risk_free_rate=0.05113):
    total_return = (account_values[-1] - account_values[0]) / account_values[0] * 100
    num_years = (len(account_values) / 252)
    annualized_return = (account_values[-1] / account_values[0])**(1 / num_years) - 1
    cagr = (account_values[-1] / account_values[0])**(1 / num_years) - 1
    running_max = np.maximum.accumulate(account_values)
    drawdown = (account_values - running_max) / running_max
    max_drawdown = drawdown.min()
    daily_returns = np.diff(account_values) / account_values[:-1]
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }

#Now we need to bring in SPY data to compare our strategies with, as a benchmark

def fetch_benchmark_data(ticker="SPY", start_date=START_DATE, end_date=END_DATE):
    return yf.download(ticker, start=start_date, end=end_date)

def keep_last_values(array, num_values=133):
    return array[-num_values:] if len(array) >= num_values else array

def calculate_benchmark_performance(close_prices, initial_balance=INITIAL_BALANCE):
    account_values = [initial_balance * (price / close_prices[0]) for price in close_prices]
    return calculate_performance(account_values)

# Fetch and prepare data for the neural network model
data = fetch_stock_data()
X, y = prepare_data_without_scaling(data, LOOK_BACK)
X_train, X_test, y_train, y_test = split_data(X, y)

# Scaling the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Train the neural network model
model = build_model(LOOK_BACK)
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Test the model
test_predictions = rolling_prediction(model, X_test, LOOK_BACK)
test_predictions_original = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test)

# Implement trading strategies

#use below for best stock(buy-and-hold)
balance1, buy_points1, sell_points1, account_values1 = trading_strategy1(y_test_original, test_predictions_original)
balance2, buy_points2, sell_points2, account_values2 = trading_strategy2(y_test_original, test_predictions_original)

#balance3, buy_points3, sell_points3, account_values3 = moving_average_crossover_strategy(y_test_original, test_predictions_original)

balance3, buy_points3, sell_points3, account_values3 = mean_reversion_strategy(y_test_original, test_predictions_original)

# Calculate performance for all strategies
account_values1_last = keep_last_values(account_values1)
account_values2_last = keep_last_values(account_values2)

#account_values3_last = keep_last_values(account_values3)

account_values3_last = keep_last_values(account_values3)

performance1 = calculate_performance(account_values1_last)
performance2 = calculate_performance(account_values2_last)

#performance3 = calculate_performance(account_values3_last)

performance3 = calculate_performance(account_values3_last)

# Fetch SPY data and calculate benchmark performance
spy_data = fetch_benchmark_data()
spy_close_prices = spy_data['Close'].values
spy_close_prices_split = keep_last_values(spy_close_prices, len(y_test_original))

benchmark_performance = calculate_benchmark_performance(spy_close_prices_split)
spy_account_values_last = keep_last_values([INITIAL_BALANCE * (price / spy_close_prices[0]) for price in spy_close_prices], 133)

# Extract dates for the test set
test_dates = data.index[-len(y_test):]
dates_last = test_dates[-133:]

# Print results
print("Strategy 1 Performance:")
for metric, value in performance1.items():
    print(f"{metric}: {value:.2f}%")

print("\nStrategy 2 Performance:")
for metric, value in performance2.items():
    print(f"{metric}: {value:.2f}%")
    
print("\nStrategy 3 Performance:")
for metric, value in performance3.items():
    print(f"{metric}: {value:.2f}%")

print("\nBenchmark (SPY) Performance:")
for metric, value in benchmark_performance.items():
    print(f"{metric}: {value:.2f}%")

# Compare the performance of all strategies with the benchmark
def compare_performance(strategy_performance, benchmark_performance):
    for metric in strategy_performance:
        if strategy_performance[metric] > benchmark_performance[metric]:
            print(f"Strategy outperformed the benchmark in terms of {metric}.")
        else:
            print(f"Strategy underperformed the benchmark in terms of {metric}.")

print("\nComparing Strategy 1 with SPY:")
compare_performance(performance1, benchmark_performance)

print("\nComparing Strategy 2 with SPY:")
compare_performance(performance2, benchmark_performance)

print("\nComparing Strategy 3 with SPY:")
compare_performance(performance3, benchmark_performance)

## PLOTTING INDIVIDUAL STRATEGIES

def plot_trading_strategy(strategy_num, y_test_original, test_predictions_original, buy_points, sell_points, account_values):
    # Extract buy and sell points
    if buy_points:
        buy_indices, buy_prices, *_ = zip(*buy_points)
    else:
        buy_indices, buy_prices, *_ = [], []

    if sell_points:
        sell_indices, sell_prices, *_ = zip(*sell_points)
    else:
        sell_indices, sell_prices, *_ = [], []

    plt.figure(figsize=(15, 6))
    ax1 = plt.gca()  # get current axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax1.plot(y_test_original, label="Actual Prices", color="blue")
    ax1.plot(test_predictions_original, label="Predicted Prices", color="red", alpha=0.7)
    ax1.scatter(buy_indices, buy_prices, color="green", marker="^", alpha=1, label="Buy (Short) Points", s=100)
    ax1.scatter(sell_indices, sell_prices, color="red", marker="v", alpha=1, label="Sell (Cover) Points", s=100)
    ax1.set_title(f"Trading Strategy {strategy_num} on Test Data")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Stock Price")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(account_values, label="Account Value", color="green", linestyle='--')
    ax2.set_ylabel("Account Value", color="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.legend(loc='upper right')

    plt.show()

#Short_sell and cover_points for worst stock, buy and sell_points for best stock

strategies = [
    (1, y_test_original, test_predictions_original, buy_points1, sell_points1, account_values1),
    (2, y_test_original, test_predictions_original, buy_points2, sell_points2, account_values2),
    #(3, y_test_original, test_predictions_original, buy_points3, sell_points3, account_values3),
    (3, y_test_original, test_predictions_original, buy_points3, sell_points3, account_values3)
 ]

for strategy in strategies:
    plot_trading_strategy(*strategy)
    

