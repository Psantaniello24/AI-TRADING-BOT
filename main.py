import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from data_fetcher import DataFetcher
from lstm_model import StockPredictor
from trading_env import TradingEnvironment
from rl_agent import TradingRLAgent

def train_lstm_model(ticker, period="2y", save_path="./models"):
    """
    Train LSTM model for price prediction
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period
        save_path (str): Path to save the trained model
        
    Returns:
        StockPredictor: Trained model
    """
    print(f"Fetching data for {ticker}...")
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_data(ticker, period=period)
    
    print("Preparing data for LSTM training...")
    X_train, y_train, X_test, y_test = data_fetcher.prepare_data(df)
    
    print("Training LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = StockPredictor(input_shape)
    history = model.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_file = f"{save_path}/lstm_{ticker.lower()}.h5"
    model.save_model(model_file)
    print(f"LSTM model saved to {model_file}")
    
    # Evaluate model
    predicted = model.predict(X_test)
    mse = np.mean(np.square(y_test - predicted))
    print(f"Test MSE: {mse:.6f}")
    
    return model

def train_rl_agent(ticker, period="2y", initial_balance=10000, save_path="./models", timesteps=10000):
    """
    Train RL agent for trading
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period
        initial_balance (float): Initial account balance
        save_path (str): Path to save the trained agent
        timesteps (int): Training timesteps
        
    Returns:
        TradingRLAgent: Trained agent
    """
    print(f"Fetching data for {ticker}...")
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_data(ticker, period=period)
    
    print("Preparing data with technical indicators...")
    df_features = data_fetcher.prepare_features(df)
    
    print("Creating trading environment...")
    env = TradingEnvironment(df_features, initial_balance=initial_balance)
    
    print("Initializing RL agent...")
    agent = TradingRLAgent(env, model_name=f"ppo_{ticker.lower()}", save_path=save_path)
    
    print(f"Training RL agent for {timesteps} timesteps...")
    agent.train(total_timesteps=timesteps, eval_freq=1000)
    
    return agent

def backtest_strategy(ticker, period="2y", initial_balance=10000, model_path=None):
    """
    Backtest the trading strategy
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period
        initial_balance (float): Initial account balance
        model_path (str): Path to the trained RL model
        
    Returns:
        dict: Backtest results
    """
    print(f"Fetching data for {ticker}...")
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_data(ticker, period=period)
    
    print("Preparing data with technical indicators...")
    df_features = data_fetcher.prepare_features(df)
    
    print("Creating trading environment for backtesting...")
    env = TradingEnvironment(df_features, initial_balance=initial_balance)
    
    print("Initializing agent for backtesting...")
    agent = TradingRLAgent(env, model_name=f"ppo_{ticker.lower()}")
    
    if model_path:
        print(f"Loading trained model from {model_path}...")
        agent.load(model_path)
    
    print("Running backtest...")
    rewards = agent.test(env)
    
    # Get final info
    final_portfolio = env.balance + env.shares_held * env.current_price
    roi = (final_portfolio - initial_balance) / initial_balance * 100
    
    # Calculate Sharpe ratio
    if len(env.returns) > 1:
        sharpe_ratio = np.mean(env.returns) / np.std(env.returns) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    print(f"Backtest Results:")
    print(f"  Initial Balance: ${initial_balance:.2f}")
    print(f"  Final Portfolio: ${final_portfolio:.2f}")
    print(f"  Return: {roi:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {env.max_drawdown*100:.2f}%")
    
    results = {
        'initial_balance': initial_balance,
        'final_portfolio': final_portfolio,
        'roi': roi,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': env.max_drawdown,
        'trades': env.trades
    }
    
    return results

def run_streamlit_dashboard():
    """
    Launch the Streamlit dashboard
    """
    print("Launching Streamlit dashboard...")
    os.system("streamlit run dashboard.py")

def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--mode", choices=["train_lstm", "train_rl", "backtest", "dashboard"], 
                        default="dashboard", help="Operation mode")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="2y", help="Data period")
    parser.add_argument("--balance", type=float, default=10000, help="Initial account balance")
    parser.add_argument("--timesteps", type=int, default=10000, help="RL training timesteps")
    parser.add_argument("--model", type=str, help="Path to trained model for backtesting")
    
    args = parser.parse_args()
    
    if args.mode == "train_lstm":
        train_lstm_model(args.ticker, args.period)
    elif args.mode == "train_rl":
        train_rl_agent(args.ticker, args.period, args.balance, timesteps=args.timesteps)
    elif args.mode == "backtest":
        backtest_strategy(args.ticker, args.period, args.balance, args.model)
    elif args.mode == "dashboard":
        run_streamlit_dashboard()
    else:
        print("Invalid mode selection")

if __name__ == "__main__":
    main() 