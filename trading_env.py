import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    """Custom Gymnasium trading environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        """
        Initialize the trading environment
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            initial_balance (float): Initial account balance
            transaction_fee_percent (float): Fee percentage for each transaction
        """
        super(TradingEnvironment, self).__init__()
        
        # Data
        self.df = df
        self.features = self._prepare_features(df)
        self.reward_range = (-np.inf, np.inf)
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: stock market features
        # Let's use normalized stock data features as our observation space
        num_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        # Trading settings
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth_history = []
        self.current_price = 0
        self.last_transaction_price = 0
        self.total_fees = 0
        self.trades = []
        
        # Performance tracking
        self.returns = []
        self.max_net_worth = initial_balance
        self.max_drawdown = 0
        
    def _prepare_features(self, df):
        """
        Extract and normalize features for the observation space
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            np.array: Normalized features
        """
        # Get a subset of features for the observation space
        features = df[['Close', 'Open', 'High', 'Low', 'Volume', 
                      'MA5', 'MA20', 'RSI', 'MACD', 'Signal']].values
        
        # Normalize features
        for i in range(features.shape[1]):
            col = features[:, i]
            if np.max(col) - np.min(col) > 0:
                features[:, i] = (col - np.min(col)) / (np.max(col) - np.min(col))
        
        return features
        
    def reset(self, seed=None):
        """
        Reset the environment to start a new episode
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth_history = [self.initial_balance]
        
        # Properly handle Series objects
        close_value = self.df['Close'].iloc[self.current_step]
        self.current_price = float(close_value.iloc[0]) if hasattr(close_value, 'iloc') else float(close_value)
        
        self.last_transaction_price = 0
        self.total_fees = 0
        self.trades = []
        
        # Reset performance metrics
        self.returns = []
        self.max_net_worth = self.initial_balance
        self.max_drawdown = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take (0=Hold, 1=Buy, 2=Sell)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Execute trade action
        self._take_action(action)
        
        # Move to next time step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        # Update current price - safely handle Series objects
        close_value = self.df['Close'].iloc[self.current_step]
        self.current_price = float(close_value.iloc[0]) if hasattr(close_value, 'iloc') else float(close_value)
        
        # Update net worth (balance + shares value)
        current_net_worth = self.balance + self.shares_held * self.current_price
        self.net_worth_history.append(current_net_worth)
        
        # Update max net worth and calculate drawdown
        if current_net_worth > self.max_net_worth:
            self.max_net_worth = current_net_worth
        
        drawdown = (self.max_net_worth - current_net_worth) / self.max_net_worth
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate returns if we have enough history
        if len(self.net_worth_history) > 1:
            daily_return = (self.net_worth_history[-1] / self.net_worth_history[-2]) - 1
            self.returns.append(daily_return)
        
        # Get reward
        reward = self._calculate_reward()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _take_action(self, action):
        """
        Execute the trade action
        
        Args:
            action (int): Action to take (0=Hold, 1=Buy, 2=Sell)
        """
        # Safely handle Series objects
        close_value = self.df['Close'].iloc[self.current_step]
        current_price = float(close_value.iloc[0]) if hasattr(close_value, 'iloc') else float(close_value)
        
        self.current_price = current_price
        
        if action == 1:  # Buy
            # Calculate maximum shares we can buy
            max_shares = self.balance / (current_price * (1 + self.transaction_fee_percent))
            shares_to_buy = max_shares  # Buy all shares possible
            
            # Calculate transaction fee
            transaction_fee = shares_to_buy * current_price * self.transaction_fee_percent
            
            # Update balance and shares
            self.balance -= (shares_to_buy * current_price + transaction_fee)
            self.shares_held += shares_to_buy
            self.total_fees += transaction_fee
            self.last_transaction_price = current_price
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'buy',
                'shares': shares_to_buy,
                'cost': shares_to_buy * current_price,
                'fee': transaction_fee
            })
            
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Calculate transaction fee
                transaction_fee = self.shares_held * current_price * self.transaction_fee_percent
                
                # Update balance and shares
                self.balance += (self.shares_held * current_price - transaction_fee)
                self.total_fees += transaction_fee
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'type': 'sell',
                    'shares': self.shares_held,
                    'proceeds': self.shares_held * current_price,
                    'fee': transaction_fee
                })
                
                self.shares_held = 0
                self.last_transaction_price = current_price
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            np.array: Current observation features
        """
        return self.features[self.current_step]
    
    def _get_info(self):
        """
        Get additional information about the current state
        
        Returns:
            dict: Information about the current state
        """
        current_net_worth = self.balance + self.shares_held * self.current_price
        profit_pct = ((current_net_worth - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate Sharpe ratio if we have enough returns
        sharpe_ratio = 0
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-10) * np.sqrt(252)  # Annualized
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'net_worth': current_net_worth,
            'profit_pct': profit_pct,
            'total_fees': self.total_fees,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current step
        
        Returns:
            float: Reward value
        """
        current_net_worth = self.balance + self.shares_held * self.current_price
        
        # Calculate profit component
        if len(self.net_worth_history) > 1:
            profit_reward = (current_net_worth - self.net_worth_history[-2]) / self.initial_balance
        else:
            profit_reward = 0
        
        # Calculate Sharpe ratio component (if we have enough returns)
        sharpe_reward = 0
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-10)
            sharpe_reward = sharpe_ratio
        
        # Combine rewards
        # We weight profit more in early training, then gradually increase weight of Sharpe
        sharpe_weight = min(0.5, len(self.returns) / 100)  # Max weight of 0.5
        profit_weight = 1 - sharpe_weight
        
        # Combined reward
        reward = (profit_weight * profit_reward) + (sharpe_weight * sharpe_reward)
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode (str): Rendering mode
        """
        current_net_worth = self.balance + self.shares_held * self.current_price
        profit = current_net_worth - self.initial_balance
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held:.6f}")
        print(f"Net worth: ${current_net_worth:.2f}")
        print(f"Profit: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)")
        print("-" * 50)
    
    def close(self):
        """Close the environment"""
        pass 