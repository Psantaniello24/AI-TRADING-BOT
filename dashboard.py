import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

from data_fetcher import DataFetcher
from lstm_model import StockPredictor
from trading_env import TradingEnvironment
from rl_agent import TradingRLAgent

# Helper function to safely convert values to float
def safe_float(value):
    """Safely convert a value to float, handling Series objects"""
    if hasattr(value, 'iloc'):
        return float(value.iloc[0])
    return float(value)

# Set default plotly theme
pio.templates.default = "plotly_dark"

# Set page configuration
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar for user inputs
st.sidebar.title("Trading Bot Controls")

# Stock selection
ticker_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
period = st.sidebar.selectbox(
    "Data Period",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

# Theme selection
theme = st.sidebar.selectbox(
    "Chart Theme",
    options=["plotly_dark", "plotly", "plotly_white"],
    index=0
)
pio.templates.default = theme

# Fetch data button
if st.sidebar.button("Fetch Data"):
    st.session_state.fetch_data = True
    st.session_state.ticker = ticker_symbol
    st.session_state.period = period
    
# Initialize session state variables if they don't exist
if 'fetch_data' not in st.session_state:
    st.session_state.fetch_data = False

if 'run_prediction' not in st.session_state:
    st.session_state.run_prediction = False

if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

# Main application
st.title("AI Trading Bot Dashboard")

# Instructions if no data is fetched yet
if not st.session_state.fetch_data:
    st.info("👈 Enter a stock symbol and click 'Fetch Data' to get started")

# Data fetching and display
if st.session_state.fetch_data:
    with st.spinner("Fetching stock data..."):
        # Create data fetcher
        data_fetcher = DataFetcher()
        
        # Fetch the data
        df = data_fetcher.fetch_data(st.session_state.ticker, period=st.session_state.period)
        
        # Store data in session state
        st.session_state.data = df
        
        # Add technical indicators
        st.session_state.data_with_features = data_fetcher.prepare_features(df)
        
        # Show success message
        st.success(f"Data for {st.session_state.ticker} fetched successfully!")
        
        # Reset the prediction and simulation flags
        st.session_state.run_prediction = False
        st.session_state.run_simulation = False
    
    # Display data
    st.subheader("Stock Price Data")
    
    # Create stock price chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'MA5' in st.session_state.data_with_features.columns:
        fig.add_trace(
            go.Scatter(
                x=st.session_state.data_with_features.index,
                y=st.session_state.data_with_features['MA5'],
                line=dict(color='#2196F3', width=1),
                name='5-day MA'
            ),
            row=1, col=1
        )
    
    if 'MA20' in st.session_state.data_with_features.columns:
        fig.add_trace(
            go.Scatter(
                x=st.session_state.data_with_features.index,
                y=st.session_state.data_with_features['MA20'],
                line=dict(color='#FFC107', width=1),
                name='20-day MA'
            ),
            row=1, col=1
        )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name='Volume',
            marker_color='rgba(128, 128, 255, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        template=theme,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        tickformat='%d %b %Y',
        tickangle=-45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    
    # Display the plot with a different renderer
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    # Display a sample of the data with features
    st.subheader("Technical Indicators")
    st.dataframe(st.session_state.data_with_features.tail(10))
    
    # LSTM Prediction
    st.sidebar.subheader("LSTM Price Prediction")
    prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)
    
    if st.sidebar.button("Run LSTM Prediction"):
        st.session_state.run_prediction = True
        st.session_state.prediction_days = prediction_days
    
    # RL Agent Simulation
    st.sidebar.subheader("RL Trading Simulation")
    initial_balance = st.sidebar.number_input("Initial Balance ($)", 1000, 100000, 10000, step=1000)
    
    if st.sidebar.button("Run Trading Simulation"):
        st.session_state.run_simulation = True
        st.session_state.initial_balance = initial_balance

# LSTM Prediction
if st.session_state.get('fetch_data', False) and st.session_state.run_prediction:
    st.header("LSTM Price Prediction")
    
    with st.spinner("Training LSTM model and generating predictions..."):
        try:
            # Get data
            df = st.session_state.data
            
            # Prepare data for LSTM
            data_fetcher = DataFetcher()
            sequence_length = 60  # Number of days to look back
            X_train, y_train, X_test, y_test = data_fetcher.prepare_data(df, sequence_length=sequence_length)
            
            # Initialize and train the LSTM model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = StockPredictor(input_shape)
            model.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
            
            # Get the last sequence for prediction
            last_sequence = df['Close'].values[-sequence_length:].reshape(-1, 1)
            last_sequence = data_fetcher.scaler.transform(last_sequence)
            last_sequence = np.array([last_sequence])
            
            # Predict the next 'prediction_days' days
            predicted_prices = []
            current_sequence = last_sequence[0].copy()
            
            for _ in range(st.session_state.prediction_days):
                # Reshape for prediction
                current_sequence_reshaped = current_sequence.reshape(1, sequence_length, 1)
                
                # Predict next day
                predicted_price = model.predict(current_sequence_reshaped)[0][0]
                predicted_prices.append(predicted_price)
                
                # Update current sequence
                current_sequence = np.append(current_sequence[1:], [[predicted_price]], axis=0)
            
            # Inverse transform the predictions
            predicted_prices = data_fetcher.scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            
            # Create date range for predictions - use business days only
            last_date = df.index[-1]
            prediction_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=st.session_state.prediction_days, 
                freq='B'
            )
            
            # Ensure we have the same number of dates as predictions
            if len(prediction_dates) != len(predicted_prices):
                # If not, create generic dates to avoid issues
                prediction_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(predicted_prices))]
            
            # Store predictions
            st.session_state.prediction_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted_Close': predicted_prices.flatten()
            })
            
            # Create a plot of historical prices and predictions
            st.subheader("Historical and Predicted Stock Prices")
            
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Historical Close Price',
                line=dict(color='blue')
            ))
            
            # Add predicted prices
            fig.add_trace(go.Scatter(
                x=st.session_state.prediction_df['Date'],
                y=st.session_state.prediction_df['Predicted_Close'],
                mode='lines+markers',
                name='Predicted Close Price',
                line=dict(color='red', dash='dash')
            ))
            
            # Add confidence interval
            lower_bound = st.session_state.prediction_df['Predicted_Close'] * 0.95  # Simple 5% lower bound
            upper_bound = st.session_state.prediction_df['Predicted_Close'] * 1.05  # Simple 5% upper bound
            
            fig.add_trace(go.Scatter(
                x=st.session_state.prediction_df['Date'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.prediction_df['Date'],
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='95% Confidence Interval'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{st.session_state.ticker} Stock Price Prediction for Next {st.session_state.prediction_days} Trading Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
                template=theme,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction metrics
            latest_close = safe_float(df['Close'].iloc[-1])
            predicted_next_day = safe_float(st.session_state.prediction_df['Predicted_Close'].iloc[0])
            price_change = predicted_next_day - latest_close
            percent_change = (price_change / latest_close) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Latest Close Price",
                    value=f"${latest_close:.2f}"
                )
            
            with col2:
                st.metric(
                    label="Predicted Next Day",
                    value=f"${predicted_next_day:.2f}",
                    delta=f"{price_change:.2f}"
                )
            
            with col3:
                prediction_trend = "Uptrend" if percent_change > 0 else "Downtrend"
                st.metric(
                    label="Predicted Trend",
                    value=prediction_trend,
                    delta=f"{percent_change:.2f}%"
                )
            
            # Show prediction table
            st.subheader("Day-by-Day Predictions")
            
            # Format the prediction dataframe
            prediction_table = st.session_state.prediction_df.copy()
            prediction_table['Date'] = prediction_table['Date'].dt.strftime('%Y-%m-%d')
            prediction_table['Predicted_Close'] = prediction_table['Predicted_Close'].apply(lambda x: f"${x:.2f}")
            
            # Display the table
            st.table(prediction_table)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.exception(e)

# RL Agent Simulation
if st.session_state.get('fetch_data', False) and st.session_state.run_simulation:
    st.header("RL Trading Simulation")
    
    with st.spinner("Running RL agent trading simulation..."):
        try:
            # Get data with features
            df = st.session_state.data_with_features.copy()
            
            # Create the trading environment
            env = TradingEnvironment(df, initial_balance=st.session_state.initial_balance)
            
            # Create and train (or load) the RL agent
            # In a real app, you would train this offline and just load it here
            # For demo purposes, we'll just create a simple simulation
            
            # Reset the environment
            observation, info = env.reset()
            
            # Simulate trading decisions
            done = False
            trade_history = []
            portfolio_history = []
            actions_history = []
            
            # For demo purposes, let's use a simple strategy instead of a fully trained RL agent
            while not done:
                # Get current observation
                close_value = env.df['Close'].iloc[env.current_step]
                current_price = safe_float(close_value)
                
                # Simple trading strategy:
                # - Buy if price is below 5-day moving average
                # - Sell if price is above 5-day moving average
                # - Hold otherwise
                if env.current_step > 5:
                    ma5_value = env.df['MA5'].iloc[env.current_step]
                    ma5 = safe_float(ma5_value)
                    if current_price < ma5 and env.balance > current_price:
                        action = 1  # Buy
                    elif current_price > ma5 and env.shares_held > 0:
                        action = 2  # Sell
                    else:
                        action = 0  # Hold
                else:
                    action = 0  # Hold in the beginning
                    
                # Take action
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Store information
                portfolio_value = env.balance + env.shares_held * env.current_price
                
                # Track trade history
                if action != 0:  # If not hold
                    trade_type = "BUY" if action == 1 else "SELL"
                    trade_history.append({
                        'date': env.df.index[env.current_step].strftime('%Y-%m-%d'),
                        'price': env.current_price,
                        'action': trade_type,
                        'shares': env.shares_held if action == 1 else 0,  # New shares held after buying or 0 after selling
                        'balance': env.balance,
                        'portfolio_value': portfolio_value
                    })
                
                # Track portfolio history
                portfolio_history.append({
                    'date': env.df.index[env.current_step],
                    'price': env.current_price,
                    'balance': env.balance,
                    'shares_held': env.shares_held,
                    'portfolio_value': portfolio_value
                })
                
                # Track actions
                actions_history.append({
                    'date': env.df.index[env.current_step],
                    'action': action,
                    'action_name': ["HOLD", "BUY", "SELL"][action],
                    'reward': reward,
                    'portfolio_value': portfolio_value
                })
                
                # Update observation
                observation = next_observation
                done = terminated or truncated
            
            # Create portfolio dataframe
            portfolio_df = pd.DataFrame(portfolio_history)
            
            # Create actions dataframe
            actions_df = pd.DataFrame(actions_history)
            
            # Create trade history dataframe
            if trade_history:
                trade_df = pd.DataFrame(trade_history)
            else:
                trade_df = pd.DataFrame(columns=['date', 'price', 'action', 'shares', 'balance', 'portfolio_value'])
            
            # Store in session state
            st.session_state.portfolio_df = portfolio_df
            st.session_state.actions_df = actions_df
            st.session_state.trade_df = trade_df
            st.session_state.final_info = info
            
            # Display portfolio performance
            st.subheader("Portfolio Performance")
            
            # Create a plot with price and portfolio value
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add stock price line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name="Stock Price",
                    line=dict(color='blue')
                ),
                secondary_y=False
            )
            
            # Add portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    name="Portfolio Value",
                    line=dict(color='green')
                ),
                secondary_y=True
            )
            
            # Add buy markers
            buy_actions = actions_df[actions_df['action'] == 1]
            if not buy_actions.empty:
                buy_dates = buy_actions['date'].tolist()
                buy_prices = []
                for date in buy_dates:
                    try:
                        price_value = df.loc[date, 'Close']
                        buy_prices.append(safe_float(price_value))
                    except (KeyError, TypeError):
                        # Handle potential KeyError if date is not in index
                        buy_prices.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        marker=dict(
                            color='green',
                            size=10,
                            symbol='triangle-up'
                        ),
                        name="Buy"
                    ),
                    secondary_y=False
                )
            
            # Add sell markers
            sell_actions = actions_df[actions_df['action'] == 2]
            if not sell_actions.empty:
                sell_dates = sell_actions['date'].tolist()
                sell_prices = []
                for date in sell_dates:
                    try:
                        price_value = df.loc[date, 'Close']
                        sell_prices.append(safe_float(price_value))
                    except (KeyError, TypeError):
                        # Handle potential KeyError if date is not in index
                        sell_prices.append(None)
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='triangle-down'
                        ),
                        name="Sell"
                    ),
                    secondary_y=False
                )
            
            # Update layout
            fig.update_layout(
                title_text=f"Portfolio Performance for {st.session_state.ticker}",
                xaxis_title="Date",
                template=theme
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=True)
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display final performance metrics
            final_portfolio = portfolio_df.iloc[-1]['portfolio_value']
            initial_investment = st.session_state.initial_balance
            total_return = final_portfolio - initial_investment
            percent_return = (total_return / initial_investment) * 100
            
            # Calculate Sharpe ratio
            if len(portfolio_df) > 1:
                # Calculate daily returns
                portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
                
                # Calculate Sharpe ratio (annualized, assuming 252 trading days)
                sharpe_ratio = portfolio_df['daily_return'].mean() / (portfolio_df['daily_return'].std() + 1e-10) * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
            # Calculate drawdown
            max_drawdown = info['max_drawdown'] * 100
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Return",
                    value=f"${total_return:.2f}",
                    delta=f"{percent_return:.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Final Portfolio",
                    value=f"${final_portfolio:.2f}"
                )
            
            with col3:
                st.metric(
                    label="Sharpe Ratio",
                    value=f"{sharpe_ratio:.2f}"
                )
            
            with col4:
                st.metric(
                    label="Max Drawdown",
                    value=f"{max_drawdown:.2f}%",
                    delta=f"-{max_drawdown:.2f}%",
                    delta_color="inverse"
                )
            
            # Display trade history
            st.subheader("Trade History")
            if not trade_df.empty:
                st.dataframe(trade_df)
            else:
                st.info("No trades were executed during this simulation.")
                
            # Trading Decisions Explanation
            st.subheader("Trading Decision Explanations")
            
            # For demo purposes, let's manually create some explanations
            if not actions_df[actions_df['action'] != 0].empty:
                explanations = []
                for _, row in actions_df[actions_df['action'] != 0].iterrows():
                    try:
                        date = row['date']
                        action = row['action']
                        price = row['price'] if 'price' in row else env.df.loc[date, 'Close']
                        ma5 = env.df.loc[date, 'MA5']
                        
                        price = safe_float(price)
                        ma5 = safe_float(ma5)
                        
                        if action == 1:
                            explanation = f"Bought because price ({price:.2f}) was below 5-day moving average ({ma5:.2f})"
                        else:
                            explanation = f"Sold because price ({price:.2f}) was above 5-day moving average ({ma5:.2f})"
                        
                        explanations.append({
                            'date': date,
                            'action': ["HOLD", "BUY", "SELL"][action],
                            'explanation': explanation
                        })
                    except Exception as e:
                        explanations.append({
                            'date': row['date'],
                            'action': ["HOLD", "BUY", "SELL"][row['action']],
                            'explanation': f"Decision based on trading strategy (details unavailable)"
                        })
                
                explanation_df = pd.DataFrame(explanations)
                st.dataframe(explanation_df[['date', 'action', 'explanation']])
            else:
                st.info("No trading decisions were made during this simulation.")
        
        except Exception as e:
            st.error(f"An error occurred during simulation: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>AI Trading Bot Dashboard | Created with Streamlit, TensorFlow, and Stable-Baselines3</p>
    </div>
""", unsafe_allow_html=True) 