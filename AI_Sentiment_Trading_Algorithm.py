# AI Sentiment Trading Algorithm for QuantConnect
# 
# This algorithm combines traditional technical indicators with AI-powered
# sentiment analysis to make trading decisions that adapt to market conditions.
# It leverages reasoning AI models (like DeepSeek R1) for sophisticated sentiment analysis.

import clr
from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import *
from QuantConnect.Securities import *
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime, timedelta

class AISentimentTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        """Initialize the algorithm settings and data subscriptions."""
        self.SetStartDate(2020, 1, 1)    # Set start date
        self.SetCash(100000)             # Set strategy cash
        
        # Universe selection - major US stocks
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Set the brokerage model for realistic trading
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Initialize indicators dictionary for each security
        self.indicators = {}
        
        # List of securities we're currently trading
        self.activeSecurities = []
        
        # Historical sentiment data cache
        self.sentimentCache = {}
        
        # AI Sentiment API Configuration (replace with your actual credentials)
        self.sentiment_api_key = self.GetParameter("SentimentAPIKey")
        self.sentiment_api_url = self.GetParameter("SentimentAPIEndpoint")
        
        # Risk management parameters
        self.max_position_size = 0.05     # Maximum position size as percentage of portfolio
        self.stop_loss_pct = 0.05         # Stop loss percentage
        self.take_profit_pct = 0.15       # Take profit percentage
        
        # Trading parameters
        self.holding_period = 5           # Default holding period in days
        self.sentiment_threshold = 0.6    # Sentiment score threshold for trading
        self.rebalance_frequency = 7      # Rebalance portfolio every n days
        
        # Schedule rebalancing function
        self.Schedule.On(self.DateRules.Every(self.rebalance_frequency), 
                         self.TimeRules.AfterMarketOpen("SPY", 30), 
                         self.RebalancePortfolio)
        
        # Custom insight model parameters
        self.lookback_period = 20         # Days for lookback
        self.volatility_window = 10       # Days for volatility calculation
        
        # Adaptive parameters
        self.market_regime = "normal"     # Track market regime (normal, volatile, trending)
        self.last_regime_check = datetime.now()
        self.regime_check_frequency = 14  # Check market regime every n days
        
        # Schedule the market regime detection
        self.Schedule.On(self.DateRules.Every(self.regime_check_frequency), 
                         self.TimeRules.AfterMarketOpen("SPY", 15), 
                         self.DetectMarketRegime)
    
    def CoarseSelectionFunction(self, coarse):
        """
        Universe selection function to pick stocks for trading.
        """
        # Filter the stocks by dollar volume and price
        filtered = [x for x in coarse if x.HasFundamentalData and x.Price > 10 and x.DollarVolume > 10000000]
        
        # Sort by dollar volume and take the top 50 stocks
        filtered = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)[:50]
        
        # Get the symbols from the filtered stocks
        symbols = [x.Symbol for x in filtered]
        
        # Update the active securities list
        self.activeSecurities = symbols
        
        # Set up indicators for new securities
        for symbol in symbols:
            if symbol not in self.indicators:
                self.indicators[symbol] = self.InitializeIndicators(symbol)
        
        return symbols
    
    def InitializeIndicators(self, symbol):
        """
        Initialize technical indicators for a given symbol.
        """
        return {
            "SMA20": self.SMA(symbol, 20, Resolution.Daily),
            "SMA50": self.SMA(symbol, 50, Resolution.Daily),
            "RSI": self.RSI(symbol, 14, MovingAverageType.Simple, Resolution.Daily),
            "BB": self.BB(symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily),
            "MACD": self.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily),
            "ATR": self.ATR(symbol, 14, MovingAverageType.Simple, Resolution.Daily)
        }
    
    def OnData(self, data):
        """
        Event handler for market data updates.
        """
        # Skip if we're not ready to trade
        if not self.indicators or len(self.activeSecurities) == 0:
            return
        
        # Process each security in our universe
        for symbol in self.activeSecurities:
            if not data.ContainsKey(symbol) or not data[symbol]:
                continue
            
            # Skip if the indicators aren't ready
            if not self.indicators[symbol]["SMA20"].IsReady or not self.indicators[symbol]["RSI"].IsReady:
                continue
            
            # Get the current holdings for this symbol
            holding = self.Portfolio[symbol]
            
            # Check if we have an existing position to manage
            if holding.Invested:
                self.ManageExistingPosition(symbol, holding, data[symbol])
    
    def ManageExistingPosition(self, symbol, holding, price_data):
        """
        Manage an existing position with trailing stops and take profit orders.
        """
        current_price = price_data.Close
        entry_price = holding.AveragePrice
        
        # Calculate profit/loss percentage
        pnl_pct = (current_price / entry_price - 1) * (1 if holding.IsLong else -1)
        
        # Dynamic stop loss based on ATR
        atr_value = self.indicators[symbol]["ATR"].Current.Value
        atr_multiplier = 2
        
        # For long positions
        if holding.IsLong:
            # Dynamic stop loss (2 x ATR below entry or fixed %, whichever is tighter)
            stop_price = max(entry_price * (1 - self.stop_loss_pct), 
                           current_price - (atr_value * atr_multiplier))
            
            # Take profit level
            take_profit = entry_price * (1 + self.take_profit_pct)
            
            # Exit if stop loss is hit or take profit is reached
            if current_price <= stop_price:
                self.Liquidate(symbol, "Stop loss triggered")
                self.Log(f"Stop loss triggered for {symbol} at {current_price}")
                
            elif current_price >= take_profit:
                self.Liquidate(symbol, "Take profit triggered")
                self.Log(f"Take profit triggered for {symbol} at {current_price}")
                
        # For short positions
        else:
            # Dynamic stop loss (2 x ATR above entry or fixed %, whichever is tighter)
            stop_price = min(entry_price * (1 + self.stop_loss_pct), 
                           current_price + (atr_value * atr_multiplier))
            
            # Take profit level
            take_profit = entry_price * (1 - self.take_profit_pct)
            
            # Exit if stop loss is hit or take profit is reached
            if current_price >= stop_price:
                self.Liquidate(symbol, "Stop loss triggered")
                self.Log(f"Stop loss triggered for {symbol} at {current_price}")
                
            elif current_price <= take_profit:
                self.Liquidate(symbol, "Take profit triggered")
                self.Log(f"Take profit triggered for {symbol} at {current_price}")
    
    def RebalancePortfolio(self):
        """
        Rebalance the portfolio based on latest sentiment and technical signals.
        """
        # Get the latest sentiment data for all active securities
        self.UpdateSentimentData()
        
        # Rank securities by combined score (sentiment + technical)
        ranked_securities = self.RankSecurities()
        
        # Calculate number of positions based on available cash and max position size
        max_positions = min(10, len(ranked_securities))
        
        # Liquidate securities that are no longer in our top ranked list
        for symbol in self.Portfolio.Keys:
            holding = self.Portfolio[symbol]
            if holding.Invested and symbol not in [s for s, _ in ranked_securities[:max_positions]]:
                self.Liquidate(symbol, "No longer in top ranked securities")
        
        # Invest in top ranked securities
        for i in range(min(max_positions, len(ranked_securities))):
            symbol, score = ranked_securities[i]
            
            # Skip if already invested
            if self.Portfolio[symbol].Invested:
                continue
            
            # Calculate position size based on score and maximum position size
            position_size = self.Portfolio.TotalPortfolioValue * self.max_position_size * (score / 2)
            
            # Adjust position size based on market regime
            if self.market_regime == "volatile":
                position_size *= 0.7  # Reduce position size in volatile markets
            elif self.market_regime == "trending":
                position_size *= 1.2  # Increase position size in trending markets
                position_size = min(position_size, self.Portfolio.TotalPortfolioValue * self.max_position_size)
            
            # Determine trade direction based on score and sentiment
            sentiment = self.GetLatestSentiment(symbol)
            if sentiment > self.sentiment_threshold and score > 1.5:
                self.SetHoldings(symbol, position_size / self.Portfolio.TotalPortfolioValue)
                self.Log(f"Opening long position in {symbol} with score {score} and sentiment {sentiment}")
            elif sentiment < -self.sentiment_threshold and score > 1.5:
                self.SetHoldings(symbol, -position_size / self.Portfolio.TotalPortfolioValue)
                self.Log(f"Opening short position in {symbol} with score {score} and sentiment {sentiment}")
    
    def UpdateSentimentData(self):
        """
        Update sentiment data for all active securities.
        This function calls the external AI sentiment API.
        """
        for symbol in self.activeSecurities:
            # Skip if we already have recent sentiment data (less than 1 day old)
            if symbol in self.sentimentCache and (datetime.now() - self.sentimentCache[symbol]["timestamp"]).days < 1:
                continue
            
            try:
                # Get company ticker 
                ticker = str(symbol.Value)
                
                # Call the AI sentiment analysis API
                sentiment_data = self.GetAISentiment(ticker)
                
                # Cache the sentiment data with timestamp
                self.sentimentCache[symbol] = {
                    "sentiment": sentiment_data,
                    "timestamp": datetime.now()
                }
                
                self.Log(f"Updated sentiment for {ticker}: {sentiment_data}")
                
            except Exception as e:
                self.Error(f"Error updating sentiment for {symbol}: {str(e)}")
    
    def GetAISentiment(self, ticker):
        """
        Call the external AI sentiment API to get sentiment data for a ticker.
        This is where you would integrate with DeepSeek R1 or similar AI.
        
        For QuantConnect, you'd need a hosted API endpoint that wraps the AI model.
        """
        try:
            # Example API call structure (replace with actual implementation)
            # In a real implementation, this would be an API that uses DeepSeek R1
            # or another reasoning AI to analyze news, social media, etc.
            
            # For testing purposes, let's return a simulated sentiment score
            # In production, you would replace this with an actual API call
            
            # Example API call (commented out):
            # headers = {"Authorization": f"Bearer {self.sentiment_api_key}"}
            # params = {"ticker": ticker, "days": 7}  # Get sentiment for past 7 days
            # response = requests.get(f"{self.sentiment_api_url}/sentiment", headers=headers, params=params)
            # response.raise_for_status()
            # sentiment_data = response.json()
            # return sentiment_data["score"]  # Score between -1 (negative) and 1 (positive)
            
            # For testing, generate a random sentiment score
            # In production, this would be replaced with real API calls
            import random
            return random.uniform(-1, 1)
            
        except Exception as e:
            self.Error(f"Error getting sentiment for {ticker}: {str(e)}")
            return 0  # Neutral sentiment on error
    
    def GetLatestSentiment(self, symbol):
        """
        Get the latest sentiment score for a symbol from the cache.
        """
        if symbol in self.sentimentCache:
            return self.sentimentCache[symbol]["sentiment"]
        return 0  # Neutral sentiment if not found
    
    def RankSecurities(self):
        """
        Rank securities based on combined technical and sentiment scores.
        """
        ranked_securities = []
        
        for symbol in self.activeSecurities:
            # Skip if indicators aren't ready
            if not self.indicators[symbol]["SMA20"].IsReady or not self.indicators[symbol]["RSI"].IsReady:
                continue
            
            # Get technical score
            technical_score = self.CalculateTechnicalScore(symbol)
            
            # Get sentiment score
            sentiment_score = abs(self.GetLatestSentiment(symbol))
            
            # Combined score (technical + sentiment)
            combined_score = technical_score + sentiment_score
            
            # Add to ranked list
            ranked_securities.append((symbol, combined_score))
        
        # Sort by combined score, descending
        ranked_securities.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_securities
    
    def CalculateTechnicalScore(self, symbol):
        """
        Calculate a technical score for a security based on various indicators.
        """
        # Get indicator values
        sma20 = self.indicators[symbol]["SMA20"].Current.Value
        sma50 = self.indicators[symbol]["SMA50"].Current.Value
        rsi = self.indicators[symbol]["RSI"].Current.Value
        macd = self.indicators[symbol]["MACD"].Current.Value
        macd_signal = self.indicators[symbol]["MACD"].Signal.Current.Value
        
        # Get current price
        current_price = self.Securities[symbol].Price
        
        # Score components
        trend_score = 0
        momentum_score = 0
        
        # Trend score based on moving averages
        if current_price > sma20 and sma20 > sma50:
            trend_score = 1  # Strong uptrend
        elif current_price > sma20:
            trend_score = 0.5  # Moderate uptrend
        elif current_price < sma20 and sma20 < sma50:
            trend_score = -1  # Strong downtrend
        elif current_price < sma20:
            trend_score = -0.5  # Moderate downtrend
        
        # Momentum score based on RSI and MACD
        if rsi > 70:
            momentum_score = -0.5  # Overbought
        elif rsi < 30:
            momentum_score = 0.5  # Oversold
        else:
            momentum_score = 0  # Neutral
        
        # Add MACD signal
        if macd > macd_signal:
            momentum_score += 0.5  # Bullish MACD
        elif macd < macd_signal:
            momentum_score -= 0.5  # Bearish MACD
        
        # Adjust based on market regime
        if self.market_regime == "volatile":
            momentum_score *= 1.5  # Give more weight to momentum in volatile markets
        elif self.market_regime == "trending":
            trend_score *= 1.5  # Give more weight to trend in trending markets
        
        # Combined technical score (absolute value for ranking purposes)
        return abs(trend_score + momentum_score)
    
    def DetectMarketRegime(self):
        """
        Detect the current market regime (normal, volatile, trending)
        based on SPY behavior.
        """
        spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Get historical data
        history = self.History(spy, self.lookback_period, Resolution.Daily)
        if history.empty:
            return
        
        # Calculate volatility
        returns = history["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend strength (R-squared of linear regression)
        import scipy.stats as stats
        x = np.arange(len(history["close"]))
        slope, _, r_value, _, _ = stats.linregress(x, history["close"])
        r_squared = r_value ** 2
        
        # Determine market regime
        high_volatility_threshold = 0.25
        trending_threshold = 0.7
        
        if volatility > high_volatility_threshold:
            self.market_regime = "volatile"
            self.Log(f"Market regime changed to volatile. Volatility: {volatility:.4f}")
        elif r_squared > trending_threshold:
            self.market_regime = "trending"
            trend_direction = "uptrend" if slope > 0 else "downtrend"
            self.Log(f"Market regime changed to trending ({trend_direction}). R-squared: {r_squared:.4f}")
        else:
            self.market_regime = "normal"
            self.Log(f"Market regime changed to normal. Volatility: {volatility:.4f}, R-squared: {r_squared:.4f}")
            
        # Adjust trading parameters based on market regime
        self.AdjustParametersForMarketRegime()
    
    def AdjustParametersForMarketRegime(self):
        """
        Adjust trading parameters based on the current market regime.
        """
        if self.market_regime == "volatile":
            # In volatile markets, be more conservative
            self.max_position_size = 0.03  # Smaller positions
            self.holding_period = 3  # Shorter holding period
            self.stop_loss_pct = 0.04  # Tighter stops
            self.take_profit_pct = 0.10  # Lower take profit
            
        elif self.market_regime == "trending":
            # In trending markets, be more aggressive
            self.max_position_size = 0.07  # Larger positions
            self.holding_period = 7  # Longer holding period
            self.stop_loss_pct = 0.06  # Wider stops
            self.take_profit_pct = 0.20  # Higher take profit
            
        else:  # normal regime
            # Reset to default parameters
            self.max_position_size = 0.05
            self.holding_period = 5
            self.stop_loss_pct = 0.05
            self.take_profit_pct = 0.15
            
        self.Log(f"Adjusted parameters for {self.market_regime} market regime")

# End of algorithm