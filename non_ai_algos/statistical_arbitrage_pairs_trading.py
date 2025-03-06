# Statistical Arbitrage Pairs Trading Strategy
#
# This algorithm implements a sophisticated pairs trading strategy that:
# 1. Identifies pairs of securities with high statistical correlation and cointegration
# 2. Dynamically adjusts entry/exit thresholds based on market volatility
# 3. Implements optimal position sizing using the Kelly criterion
# 4. Manages risk through dynamic stop-loss and take-profit levels
#
# Designed for QuantConnect platform by Zachary Amador

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import scipy.stats as stats

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import *

class StatisticalArbitragePairsTrading(QCAlgorithm):
    
    def Initialize(self):
        """Initialize the algorithm with required settings and data."""
        # Set start date and cash
        self.SetStartDate(2018, 1, 1)  # Start date
        self.SetCash(1000000)           # Initial capital
        
        # Set brokerage model for realistic trading
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Parameters for pairs formation
        self.lookback = 252              # Lookback period for pair formation (1 year)
        self.formation_period = 63       # Formation period in days (3 months)
        self.trading_period = 63         # Trading period in days (3 months)
        self.max_pairs = 10              # Maximum number of pairs to trade simultaneously
        self.minimum_history = 252       # Minimum history required for pair selection
        
        # Parameters for pair trading
        self.entry_threshold = 2.0       # Entry threshold in standard deviations
        self.exit_threshold = 0.0        # Exit threshold in standard deviations (mean reversion)
        self.stop_loss_threshold = 4.0   # Stop loss threshold in standard deviations
        self.max_position_size = 0.05    # Maximum position size per pair (5% of portfolio)
        self.kelly_fraction = 0.3        # Fraction of Kelly criterion to use (more conservative)
        self.portfolio_fraction = 0.8    # Maximum fraction of portfolio to allocate to all pairs
        
        # Risk management parameters
        self.max_drawdown = 0.10         # Maximum drawdown allowed (10%)
        self.max_leverage = 2.0          # Maximum leverage allowed
        self.correlation_threshold = 0.7  # Minimum correlation for pair selection
        self.pvalue_threshold = 0.05     # Maximum p-value for cointegration test
        self.half_life_max = 21          # Maximum half-life for mean reversion (days)
        
        # Initialize dictionaries and lists
        self.pairs = []                  # List of active pairs
        self.pair_stats = {}             # Statistics for each pair
        self.pairs_in_trade = {}         # Pairs currently in trade
        self.universe_symbols = []       # Symbols in our trading universe
        self.last_rebalance_time = None  # Time of last pairs rebalancing
        
        # Add SPY for market reference
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Universe selection
        self.sector_etfs = {
            "XLF": "Financials",
            "XLK": "Technology",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary",
            "XLB": "Materials",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
            "XLC": "Communication Services"
        }
        
        # Add sector ETFs for tracking
        self.sectors = {}
        for etf, sector_name in self.sector_etfs.items():
            self.sectors[sector_name] = self.AddEquity(etf, Resolution.Daily).Symbol
        
        # Create a custom universe of liquid stocks within sectors
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Schedule pair formation and rebalancing
        self.Schedule.On(self.DateRules.MonthStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.FormPairs)
        
        # Schedule daily trade management (stops, exits, etc.)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 15), 
                        self.ManageTrades)
                        
        # Schedule risk management check
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                       self.TimeRules.AfterMarketOpen("SPY", 5), 
                       self.RiskManagement)
    
    def CoarseSelectionFunction(self, coarse):
        """
        Select securities based on liquidity and price.
        """
        # Filter for stocks with price > $5 and volume > $5M
        filtered = [x for x in coarse if x.Price > 5 and x.DollarVolume > 5000000]
        
        # Sort by dollar volume and take top 200 stocks
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        selected = [x.Symbol for x in sorted_by_volume[:200]]
        
        # Update universe symbols
        self.universe_symbols = selected
        
        return selected
    
    def FormPairs(self):
        """
        Form pairs based on correlation and cointegration analysis.
        """
        # Check if we have enough symbols in the universe
        if len(self.universe_symbols) < 2:
            self.Log("Not enough symbols in the universe to form pairs")
            return
        
        # Check if we're ready to rebalance (once per formation period)
        if self.last_rebalance_time is not None:
            days_elapsed = (self.Time - self.last_rebalance_time).days
            if days_elapsed < self.formation_period:
                return
        
        # Close any open positions before rebalancing
        self.CloseAllPositions()
        
        # Clear existing pairs
        self.pairs = []
        self.pair_stats = {}
        self.pairs_in_trade = {}
        
        # Get historical data for all securities
        history = self.History(self.universe_symbols, self.lookback, Resolution.Daily)
        if history.empty:
            self.Log("No historical data available for pair formation")
            return
            
        # Convert history to dataframe with close prices
        price_df = history['close'].unstack(level=0)
        
        # Remove securities with missing data
        min_data_points = self.lookback * 0.95  # Allow 5% missing data
        valid_columns = [col for col in price_df.columns if price_df[col].count() >= min_data_points]
        price_df = price_df[valid_columns]
        
        # Only proceed if we have enough securities
        if len(valid_columns) < 2:
            self.Log("Not enough valid securities with sufficient history")
            return
        
        # Calculate returns for correlation analysis
        returns_df = price_df.pct_change().dropna()
        
        # Get correlation matrix
        corr_matrix = returns_df.corr()
        
        # Identify potential pairs based on correlation
        potential_pairs = []
        for i in range(len(valid_columns)):
            for j in range(i+1, len(valid_columns)):
                sym1 = valid_columns[i]
                sym2 = valid_columns[j]
                
                # Check if correlation is above threshold
                if corr_matrix.loc[sym1, sym2] > self.correlation_threshold:
                    # Test for cointegration
                    y = price_df[sym1].dropna()
                    x = price_df[sym2].dropna()
                    
                    # Check if we have enough data points after dropping NaNs
                    if len(y) < self.minimum_history or len(x) < self.minimum_history:
                        continue
                    
                    # Ensure equal length
                    min_len = min(len(y), len(x))
                    y = y[-min_len:]
                    x = x[-min_len:]
                    
                    # Perform cointegration test
                    score, pvalue, _ = coint(y, x)
                    
                    # Only proceed if cointegration is statistically significant
                    if pvalue < self.pvalue_threshold:
                        # Estimate hedge ratio using OLS regression
                        model = OLS(y, x).fit()
                        hedge_ratio = model.params[0]
                        
                        # Calculate spread
                        spread = y - hedge_ratio * x
                        
                        # Calculate half-life of mean reversion
                        half_life = self.CalculateHalfLife(spread)
                        
                        # Only select pairs with reasonable half-life
                        if 1 <= half_life <= self.half_life_max:
                            # Calculate spread z-score
                            spread_mean = spread.mean()
                            spread_std = spread.std()
                            
                            # Calculate Hurst exponent to confirm mean reversion
                            hurst = self.CalculateHurstExponent(spread)
                            
                            # Add pair if it shows mean reversion (Hurst < 0.5)
                            if hurst < 0.5:
                                potential_pairs.append({
                                    "sym1": sym1,
                                    "sym2": sym2,
                                    "correlation": corr_matrix.loc[sym1, sym2],
                                    "hedge_ratio": hedge_ratio,
                                    "pvalue": pvalue,
                                    "half_life": half_life,
                                    "spread_mean": spread_mean,
                                    "spread_std": spread_std,
                                    "hurst": hurst
                                })
        
        # Sort potential pairs by cointegration strength (lower p-value)
        potential_pairs.sort(key=lambda x: x["pvalue"])
        
        # Select top N pairs
        selected_pairs = potential_pairs[:self.max_pairs]
        
        # Log selected pairs
        self.Log(f"Selected {len(selected_pairs)} pairs for trading")
        for pair in selected_pairs:
            self.Log(f"Pair: {pair['sym1'].Value} - {pair['sym2'].Value}, Correlation: {pair['correlation']:.2f}, " + 
                    f"P-value: {pair['pvalue']:.4f}, Half-life: {pair['half_life']:.2f} days, Hurst: {pair['hurst']:.2f}")
            
            # Add to our pairs list and store statistics
            pair_id = f"{pair['sym1'].Value}_{pair['sym2'].Value}"
            self.pairs.append(pair_id)
            self.pair_stats[pair_id] = pair
        
        # Update last rebalance time
        self.last_rebalance_time = self.Time
        
        # Subscribe to data for selected pairs
        symbols_to_subscribe = []
        for pair in selected_pairs:
            symbols_to_subscribe.append(pair["sym1"])
            symbols_to_subscribe.append(pair["sym2"])
        
        # Ensure all required symbols are in the universe
        for symbol in symbols_to_subscribe:
            if symbol not in self.Securities:
                self.AddEquity(symbol.Value, Resolution.Daily)
    
    def ManageTrades(self):
        """
        Check for entry/exit signals for all pairs and manage existing trades.
        """
        # Skip if no pairs have been formed
        if len(self.pairs) == 0:
            return
        
        # Calculate total capital available for pairs trading
        available_capital = self.Portfolio.TotalPortfolioValue * self.portfolio_fraction
        capital_per_pair = available_capital / len(self.pairs)
        
        # Check each pair for signals
        for pair_id in self.pairs:
            # Extract pair statistics
            pair = self.pair_stats[pair_id]
            sym1 = pair["sym1"]
            sym2 = pair["sym2"]
            hedge_ratio = pair["hedge_ratio"]
            spread_mean = pair["spread_mean"]
            spread_std = pair["spread_std"]
            
            # Skip if we don't have recent prices for both symbols
            if not (self.Securities.ContainsKey(sym1) and self.Securities.ContainsKey(sym2)):
                continue
                
            # Get current prices
            price1 = self.Securities[sym1].Price
            price2 = self.Securities[sym2].Price
            
            # Calculate current spread
            current_spread = price1 - hedge_ratio * price2
            
            # Calculate z-score of current spread
            z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Determine if the pair is currently in a trade
            in_trade = pair_id in self.pairs_in_trade
            
            # For pairs not in a trade, check entry signals
            if not in_trade:
                # Entry signal when spread deviates significantly from mean
                if z_score > self.entry_threshold:
                    # Spread is positive and above threshold - go short sym1, long sym2
                    self.Log(f"ENTRY SIGNAL: {pair_id} Z-score: {z_score:.2f} > {self.entry_threshold}")
                    
                    # Calculate optimal position size using Kelly criterion
                    position_size = self.CalculateKellyPositionSize(pair, capital_per_pair, "negative")
                    
                    # Open the trade if position size is significant
                    if position_size > 0.001:  # Minimum position size threshold
                        # Calculate number of shares based on price ratio and hedge ratio
                        capital_for_trade = position_size * capital_per_pair
                        
                        # Allocate capital between the two legs ensuring dollar-neutral positions
                        price_ratio = price1 / (hedge_ratio * price2)
                        sym1_capital = capital_for_trade / (1 + price_ratio)
                        sym2_capital = capital_for_trade - sym1_capital
                        
                        # Execute the trade
                        self.SetHoldings(sym1, -sym1_capital / self.Portfolio.TotalPortfolioValue)
                        self.SetHoldings(sym2, sym2_capital / self.Portfolio.TotalPortfolioValue)
                        
                        # Record trade information
                        self.pairs_in_trade[pair_id] = {
                            "entry_time": self.Time,
                            "entry_z_score": z_score,
                            "entry_spread": current_spread,
                            "direction": "negative",  # Expecting spread to decrease
                            "sym1_entry_price": price1,
                            "sym2_entry_price": price2,
                            "stop_loss": z_score + self.stop_loss_threshold
                        }
                        
                        self.Log(f"TRADE OPENED: Short {sym1.Value}, Long {sym2.Value}, Size: {position_size:.2%} of capital per pair")
                
                elif z_score < -self.entry_threshold:
                    # Spread is negative and below threshold - go long sym1, short sym2
                    self.Log(f"ENTRY SIGNAL: {pair_id} Z-score: {z_score:.2f} < -{self.entry_threshold}")
                    
                    # Calculate optimal position size using Kelly criterion
                    position_size = self.CalculateKellyPositionSize(pair, capital_per_pair, "positive")
                    
                    # Open the trade if position size is significant
                    if position_size > 0.001:  # Minimum position size threshold
                        # Calculate number of shares based on price ratio and hedge ratio
                        capital_for_trade = position_size * capital_per_pair
                        
                        # Allocate capital between the two legs ensuring dollar-neutral positions
                        price_ratio = price1 / (hedge_ratio * price2)
                        sym1_capital = capital_for_trade / (1 + price_ratio)
                        sym2_capital = capital_for_trade - sym1_capital
                        
                        # Execute the trade
                        self.SetHoldings(sym1, sym1_capital / self.Portfolio.TotalPortfolioValue)
                        self.SetHoldings(sym2, -sym2_capital / self.Portfolio.TotalPortfolioValue)
                        
                        # Record trade information
                        self.pairs_in_trade[pair_id] = {
                            "entry_time": self.Time,
                            "entry_z_score": z_score,
                            "entry_spread": current_spread,
                            "direction": "positive",  # Expecting spread to increase
                            "sym1_entry_price": price1,
                            "sym2_entry_price": price2,
                            "stop_loss": z_score - self.stop_loss_threshold
                        }
                        
                        self.Log(f"TRADE OPENED: Long {sym1.Value}, Short {sym2.Value}, Size: {position_size:.2%} of capital per pair")
            
            # For pairs in a trade, check exit signals
            else:
                trade_info = self.pairs_in_trade[pair_id]
                direction = trade_info["direction"]
                stop_loss = trade_info["stop_loss"]
                
                # Check for target reached (mean reversion)
                if (direction == "negative" and z_score <= self.exit_threshold) or \
                   (direction == "positive" and z_score >= self.exit_threshold):
                    # Target reached - close the trade
                    self.Log(f"TARGET REACHED: {pair_id} Z-score: {z_score:.2f}, Direction: {direction}")
                    self.ClosePairPosition(sym1, sym2)
                    del self.pairs_in_trade[pair_id]
                    self.Log(f"TRADE CLOSED: Profit target reached for {pair_id}")
                
                # Check for stop loss
                elif (direction == "negative" and z_score >= stop_loss) or \
                     (direction == "positive" and z_score <= stop_loss):
                    # Stop loss triggered - close the trade
                    self.Log(f"STOP LOSS: {pair_id} Z-score: {z_score:.2f}, Stop: {stop_loss:.2f}")
                    self.ClosePairPosition(sym1, sym2)
                    del self.pairs_in_trade[pair_id]
                    self.Log(f"TRADE CLOSED: Stop loss hit for {pair_id}")
                
                # Check for time-based exit (exit if trade duration exceeds half-life * 2)
                days_in_trade = (self.Time - trade_info["entry_time"]).days
                if days_in_trade > pair["half_life"] * 2:
                    self.Log(f"TIME EXIT: {pair_id} has been in trade for {days_in_trade} days, exceeding 2x half-life")
                    self.ClosePairPosition(sym1, sym2)
                    del self.pairs_in_trade[pair_id]
                    self.Log(f"TRADE CLOSED: Time-based exit for {pair_id}")
    
    def RiskManagement(self):
        """
        Manage overall portfolio risk.
        """
        # Check for excessive drawdown
        current_equity = self.Portfolio.TotalPortfolioValue
        peak_equity = self.Portfolio.TotalProfit + current_equity
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            
            # If drawdown exceeds threshold, reduce exposure by closing worst performing pairs
            if drawdown > self.max_drawdown and len(self.pairs_in_trade) > 0:
                self.Log(f"RISK ALERT: Drawdown of {drawdown:.2%} exceeds threshold of {self.max_drawdown:.2%}")
                
                # Calculate PnL for each pair
                pair_pnl = {}
                for pair_id, trade_info in self.pairs_in_trade.items():
                    pair = self.pair_stats[pair_id]
                    sym1 = pair["sym1"]
                    sym2 = pair["sym2"]
                    
                    # Skip if we don't have the securities
                    if not (self.Securities.ContainsKey(sym1) and self.Securities.ContainsKey(sym2)):
                        continue
                    
                    # Calculate unrealized PnL
                    if trade_info["direction"] == "negative":
                        # Short sym1, Long sym2
                        sym1_pnl = (trade_info["sym1_entry_price"] - self.Securities[sym1].Price) / trade_info["sym1_entry_price"]
                        sym2_pnl = (self.Securities[sym2].Price - trade_info["sym2_entry_price"]) / trade_info["sym2_entry_price"]
                    else:
                        # Long sym1, Short sym2
                        sym1_pnl = (self.Securities[sym1].Price - trade_info["sym1_entry_price"]) / trade_info["sym1_entry_price"]
                        sym2_pnl = (trade_info["sym2_entry_price"] - self.Securities[sym2].Price) / trade_info["sym2_entry_price"]
                    
                    # Total PnL for the pair
                    pair_pnl[pair_id] = sym1_pnl + sym2_pnl
                
                # Sort pairs by PnL (worst first)
                sorted_pairs = sorted(pair_pnl.items(), key=lambda x: x[1])
                
                # Close the worst performing half of pairs
                pairs_to_close = len(sorted_pairs) // 2
                pairs_to_close = max(1, pairs_to_close)  # Close at least one pair
                
                for i in range(pairs_to_close):
                    pair_id = sorted_pairs[i][0]
                    pair = self.pair_stats[pair_id]
                    self.Log(f"RISK MANAGEMENT: Closing worst performing pair {pair_id} with PnL {sorted_pairs[i][1]:.2%}")
                    self.ClosePairPosition(pair["sym1"], pair["sym2"])
                    del self.pairs_in_trade[pair_id]
        
        # Check for excessive leverage
        current_leverage = self.Portfolio.TotalAbsoluteHoldingsCost / self.Portfolio.TotalPortfolioValue
        if current_leverage > self.max_leverage:
            self.Log(f"RISK ALERT: Leverage of {current_leverage:.2f}x exceeds maximum of {self.max_leverage:.2f}x")
            
            # Reduce positions proportionally to bring leverage down
            target_leverage = self.max_leverage * 0.9  # Target slightly below maximum
            reduction_factor = target_leverage / current_leverage
            
            # Adjust all positions
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    self.SetHoldings(holding.Symbol, holding.Quantity * reduction_factor / self.Portfolio.TotalPortfolioValue)
            
            self.Log(f"RISK MANAGEMENT: Reduced positions to target leverage of {target_leverage:.2f}x")
    
    def CloseAllPositions(self):
        """
        Close all open positions.
        """
        # Liquidate all holdings except SPY and sector ETFs
        for holding in self.Portfolio.Values:
            if holding.Invested and holding.Symbol != self.spy and holding.Symbol not in self.sectors.values():
                self.Liquidate(holding.Symbol)
        
        # Clear pairs in trade
        self.pairs_in_trade = {}
    
    def ClosePairPosition(self, sym1, sym2):
        """
        Close positions for a specific pair.
        """
        # Liquidate both legs of the pair
        if self.Portfolio[sym1].Invested:
            self.Liquidate(sym1)
        
        if self.Portfolio[sym2].Invested:
            self.Liquidate(sym2)
    
    def CalculateHalfLife(self, spread):
        """
        Calculate the half-life of mean reversion for a spread.
        """
        # Compute autoregression coefficient
        spread_lag = spread.shift(1).dropna()
        spread = spread[1:]  # Align with lagged series
        
        # Run regression on spread vs. lagged spread
        model = OLS(spread.values - spread.mean(), spread_lag.values - spread_lag.mean()).fit()
        
        # Extract autoregression coefficient
        beta = model.params[0]
        
        # Calculate half-life
        if beta < 0:  # Non-mean reverting
            return float('inf')
        elif beta >= 1:  # Unit root or explosive
            return float('inf')
        else:
            return -np.log(2) / np.log(beta)
    
    def CalculateHurstExponent(self, series):
        """
        Calculate the Hurst exponent of a time series.
        
        The Hurst exponent is used to determine if a time series is mean-reverting (H < 0.5),
        random walk (H = 0.5), or trending (H > 0.5).
        """
        # Convert to numpy array
        series = np.array(series)
        
        # Create a range of lag values
        lags = range(2, 20)
        
        # Calculate the array of the variances of the lagged differences
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Return the Hurst exponent
        return poly[0] / 2.0
    
    def CalculateKellyPositionSize(self, pair, capital, direction):
        """
        Calculate the optimal position size using the Kelly Criterion.
        
        The Kelly Criterion is f* = (p * b - (1 - p)) / b
        where:
            f* = fraction of current capital to bet
            p = probability of winning
            b = odds received on the bet (how much you win per unit wagered)
        """
        # For pairs trading, we can estimate the win probability and odds based on historical data
        # We'll use the half-life and historical z-score distribution
        
        # Extract pair statistics
        half_life = pair["half_life"]
        
        # Probability of mean reversion within time period can be estimated from half-life
        # A shorter half-life means higher probability of reversion
        p_win = 1 - np.exp(-self.trading_period / half_life)
        p_win = min(max(p_win, 0.5), 0.95)  # Constrain between 50% and 95%
        
        # Expected return can be estimated from z-score entry threshold to exit threshold
        expected_z_move = self.entry_threshold  # From entry threshold to zero (mean)
        
        # Estimate standard deviation of daily spread moves
        spread_volatility = pair["spread_std"] / np.sqrt(half_life)
        
        # Expected return in dollars
        expected_return = expected_z_move * spread_volatility
        
        # Potential loss
        stop_loss_distance = self.stop_loss_threshold
        potential_loss = stop_loss_distance * spread_volatility
        
        # Calculate odds (how much you win per unit risked)
        odds = expected_return / potential_loss if potential_loss > 0 else 1
        
        # Kelly formula: f* = (p * b - (1 - p)) / b
        kelly = (p_win * odds - (1 - p_win)) / odds if odds > 0 else 0
        
        # Apply a fraction of Kelly for conservatism
        kelly *= self.kelly_fraction
        
        # Ensure we don't exceed max position size
        position_size = min(kelly, self.max_position_size)
        
        # Ensure position size is positive
        position_size = max(position_size, 0)
        
        return position_size
    
    def OnData(self, data):
        """
        Event handler for market data updates.
        """
        # The strategy logic is primarily implemented in scheduled events
        pass
    
    def OnOrderEvent(self, orderEvent):
        """
        Event handler for order status updates.
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order {orderEvent.OrderId} filled: {orderEvent.FillQuantity} of {orderEvent.Symbol} at {orderEvent.FillPrice}")
    
    def OnEndOfAlgorithm(self):
        """
        Event handler called at the end of the algorithm.
        """
        self.Log("Algorithm completed")
        
        # Report final statistics
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"Total Trades: {self.TradeBuilder.ClosedTrades.Count}")
        
        # Report pairs performance
        self.Log(f"Total Pairs Formed: {len(self.pairs)}")
        self.Log(f"Pairs in Trade at End: {len(self.pairs_in_trade)}")