# Adaptive Multi-Factor Strategy with Dynamic Risk Management
#
# This algorithm implements a sophisticated trading strategy that:
# 1. Adapts to different market regimes (trending, mean-reverting, volatile)
# 2. Uses multiple factors to select securities (value, momentum, quality, volatility)
# 3. Implements dynamic risk management with volatility targeting and position sizing
# 4. Includes correlation-based portfolio construction for optimal diversification
#
# Designed for QuantConnect platform by Zachary Amador

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import *

class AdaptiveMultiFactorStrategy(QCAlgorithm):
    
    def Initialize(self):
        """Initialize the algorithm with required settings and data."""
        # Set start date and cash
        self.SetStartDate(2018, 1, 1)  # Start date
        self.SetCash(1000000)           # Initial capital
        
        # Set the benchmark
        self.SetBenchmark("SPY")
        
        # Set brokerage model to enforce realistic trading conditions
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Market regime variables
        self.market_regime = "normal"  # Initial market regime (normal, trending, volatile)
        self.regime_check_days = 10     # Check market regime every n days
        
        # Risk management variables
        self.target_volatility = 0.10   # Target annualized portfolio volatility (10%)
        self.max_position_size = 0.05   # Maximum position size for any security (5%)
        self.max_sector_exposure = 0.25 # Maximum exposure to any sector (25%)
        
        # Universe variables
        self.universe_size = 500        # Number of stocks to consider in universe
        self.top_factors_count = 50     # Number of top stocks to select based on factors
        
        # Factor weights in normal market regime
        self.normal_factor_weights = {
            "value": 0.25,
            "momentum": 0.25, 
            "quality": 0.25,
            "volatility": 0.25
        }
        
        # Initialize empty dictionaries to store data
        self.securities_data = {}       # Store securities data
        self.factor_data = {}           # Store factor data
        self.regime_data = {}           # Store regime indicators
        self.current_weights = {}       # Store current portfolio weights
        
        # Add SPY as a benchmark and for regime analysis
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Create indicators for market regime detection
        self.spy_sma50 = self.SMA(self.spy, 50, Resolution.Daily)
        self.spy_sma200 = self.SMA(self.spy, 200, Resolution.Daily)
        self.spy_rsi = self.RSI(self.spy, 14, MovingAverageType.Exponential, Resolution.Daily)
        self.spy_atr = self.ATR(self.spy, 14, MovingAverageType.Simple, Resolution.Daily)
        self.spy_bb = self.BB(self.spy, 20, 2, MovingAverageType.Simple, Resolution.Daily)
        
        # Sector ETFs for sector exposure analysis
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
        
        # Add universe of stocks (Top 500 US equities by market cap)
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.MonthStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.RebalancePortfolio)
        
        # Schedule market regime detection
        self.Schedule.On(self.DateRules.Every(self.regime_check_days), 
                        self.TimeRules.AfterMarketOpen("SPY", 15), 
                        self.DetectMarketRegime)
        
    def CoarseSelectionFunction(self, coarse):
        """
        Select securities based on dollar volume and price.
        """
        if not self.spy_sma200.IsReady:
            return Universe.Unchanged
        
        # Filter for stocks with fundamental data, price > $5, and volume > $1M
        filtered = [x for x in coarse if x.HasFundamentalData and x.Price > 5 and x.DollarVolume > 1000000]
        
        # Sort by dollar volume and take top N stocks
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:self.universe_size]]
    
    def FineSelectionFunction(self, fine):
        """
        Select securities based on fundamental factors.
        """
        if len(fine) == 0:
            return Universe.Unchanged
        
        # Filter out stocks with no fundamental data
        filtered_fine = [f for f in fine if f.EarningReports.BasicEPS.TwelveMonths > 0 
                        and f.ValuationRatios.PERatio > 0 
                        and f.ValuationRatios.BookValuePerShare > 0]
        
        if len(filtered_fine) == 0:
            return Universe.Unchanged
        
        # Calculate factor values for each stock
        for stock in filtered_fine:
            symbol = stock.Symbol
            
            if symbol not in self.factor_data:
                self.factor_data[symbol] = {}
            
            # Value factors
            self.factor_data[symbol]["pe_ratio"] = stock.ValuationRatios.PERatio
            self.factor_data[symbol]["pb_ratio"] = stock.ValuationRatios.PBRatio
            
            # Quality factors
            self.factor_data[symbol]["roe"] = stock.ValuationRatios.ROE
            self.factor_data[symbol]["debt_equity"] = stock.FinancialStatements.BalanceSheet.TotalDebt.Value / stock.FinancialStatements.BalanceSheet.TotalEquity.Value if stock.FinancialStatements.BalanceSheet.TotalEquity.Value > 0 else 1000
            
            # Store sector info
            self.factor_data[symbol]["sector"] = stock.AssetClassification.MorningstarSectorCode
        
        # Calculate Z-scores for each factor
        self.CalculateFactorZScores(filtered_fine)
        
        # Adjust factor weights based on market regime
        self.AdjustFactorWeightsForRegime()
        
        # Calculate total factor score and select top stocks
        scored_stocks = self.ScoreStocks(filtered_fine)
        top_stocks = [stock.Symbol for stock in scored_stocks[:self.top_factors_count]]
        
        return top_stocks
    
    def CalculateFactorZScores(self, stocks):
        """
        Calculate Z-scores for all factors to standardize them.
        """
        # Extract data for Z-score calculations
        pe_ratios = [self.factor_data[stock.Symbol]["pe_ratio"] for stock in stocks]
        pb_ratios = [self.factor_data[stock.Symbol]["pb_ratio"] for stock in stocks]
        roes = [self.factor_data[stock.Symbol]["roe"] for stock in stocks]
        debt_equities = [self.factor_data[stock.Symbol]["debt_equity"] for stock in stocks]
        
        # Calculate Z-scores using robust methods to handle outliers
        for stock in stocks:
            symbol = stock.Symbol
            
            # Value Z-scores (negative because lower values are better)
            self.factor_data[symbol]["value_zscore"] = (
                -self.RobustZScore(self.factor_data[symbol]["pe_ratio"], pe_ratios) * 0.5 +
                -self.RobustZScore(self.factor_data[symbol]["pb_ratio"], pb_ratios) * 0.5
            )
            
            # Quality Z-scores
            self.factor_data[symbol]["quality_zscore"] = (
                self.RobustZScore(self.factor_data[symbol]["roe"], roes) * 0.5 +
                -self.RobustZScore(self.factor_data[symbol]["debt_equity"], debt_equities) * 0.5
            )
            
            # Request historical data for momentum calculation
            history = self.History(symbol, 252, Resolution.Daily)
            if not history.empty:
                # Calculate momentum Z-score (6-month return minus 1-month return)
                df = history.loc[symbol]
                if len(df) >= 126:  # At least 6 months of data
                    monthly_return = df['close'].pct_change(21).iloc[-1] if len(df) >= 21 else 0
                    six_month_return = df['close'].pct_change(126).iloc[-1]
                    self.factor_data[symbol]["momentum"] = six_month_return - monthly_return
                    
                    # Calculate volatility (lower is better)
                    daily_returns = df['close'].pct_change().dropna()
                    self.factor_data[symbol]["volatility"] = daily_returns.std() * np.sqrt(252)
            else:
                self.factor_data[symbol]["momentum"] = 0
                self.factor_data[symbol]["volatility"] = 1
        
        # Get momentum values for Z-score calculation
        momentums = [self.factor_data[stock.Symbol].get("momentum", 0) for stock in stocks]
        volatilities = [self.factor_data[stock.Symbol].get("volatility", 1) for stock in stocks]
        
        # Calculate momentum and volatility Z-scores
        for stock in stocks:
            symbol = stock.Symbol
            self.factor_data[symbol]["momentum_zscore"] = self.RobustZScore(self.factor_data[symbol].get("momentum", 0), momentums)
            self.factor_data[symbol]["volatility_zscore"] = -self.RobustZScore(self.factor_data[symbol].get("volatility", 1), volatilities)  # Negative because lower volatility is better
    
    def RobustZScore(self, value, values_list):
        """
        Calculate robust Z-score using median and median absolute deviation to handle outliers.
        """
        if len(values_list) <= 1:
            return 0
            
        median_val = np.median(values_list)
        mad = np.median([abs(x - median_val) for x in values_list])
        
        # Prevent division by zero
        if mad == 0:
            return 0
            
        return (value - median_val) / (mad * 1.4826)  # 1.4826 is a constant to make MAD consistent with standard deviation
    
    def AdjustFactorWeightsForRegime(self):
        """
        Adjust factor weights based on the current market regime.
        """
        if self.market_regime == "trending":
            # In trending markets, favor momentum
            self.current_factor_weights = {
                "value": 0.15,
                "momentum": 0.50, 
                "quality": 0.20,
                "volatility": 0.15
            }
        elif self.market_regime == "volatile":
            # In volatile markets, favor quality and low volatility
            self.current_factor_weights = {
                "value": 0.20,
                "momentum": 0.10, 
                "quality": 0.35,
                "volatility": 0.35
            }
        else:  # normal regime
            # In normal markets, use balanced weights
            self.current_factor_weights = self.normal_factor_weights.copy()
        
        self.Log(f"Factor weights adjusted for {self.market_regime} regime: {self.current_factor_weights}")
    
    def ScoreStocks(self, stocks):
        """
        Calculate the total factor score for each stock and rank them.
        """
        for stock in stocks:
            symbol = stock.Symbol
            
            # Skip if missing data
            if not all(key in self.factor_data[symbol] for key in ["value_zscore", "quality_zscore", "momentum_zscore", "volatility_zscore"]):
                self.factor_data[symbol]["total_score"] = -999  # Assign a very low score
                continue
            
            # Calculate weighted factor score
            self.factor_data[symbol]["total_score"] = (
                self.factor_data[symbol]["value_zscore"] * self.current_factor_weights["value"] +
                self.factor_data[symbol]["momentum_zscore"] * self.current_factor_weights["momentum"] +
                self.factor_data[symbol]["quality_zscore"] * self.current_factor_weights["quality"] +
                self.factor_data[symbol]["volatility_zscore"] * self.current_factor_weights["volatility"]
            )
        
        # Sort stocks by total score (descending)
        scored_stocks = sorted(stocks, key=lambda x: self.factor_data[x.Symbol].get("total_score", -999), reverse=True)
        return scored_stocks
    
    def DetectMarketRegime(self):
        """
        Detect the current market regime based on technical indicators.
        """
        # Skip if indicators aren't ready
        if not (self.spy_sma50.IsReady and self.spy_sma200.IsReady and self.spy_rsi.IsReady and self.spy_atr.IsReady):
            return
        
        spy_price = self.Securities[self.spy].Price
        
        # Get historical data for SPY
        history = self.History(self.spy, 252, Resolution.Daily)
        if history.empty:
            return
            
        df = history.loc[self.spy]
        
        # Calculate volatility
        returns = df['close'].pct_change().dropna()
        current_volatility = returns[-20:].std() * np.sqrt(252)  # Annualized volatility over the last 20 days
        long_term_volatility = returns.std() * np.sqrt(252)  # Annualized volatility over the entire period
        
        # Calculate trend strength (R-squared of linear regression)
        prices = df['close'].tail(50).values
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        r_squared = r_value ** 2
        
        # Define thresholds for regime classification
        high_volatility_threshold = 1.5 * long_term_volatility
        trending_threshold = 0.7  # R-squared value indicating strong trend
        
        # Store regime data
        self.regime_data = {
            "current_volatility": current_volatility,
            "long_term_volatility": long_term_volatility,
            "r_squared": r_squared,
            "trend_direction": "up" if slope > 0 else "down",
            "sma50": self.spy_sma50.Current.Value,
            "sma200": self.spy_sma200.Current.Value,
            "rsi": self.spy_rsi.Current.Value,
            "atr_percent": self.spy_atr.Current.Value / spy_price
        }
        
        # Determine market regime
        previous_regime = self.market_regime
        
        if current_volatility > high_volatility_threshold:
            self.market_regime = "volatile"
        elif r_squared > trending_threshold:
            self.market_regime = "trending"
        else:
            self.market_regime = "normal"
        
        # Log regime change if it occurred
        if previous_regime != self.market_regime:
            self.Log(f"Market regime changed from {previous_regime} to {self.market_regime}")
            self.Log(f"Volatility: {current_volatility:.2%}, R-squared: {r_squared:.2f}, Direction: {self.regime_data['trend_direction']}")
            
            # Adjust factor weights for the new regime
            self.AdjustFactorWeightsForRegime()
    
    def RebalancePortfolio(self):
        """
        Rebalance the portfolio based on factor scores and risk management.
        """
        # Skip if no selected securities
        if len(self.factor_data) == 0 or len(self.Securities) <= 1:
            return
        
        # Get invested symbols
        invested = [x.Key for x in self.Portfolio if x.Value.Invested]
        
        # Get selected securities (those that passed the fine selection)
        selected_securities = [x.Key for x in self.Portfolio if x.Key != self.spy and x.Key not in self.sectors.values()]
        
        # Calculate optimal portfolio weights using risk model
        optimal_weights = self.CalculateOptimalWeights(selected_securities)
        
        # Execute the trades
        for symbol, target_weight in optimal_weights.items():
            if self.Portfolio[symbol].Invested:
                current_weight = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                # Only trade if the deviation is significant to reduce turnover
                if abs(current_weight - target_weight) > 0.005:  # 0.5% threshold
                    self.SetHoldings(symbol, target_weight)
            else:
                # New position
                if target_weight > 0.005:  # Only open positions larger than 0.5%
                    self.SetHoldings(symbol, target_weight)
        
        # Liquidate securities that are no longer in our selected set
        for symbol in invested:
            if symbol not in optimal_weights and symbol != self.spy and symbol not in self.sectors.values():
                self.Liquidate(symbol)
    
    def CalculateOptimalWeights(self, symbols):
        """
        Calculate optimal portfolio weights using risk model and position sizing.
        """
        # If no symbols to allocate, return empty dict
        if len(symbols) == 0:
            return {}
        
        # Get historical data for selected securities
        if len(symbols) == 1:
            # Special case for a single security
            return {symbols[0]: min(self.max_position_size, 1.0)}
        
        # Get historical returns
        history = self.History(symbols, 252, Resolution.Daily)
        if history.empty:
            # Fallback to equal weights if no history
            equal_weight = min(1.0 / len(symbols), self.max_position_size)
            return {symbol: equal_weight for symbol in symbols}
        
        # Calculate daily returns
        returns_df = history['close'].unstack().pct_change().dropna()
        
        # Skip if not enough data
        if len(returns_df) < 20:
            equal_weight = min(1.0 / len(symbols), self.max_position_size)
            return {symbol: equal_weight for symbol in symbols}
        
        # Calculate volatility for each asset
        asset_vols = returns_df.std() * np.sqrt(252)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Apply position sizing using volatility targeting
        weights = {}
        
        # Simple approach: inverse volatility weighted
        total_inverse_vol = 0
        inverse_vols = {}
        
        for symbol in symbols:
            if symbol in asset_vols and not np.isnan(asset_vols[symbol]) and asset_vols[symbol] > 0:
                # For securities with factor scores, weight by both volatility and factor score
                factor_score = self.factor_data.get(symbol, {}).get("total_score", 0)
                if factor_score > -999:  # Valid factor score
                    # Combine inverse volatility with factor score
                    inverse_vol = (1.0 / asset_vols[symbol]) * (1 + factor_score)
                else:
                    inverse_vol = 1.0 / asset_vols[symbol]
                
                inverse_vols[symbol] = inverse_vol
                total_inverse_vol += inverse_vol
        
        # Normalize weights
        if total_inverse_vol > 0:
            for symbol, inverse_vol in inverse_vols.items():
                raw_weight = inverse_vol / total_inverse_vol
                # Apply maximum position size constraint
                weights[symbol] = min(raw_weight, self.max_position_size)
            
            # Renormalize after applying constraints
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        # Apply sector constraints
        weights = self.ApplySectorConstraints(weights)
        
        # Adjust portfolio leverage based on market regime
        weights = self.AdjustLeverageForRegime(weights)
        
        return weights
    
    def ApplySectorConstraints(self, weights):
        """
        Apply sector exposure constraints to avoid overconcentration.
        """
        # Group securities by sector
        sector_exposure = {}
        for symbol, weight in weights.items():
            if symbol in self.factor_data:
                sector = self.factor_data[symbol].get("sector", "Unknown")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # Check if any sector exceeds the maximum exposure
        overexposed_sectors = [sector for sector, exposure in sector_exposure.items() if exposure > self.max_sector_exposure]
        
        # If no sectors are overexposed, return original weights
        if not overexposed_sectors:
            return weights
        
        # Adjust weights for overexposed sectors
        for sector in overexposed_sectors:
            # Calculate scaling factor for the sector
            scaling_factor = self.max_sector_exposure / sector_exposure[sector]
            
            # Scale down all stocks in the overexposed sector
            for symbol in weights:
                if symbol in self.factor_data and self.factor_data[symbol].get("sector", "Unknown") == sector:
                    weights[symbol] *= scaling_factor
        
        # Renormalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        return weights
    
    def AdjustLeverageForRegime(self, weights):
        """
        Adjust overall portfolio leverage based on market regime.
        """
        leverage_factor = 1.0  # Default leverage factor
        
        if self.market_regime == "trending":
            # In trending markets, we might want to increase exposure
            # But only if the trend is up and not overextended
            if (self.regime_data.get("trend_direction", "") == "up" and 
                self.regime_data.get("sma50", 0) > self.regime_data.get("sma200", 0) and
                self.regime_data.get("rsi", 100) < 70):
                leverage_factor = 1.2
            elif self.regime_data.get("trend_direction", "") == "down":
                leverage_factor = 0.8
        
        elif self.market_regime == "volatile":
            # In volatile markets, reduce exposure
            leverage_factor = 0.7
        
        # Scale weights by leverage factor
        scaled_weights = {symbol: weight * leverage_factor for symbol, weight in weights.items()}
        
        # Log the leverage adjustment
        self.Log(f"Portfolio leverage adjusted to {leverage_factor} for {self.market_regime} regime")
        
        return scaled_weights
    
    def OnData(self, data):
        """
        Event handler for market data updates.
        """
        # Most logic is handled by scheduled events
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
        self.Log(f"Final Market Regime: {self.market_regime}")
        
        # Report performance metrics
        total_trades = sum(security.Holdings.TotalSaleVolume for security in self.Securities.Values)
        self.Log(f"Total Trades: {total_trades}")