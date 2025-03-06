# Volatility Regime-Based Risk Parity Strategy
#
# This algorithm implements a sophisticated asset allocation strategy that:
# 1. Dynamically detects different market volatility regimes
# 2. Implements risk parity portfolio construction across asset classes
# 3. Uses volatility targeting for consistent risk exposure
# 4. Applies dynamic tactical tilts based on momentum and trend signals
#
# Designed for QuantConnect platform by Zachary Amador

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import scipy.stats as stats

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import *

class VolatilityRegimeRiskParity(QCAlgorithm):
    
    def Initialize(self):
        """Initialize the algorithm with required settings and data."""
        # Set start date and cash
        self.SetStartDate(2018, 1, 1)  # Start date
        self.SetCash(1000000)           # Initial capital
        
        # Set the benchmark
        self.SetBenchmark("SPY")
        
        # Set brokerage model for realistic trading
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        
        # Risk parameters
        self.target_volatility = 0.10  # Target annualized portfolio volatility (10%)
        self.volatility_lookback = 63  # Days to use for volatility calculation
        self.regime_lookback = 252     # Days to use for regime detection
        self.rebalance_threshold = 0.1 # Rebalance when weights deviate by more than 10%
        
        # Risk parity parameters
        self.risk_contribution_target = {}  # Target risk contribution for each asset class
        
        # Momentum and trend parameters
        self.momentum_lookback = 126   # Momentum lookback period (6 months)
        self.trend_lookback = 200      # Trend lookback period
        self.volatility_threshold = 0.15  # Threshold for high volatility regime
        
        # ETFs for different asset classes
        self.assets = {
            # U.S. Equities
            "SPY": {"class": "US_Equity", "weight": 0.20},
            
            # International Equities
            "VEA": {"class": "Intl_Equity", "weight": 0.15},
            "VWO": {"class": "Emerging_Equity", "weight": 0.10},
            
            # Fixed Income
            "TLT": {"class": "Long_Treasury", "weight": 0.15},
            "IEF": {"class": "Mid_Treasury", "weight": 0.15},
            "SHY": {"class": "Short_Treasury", "weight": 0.05},
            "LQD": {"class": "Corp_Bond", "weight": 0.05},
            
            # Alternative Assets
            "GLD": {"class": "Gold", "weight": 0.05},
            "GSG": {"class": "Commodities", "weight": 0.05},
            "VNQ": {"class": "Real_Estate", "weight": 0.05}
        }
        
        # Add ETFs to the universe
        for symbol, info in self.assets.items():
            self.assets[symbol]["symbol"] = self.AddEquity(symbol, Resolution.Daily).Symbol
            
            # Create SMA indicator for trend following
            self.assets[symbol]["sma200"] = self.SMA(self.assets[symbol]["symbol"], 200, Resolution.Daily)
        
        # Initialize dictionaries for data storage
        self.asset_prices = {}         # Historical prices
        self.asset_returns = {}        # Historical returns
        self.asset_volatilities = {}   # Historical volatilities
        self.asset_weights = {}        # Current portfolio weights
        self.target_weights = {}       # Target portfolio weights
        self.regime = "normal"         # Current volatility regime (normal, high, low)
        
        # Initialize VIX tracker for regime assessment
        self.vix = self.AddEquity("VIXY", Resolution.Daily).Symbol  # VIX ETF as proxy
        self.vixma20 = self.SMA(self.vix, 20, Resolution.Daily)     # 20-day moving average of VIX
        self.vixma50 = self.SMA(self.vix, 50, Resolution.Daily)     # 50-day moving average of VIX
        
        # Schedule rebalancing monthly
        self.Schedule.On(self.DateRules.MonthStart("SPY"), 
                        self.TimeRules.AfterMarketOpen("SPY", 30), 
                        self.RebalancePortfolio)
        
        # Schedule regime detection weekly
        self.Schedule.On(self.DateRules.WeekStart("SPY"), 
                       self.TimeRules.AfterMarketOpen("SPY", 15), 
                       self.DetectRegime)
        
        # Schedule daily risk management check
        self.Schedule.On(self.DateRules.EveryDay("SPY"), 
                       self.TimeRules.AfterMarketOpen("SPY", 10), 
                       self.RiskManagement)
        
        # Initialize risk contribution targets based on asset class
        self.InitializeRiskContributionTargets()
    
    def InitializeRiskContributionTargets(self):
        """
        Initialize the target risk contribution for each asset class.
        These will vary according to volatility regime.
        """
        # Group assets by class
        asset_classes = {}
        for symbol, info in self.assets.items():
            asset_class = info["class"]
            if asset_class not in asset_classes:
                asset_classes[asset_class] = []
            asset_classes[asset_class].append(symbol)
        
        # Normal regime risk contributions (balanced across asset classes)
        self.regime_risk_contributions = {
            "normal": {
                "US_Equity": 0.20,
                "Intl_Equity": 0.15,
                "Emerging_Equity": 0.10,
                "Long_Treasury": 0.15,
                "Mid_Treasury": 0.15,
                "Short_Treasury": 0.05,
                "Corp_Bond": 0.05,
                "Gold": 0.05,
                "Commodities": 0.05,
                "Real_Estate": 0.05
            },
            "low_vol": {
                "US_Equity": 0.25,
                "Intl_Equity": 0.20,
                "Emerging_Equity": 0.15,
                "Long_Treasury": 0.10,
                "Mid_Treasury": 0.10,
                "Short_Treasury": 0.03,
                "Corp_Bond": 0.05,
                "Gold": 0.04,
                "Commodities": 0.05,
                "Real_Estate": 0.03
            },
            "high_vol": {
                "US_Equity": 0.12,
                "Intl_Equity": 0.08,
                "Emerging_Equity": 0.05,
                "Long_Treasury": 0.23,
                "Mid_Treasury": 0.20,
                "Short_Treasury": 0.10,
                "Corp_Bond": 0.05,
                "Gold": 0.10,
                "Commodities": 0.04,
                "Real_Estate": 0.03
            }
        }
        
        # Set initial risk contribution targets based on normal regime
        self.risk_contribution_target = self.regime_risk_contributions["normal"].copy()
    
    def DetectRegime(self):
        """
        Detect the current market volatility regime.
        """
        # Skip if VIX data is not ready
        if not (self.vixma20.IsReady and self.vixma50.IsReady):
            return
        
        # Get historical data for SPY (as market proxy)
        spy_symbol = self.assets["SPY"]["symbol"]
        history = self.History(spy_symbol, self.regime_lookback, Resolution.Daily)
        if history.empty:
            return
            
        # Calculate realized volatility of SPY
        spy_returns = history.loc[spy_symbol]['close'].pct_change().dropna()
        realized_vol = spy_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Get current VIX data
        vix_value = self.Securities[self.vix].Price
        vix_ma20 = self.vixma20.Current.Value
        vix_ma50 = self.vixma50.Current.Value
        
        # Get historical VIX data
        vix_history = self.History(self.vix, 252, Resolution.Daily)
        if not vix_history.empty:
            vix_series = vix_history.loc[self.vix]['close']
            vix_percentile = stats.percentileofscore(vix_series, vix_value)
        else:
            vix_percentile = 50  # Default to median if no history
        
        # Determine regime based on VIX and realized volatility
        old_regime = self.regime
        
        # High volatility regime
        if (vix_value > vix_ma50 * 1.2 and vix_percentile > 80) or realized_vol > self.volatility_threshold:
            self.regime = "high_vol"
        # Low volatility regime
        elif (vix_value < vix_ma50 * 0.8 and vix_percentile < 20) or realized_vol < self.volatility_threshold * 0.5:
            self.regime = "low_vol"
        # Normal volatility regime
        else:
            self.regime = "normal"
        
        # Log regime change if it occurred
        if old_regime != self.regime:
            self.Log(f"Volatility regime changed from {old_regime} to {self.regime}")
            self.Log(f"Realized volatility: {realized_vol:.2%}, VIX: {vix_value:.2f}, VIX percentile: {vix_percentile:.1f}%")
            
            # Update risk contribution targets based on new regime
            self.risk_contribution_target = self.regime_risk_contributions[self.regime].copy()
    
    def RiskManagement(self):
        """
        Daily risk management check to ensure the portfolio stays within target risk parameters.
        """
        # Check if we have position data
        if not self.asset_weights:
            return
        
        # Check if current portfolio weights deviate significantly from targets
        needs_rebalance = False
        
        # Calculate current weights based on current prices
        current_weights = {}
        portfolio_value = self.Portfolio.TotalPortfolioValue
        
        for symbol, info in self.assets.items():
            etf_symbol = info["symbol"]
            if self.Portfolio[etf_symbol].Invested:
                current_weights[symbol] = self.Portfolio[etf_symbol].HoldingsValue / portfolio_value
            else:
                current_weights[symbol] = 0
        
        # Check for significant deviation from target weights
        for symbol, target in self.target_weights.items():
            current = current_weights.get(symbol, 0)
            if abs(current - target) > self.rebalance_threshold:
                needs_rebalance = True
                break
        
        # Rebalance if necessary
        if needs_rebalance:
            self.Log("Risk check: Portfolio weights have deviated significantly from target. Rebalancing...")
            self.RebalancePortfolio()
        
        # Calculate current portfolio volatility
        if not self.asset_returns:
            return
            
        # Calculate weighted sum of asset volatilities
        portfolio_var = 0
        for symbol1, weight1 in current_weights.items():
            if symbol1 not in self.asset_volatilities or weight1 == 0:
                continue
                
            for symbol2, weight2 in current_weights.items():
                if symbol2 not in self.asset_volatilities or weight2 == 0:
                    continue
                    
                # Get correlation between assets
                if symbol1 in self.asset_returns and symbol2 in self.asset_returns:
                    correlation = self.asset_returns[symbol1].corr(self.asset_returns[symbol2])
                else:
                    correlation = 1 if symbol1 == symbol2 else 0
                
                # Add to portfolio variance
                contribution = weight1 * weight2 * self.asset_volatilities[symbol1] * self.asset_volatilities[symbol2] * correlation
                portfolio_var += contribution
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Check if portfolio volatility is significantly different from target
        vol_ratio = portfolio_vol / self.target_volatility
        
        if vol_ratio > 1.2:  # Volatility is too high
            self.Log(f"Risk check: Portfolio volatility ({portfolio_vol:.2%}) exceeds target ({self.target_volatility:.2%}). Reducing exposure.")
            
            # Scale down all positions
            scaling_factor = self.target_volatility / portfolio_vol
            for symbol, info in self.assets.items():
                etf_symbol = info["symbol"]
                if self.Portfolio[etf_symbol].Invested:
                    current_weight = self.Portfolio[etf_symbol].HoldingsValue / portfolio_value
                    new_weight = current_weight * scaling_factor
                    self.SetHoldings(etf_symbol, new_weight)
        
        elif vol_ratio < 0.8:  # Volatility is too low
            self.Log(f"Risk check: Portfolio volatility ({portfolio_vol:.2%}) is below target ({self.target_volatility:.2%}). Increasing exposure.")
            
            # Scale up all positions
            scaling_factor = self.target_volatility / portfolio_vol
            for symbol, info in self.assets.items():
                etf_symbol = info["symbol"]
                if self.Portfolio[etf_symbol].Invested:
                    current_weight = self.Portfolio[etf_symbol].HoldingsValue / portfolio_value
                    new_weight = current_weight * scaling_factor
                    self.SetHoldings(etf_symbol, new_weight)
    
    def RebalancePortfolio(self):
        """
        Rebalance the portfolio using risk parity principles adjusted by regime.
        """
        # Update historical prices and returns
        self.UpdateHistoricalData()
        
        # Skip if we don't have enough data
        if not self.asset_returns:
            return
        
        # Calculate volatilities
        for symbol, returns in self.asset_returns.items():
            self.asset_volatilities[symbol] = returns.std() * np.sqrt(252)  # Annualize daily vol
        
        # Log volatilities for each asset
        self.Log("Asset volatilities:")
        for symbol, vol in self.asset_volatilities.items():
            self.Log(f"{symbol}: {vol:.2%}")
        
        # Calculate risk parity weights
        risk_parity_weights = self.CalculateRiskParityWeights()
        
        # Apply momentum and trend overlays
        final_weights = self.ApplyTacticalOverlays(risk_parity_weights)
        
        # Store the target weights
        self.target_weights = final_weights
        
        # Apply volatility targeting
        volatility_targeted_weights = self.ApplyVolatilityTargeting(final_weights)
        
        # Execute trades to achieve target weights
        for symbol, weight in volatility_targeted_weights.items():
            etf_symbol = self.assets[symbol]["symbol"]
            self.SetHoldings(etf_symbol, weight)
        
        # Log the new portfolio allocation
        self.Log(f"Portfolio rebalanced. Regime: {self.regime}")
        for symbol, weight in volatility_targeted_weights.items():
            self.Log(f"{symbol}: {weight:.2%}")
    
    def UpdateHistoricalData(self):
        """
        Update historical price and return data for all assets.
        """
        # Get historical data
        symbols = [info["symbol"] for _, info in self.assets.items()]
        history = self.History(symbols, self.regime_lookback, Resolution.Daily)
        if history.empty:
            return
        
        # Process historical data for each asset
        for symbol_str, info in self.assets.items():
            symbol = info["symbol"]
            if symbol in history.index.levels[0]:
                # Get historical prices
                prices = history.loc[symbol]['close']
                self.asset_prices[symbol_str] = prices
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                self.asset_returns[symbol_str] = returns
    
    def CalculateRiskParityWeights(self):
        """
        Calculate risk parity weights based on volatility and correlations.
        """
        # Get list of assets with valid volatility data
        valid_assets = [symbol for symbol in self.assets.keys() if symbol in self.asset_volatilities]
        
        if len(valid_assets) == 0:
            return {symbol: 1.0 / len(self.assets) for symbol in self.assets.keys()}
        
        # Create covariance matrix
        returns_df = pd.DataFrame({symbol: self.asset_returns[symbol] for symbol in valid_assets if symbol in self.asset_returns})
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        
        # Create initial weights
        init_weights = np.array([1.0 / len(valid_assets)] * len(valid_assets))
        
        # Target risk contributions for valid assets
        asset_risk_targets = np.array([self.risk_contribution_target.get(self.assets[symbol]["class"], 1.0/len(valid_assets)) for symbol in valid_assets])
        asset_risk_targets = asset_risk_targets / asset_risk_targets.sum()  # Normalize to sum to 1
        
        # Define the objective function to minimize (risk concentration)
        def objective(weights):
            weights = np.array(weights)
            
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Calculate portfolio variance and volatility
            port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            port_vol = np.sqrt(port_var)
            
            # Calculate marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            
            # Calculate risk contributions
            risk_contrib = np.multiply(weights, marginal_contrib)
            
            # Normalize risk contributions to sum to 1
            risk_contrib = risk_contrib / risk_contrib.sum()
            
            # Calculate squared error from target risk contributions
            error = np.sum((risk_contrib - asset_risk_targets) ** 2)
            
            return error
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})  # weights sum to 1
        
        # Bounds
        bounds = tuple((0.01, 0.4) for _ in valid_assets)  # Min 1%, max 40% per asset
        
        # Solve for the optimal weights
        result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Check if optimization was successful
        if result.success:
            # Normalize the weights to ensure they sum to 1
            optimized_weights = result.x / result.x.sum()
            
            # Create dictionary of weights
            weights_dict = {valid_assets[i]: optimized_weights[i] for i in range(len(valid_assets))}
            
            # Add any missing assets with zero weight
            for asset in self.assets.keys():
                if asset not in weights_dict:
                    weights_dict[asset] = 0
                    
            return weights_dict
        else:
            self.Log(f"Optimization failed: {result.message}")
            # Fallback to inverse volatility weighting
            return self.CalculateInverseVolatilityWeights()
    
    def CalculateInverseVolatilityWeights(self):
        """
        Calculate weights based on inverse volatility (fallback method).
        """
        # Calculate inverse volatility for each asset
        inv_vol = {}
        total_inv_vol = 0
        
        for symbol, info in self.assets.items():
            if symbol in self.asset_volatilities and self.asset_volatilities[symbol] > 0:
                asset_class = info["class"]
                target_risk = self.risk_contribution_target.get(asset_class, 1.0/len(self.assets))
                
                # Weight by inverse volatility and target risk
                inv_vol[symbol] = target_risk / self.asset_volatilities[symbol]
                total_inv_vol += inv_vol[symbol]
        
        # Normalize weights to sum to 1
        weights = {}
        for symbol in self.assets.keys():
            if symbol in inv_vol and total_inv_vol > 0:
                weights[symbol] = inv_vol[symbol] / total_inv_vol
            else:
                weights[symbol] = 0
        
        return weights
    
    def ApplyTacticalOverlays(self, base_weights):
        """
        Apply momentum and trend overlays to adjust the base weights.
        """
        # Calculate momentum scores
        momentum_scores = {}
        for symbol, info in self.assets.items():
            if symbol in self.asset_prices:
                prices = self.asset_prices[symbol]
                if len(prices) > self.momentum_lookback:
                    # Calculate 6-month return
                    momentum = prices.iloc[-1] / prices.iloc[-self.momentum_lookback] - 1
                    momentum_scores[symbol] = momentum
        
        # Normalize momentum scores
        if momentum_scores:
            min_score = min(momentum_scores.values())
            max_score = max(momentum_scores.values())
            score_range = max_score - min_score
            
            normalized_scores = {}
            for symbol, score in momentum_scores.items():
                if score_range > 0:
                    normalized_scores[symbol] = (score - min_score) / score_range
                else:
                    normalized_scores[symbol] = 0.5  # Default if all scores are the same
        else:
            normalized_scores = {symbol: 0.5 for symbol in self.assets.keys()}
        
        # Apply trend overlay
        trend_overlay = {}
        for symbol, info in self.assets.items():
            # Check if price is above 200-day moving average
            if info["sma200"].IsReady and symbol in self.asset_prices:
                current_price = self.Securities[info["symbol"]].Price
                sma200 = info["sma200"].Current.Value
                
                # Trend factor: 1.2 if price > SMA200, 0.8 if price < SMA200
                trend_overlay[symbol] = 1.2 if current_price > sma200 else 0.8
            else:
                trend_overlay[symbol] = 1.0  # Neutral if not enough data
        
        # Combine base weights with tactical overlays
        final_weights = {}
        for symbol in self.assets.keys():
            base_weight = base_weights.get(symbol, 0)
            momentum_factor = normalized_scores.get(symbol, 0.5)
            trend_factor = trend_overlay.get(symbol, 1.0)
            
            # Adjust weight based on overlays
            adjusted_weight = base_weight * (0.8 + 0.4 * momentum_factor) * trend_factor
            final_weights[symbol] = adjusted_weight
        
        # Check regime-specific adjustments
        if self.regime == "high_vol":
            # In high volatility, reduce equity exposure further based on trend
            for symbol, info in self.assets.items():
                if info["class"] in ["US_Equity", "Intl_Equity", "Emerging_Equity", "Real_Estate"]:
                    if trend_overlay[symbol] < 1.0:  # Downtrend
                        final_weights[symbol] *= 0.75  # Further reduce weight
                    
        # Renormalize weights to sum to 1
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {symbol: weight / total_weight for symbol, weight in final_weights.items()}
        
        return final_weights
    
    def ApplyVolatilityTargeting(self, weights):
        """
        Scale weights to target a specific portfolio volatility.
        """
        # Create returns dataframe for portfolio volatility calculation
        portfolio_returns = pd.DataFrame()
        for symbol, weight in weights.items():
            if symbol in self.asset_returns and weight > 0:
                portfolio_returns[symbol] = self.asset_returns[symbol]
        
        if portfolio_returns.empty:
            return weights
        
        # Calculate covariance matrix
        cov_matrix = portfolio_returns.cov() * 252  # Annualized covariance
        
        # Extract weights for assets in the covariance matrix
        weight_vector = np.array([weights.get(symbol, 0) for symbol in portfolio_returns.columns])
        
        # Calculate portfolio variance
        port_var = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
        port_vol = np.sqrt(port_var)
        
        # Scale weights to target volatility
        if port_vol > 0:
            volatility_scalar = self.target_volatility / port_vol
            volatility_targeted_weights = {symbol: weight * volatility_scalar for symbol, weight in weights.items()}
        else:
            volatility_targeted_weights = weights
        
        # Log the portfolio volatility
        self.Log(f"Portfolio volatility: {port_vol:.2%}, Target: {self.target_volatility:.2%}, Scalar: {volatility_scalar:.2f}")
        
        return volatility_targeted_weights
    
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
        self.Log(f"Final Volatility Regime: {self.regime}")
        
        # Report asset allocations
        self.Log("Final Asset Allocations:")
        for symbol, info in self.assets.items():
            etf_symbol = info["symbol"]
            if self.Portfolio[etf_symbol].Invested:
                weight = self.Portfolio[etf_symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                self.Log(f"{symbol}: {weight:.2%}")