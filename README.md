# AI Sentiment-Based Trading Algorithm for QuantConnect

This repository contains a sophisticated algorithmic trading strategy designed for QuantConnect that uses AI sentiment analysis to enhance trading decisions. The algorithm is adaptive to changing market conditions and incorporates advanced risk management.

## Features

- **AI Sentiment Analysis**: Uses reasoning AI models (like DeepSeek R1) to analyze market sentiment
- **Adaptive Market Regime Detection**: Automatically detects market regimes (normal, volatile, trending) and adjusts strategy parameters
- **Dynamic Risk Management**: Implements advanced position sizing and stop-loss strategies
- **Technical Analysis Integration**: Combines sentiment with technical indicators for robust signal generation
- **Portfolio Optimization**: Rebalances the portfolio based on ranking securities by combined scores

## Getting Started with QuantConnect

1. Create an account on [QuantConnect](https://www.quantconnect.com/)
2. Create a new algorithm in the QuantConnect IDE
3. Copy the contents of `AI_Sentiment_Trading_Algorithm.py` into your new algorithm
4. Set up your API keys for sentiment analysis in the algorithm parameters

## Setting Up AI Sentiment Analysis

The algorithm is designed to work with external AI sentiment analysis APIs. Here are several options for implementing the sentiment analysis component:

### Option 1: Using a Hosted DeepSeek R1 API Service

1. Subscribe to a hosted DeepSeek R1 API service (or similar reasoning AI API)
2. Set up your API keys in the QuantConnect algorithm parameters:
   - `SentimentAPIKey`: Your API key
   - `SentimentAPIEndpoint`: The API endpoint URL

### Option 2: Creating Your Own AI Sentiment Analysis Service

If you want to create your own service:

1. Set up a server with DeepSeek R1 (or another reasoning AI model) running
2. Create an API that:
   - Takes a stock ticker as input
   - Collects recent news, social media data, earnings reports, etc.
   - Processes this data through the reasoning AI model to generate a sentiment score
   - Returns a score between -1 (highly negative) and 1 (highly positive)
3. Host this API service and use its endpoint in the algorithm

### Option 3: Using Pre-Computed Sentiment Data

For simpler implementation:

1. Subscribe to a financial news sentiment data provider (like RavenPack, Sentifi, or MarketPsych)
2. Modify the `GetAISentiment` function to pull data from these services
3. Adapt the algorithm to use the sentiment data format provided by your chosen service

## Algorithm Parameters

You can customize these parameters in the algorithm:

- `max_position_size`: Maximum position size as a percentage of portfolio (default: 5%)
- `stop_loss_pct`: Stop loss percentage (default: 5%)
- `take_profit_pct`: Take profit percentage (default: 15%)
- `holding_period`: Default holding period in days (default: 5)
- `sentiment_threshold`: Sentiment score threshold for trading (default: 0.6)
- `rebalance_frequency`: Rebalance portfolio every n days (default: 7)

## Market Regime Adaptation

The algorithm automatically detects market regimes and adjusts its parameters:

- **Normal Market**: Uses default parameters
- **Volatile Market**: Reduces position sizes, shortens holding periods, tightens stops
- **Trending Market**: Increases position sizes, extends holding periods, widens stops

## Implementation Details

### AI Sentiment Integration

The key to this algorithm is the integration of AI sentiment analysis. The code includes a placeholder `GetAISentiment` function where you should implement your specific AI sentiment analysis integration.

The function is expected to return a sentiment score between -1 (highly negative) and 1 (highly positive). This score is then used in combination with technical indicators to generate trading signals.

#### Using DeepSeek R1 for Sentiment Analysis

[DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-Coder) is a powerful reasoning AI that can analyze text and extract sentiment. To use it effectively for trading:

1. Collect relevant data sources:
   - Financial news articles related to the stock
   - Social media mentions (Twitter, Reddit, etc.)
   - Earnings call transcripts
   - Analyst reports

2. Prompt the model to analyze sentiment, for example:
   ```
   Analyze the sentiment of the following content about [TICKER] and rate it on a scale from -1 (extremely negative) to 1 (extremely positive), where 0 is neutral:
   
   [CONTENT]
   
   Provide a single numeric score that represents the overall sentiment.
   ```

3. Parse the response to extract the numerical sentiment score

4. Optionally, aggregate multiple sentiment scores from different sources using a weighted average

### Key Components

1. **Universe Selection**: Selects top 50 liquid stocks by dollar volume
2. **Indicator Initialization**: Sets up technical indicators (SMA, RSI, Bollinger Bands, MACD)
3. **Sentiment Analysis**: Updates and caches sentiment data for all active securities
4. **Position Management**: Manages existing positions with trailing stops and take profit orders
5. **Portfolio Rebalancing**: Ranks securities and adjusts portfolio based on combined scores
6. **Market Regime Detection**: Analyzes SPY behavior to determine market regime and adapt parameters

## Customization and Extension

You can extend this algorithm in several ways:

1. **Additional Data Sources**: Incorporate alternative data sources like insider trading, options flow, etc.
2. **Enhanced AI Models**: Experiment with different reasoning AI models or ensemble methods
3. **Sector Rotation**: Add sector-based rotation strategies based on sentiment trends
4. **Option Strategies**: Incorporate options for hedging or enhanced returns
5. **ML Prediction Models**: Add machine learning models to predict price movements

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Disclaimer

This algorithm is provided for educational and research purposes only. Trading involves significant risk of loss and is not suitable for all investors. Past performance is not indicative of future results.