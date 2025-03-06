# Sentiment Analysis API Example Using Reasoning AI
#
# This script demonstrates how to create a simple API server that uses a reasoning AI model
# (like DeepSeek R1) to analyze sentiment for stocks. This can be hosted and used by the
# QuantConnect algorithm.

import os
import json
import requests
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import numpy as np

# Uncomment the appropriate AI model import based on which one you're using
# For DeepSeek R1 you might use their API or a huggingface client
# import deepseek
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Configuration
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")  # For accessing financial news
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")  # For social media data
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")  # If using HuggingFace models

# Cache for sentiment results to reduce API calls
sentiment_cache = {}
CACHE_EXPIRY = 3600  # Cache expiry in seconds (1 hour)

# Initialize the AI model
# For demo purposes, using HuggingFace pipeline
# In production, you'd use DeepSeek R1 or another reasoning model
sentiment_analyzer = pipeline(
    "text-classification", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def get_financial_news(ticker, days=3):
    """
    Get recent financial news for a ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to look back
        
    Returns:
        list: List of news articles
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Call news API (example using NewsAPI)
    url = f"https://newsapi.org/v2/everything?q={ticker}+stock&from={from_date}&to={to_date}&language=en&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract articles
        articles = []
        if 'articles' in data:
            for article in data['articles']:
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                })
        
        return articles
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def get_social_media_data(ticker, days=3):
    """
    Get social media mentions for a ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to look back
        
    Returns:
        list: List of social media posts
    """
    # This is a placeholder function
    # In a real implementation, you would use Twitter API, Reddit API, etc.
    
    # For demo purposes, returning empty list
    # In production, you'd fetch actual social media data
    return []

def analyze_sentiment_with_reasoner(texts, ticker):
    """
    Analyze sentiment using a reasoning AI model.
    
    Args:
        texts (list): List of text pieces to analyze
        ticker (str): Stock ticker for context
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    # This is where you'd use DeepSeek R1 or another reasoning AI
    # For this demo, we'll use a simpler sentiment analyzer
    
    if not texts:
        return 0  # Neutral sentiment if no texts
    
    # Concatenate all texts (with limits to avoid token issues)
    combined_text = " ".join([t[:500] for t in texts])[:5000]
    
    # Example for DeepSeek R1 API call (commented out)
    # prompt = f"""
    # Analyze the sentiment of the following content about {ticker} stock and rate it 
    # on a scale from -1 (extremely negative) to 1 (extremely positive), where 0 is neutral.
    # 
    # Content: {combined_text}
    # 
    # Provide your analysis in JSON format with a single 'score' field containing the numerical sentiment score.
    # """
    # 
    # response = deepseek.Completion.create(
    #    prompt=prompt,
    #    max_tokens=100
    # )
    # 
    # try:
    #    result = json.loads(response.choices[0].text)
    #    return result['score']
    # except:
    #    return 0  # Default to neutral on error
    
    # For this demo, using a simpler sentiment approach
    try:
        results = sentiment_analyzer(combined_text)
        
        # Map labels to scores
        if results[0]['label'] == 'POSITIVE':
            score = results[0]['score']  # Between 0 and 1
        else:
            score = -results[0]['score']  # Between -1 and 0
            
        return score
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0  # Default to neutral on error

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment():
    """
    API endpoint to get sentiment for a ticker.
    
    Query parameters:
    - ticker: Stock ticker symbol
    - days: Number of days to look back (default: 3)
    
    Returns:
        JSON with sentiment score and metadata
    """
    # Get query parameters
    ticker = request.args.get('ticker', '')
    days = int(request.args.get('days', '3'))
    
    if not ticker:
        return jsonify({'error': 'No ticker provided'}), 400
    
    # Check cache first
    cache_key = f"{ticker}_{days}"
    if cache_key in sentiment_cache:
        cache_entry = sentiment_cache[cache_key]
        if (datetime.now() - cache_entry['timestamp']).seconds < CACHE_EXPIRY:
            return jsonify(cache_entry['data'])
    
    # Get news and social media data
    news_articles = get_financial_news(ticker, days)
    social_posts = get_social_media_data(ticker, days)
    
    # Extract text for analysis
    texts = []
    for article in news_articles:
        if article['title']:
            texts.append(article['title'])
        if article['description']:
            texts.append(article['description'])
        if article['content']:
            texts.append(article['content'])
    
    for post in social_posts:
        texts.append(post['text'])
    
    # Analyze sentiment
    score = analyze_sentiment_with_reasoner(texts, ticker)
    
    # Prepare response
    result = {
        'ticker': ticker,
        'score': score,
        'sources': {
            'news_count': len(news_articles),
            'social_count': len(social_posts)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Cache the result
    sentiment_cache[cache_key] = {
        'data': result,
        'timestamp': datetime.now()
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port)