# Deploying the AI Sentiment Analysis API

This guide provides instructions on how to deploy the sentiment analysis API that powers the QuantConnect trading algorithm.

## Prerequisites

- A cloud service provider account (AWS, GCP, Azure, or similar)
- Basic knowledge of Docker and cloud deployment
- API keys for news sources (NewsAPI, Bloomberg, etc.)
- API keys for social media sources (Twitter, Reddit, etc.)
- Access to a reasoning AI model (DeepSeek R1 or similar)

## Option 1: Deploy with Docker

### 1. Set Up Environment Variables

Create a `.env` file with your API keys:

```
NEWS_API_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_api_key
HUGGINGFACE_API_KEY=your_huggingface_key
PORT=5000
```

### 2. Create a Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment_analysis_api.py .
COPY .env .

EXPOSE 5000

CMD ["python", "sentiment_analysis_api.py"]
```

### 3. Create requirements.txt

```
flask==2.0.1
requests==2.26.0
transformers==4.12.5
torch==1.10.0
numpy==1.21.4
python-dotenv==0.19.2
```

### 4. Build and Run the Docker Container

```bash
docker build -t sentiment-api .
docker run -p 5000:5000 --env-file .env sentiment-api
```

### 5. Deploy to a Cloud Provider

#### AWS Elastic Beanstalk

1. Install the EB CLI and initialize your application:
   ```bash
   pip install awsebcli
   eb init -p docker sentiment-api
   ```

2. Create an environment and deploy:
   ```bash
   eb create sentiment-api-env
   ```

3. Access your API at the provided URL

#### Google Cloud Run

1. Build and push your container to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/your-project/sentiment-api
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy sentiment-api \
     --image gcr.io/your-project/sentiment-api \
     --platform managed \
     --allow-unauthenticated
   ```

## Option 2: Deploy Using Serverless Functions

### AWS Lambda

1. Package your application for Lambda:
   ```bash
   pip install -r requirements.txt -t ./package
   cp sentiment_analysis_api.py ./package
   cd package && zip -r ../lambda_function.zip . && cd ..
   ```

2. Create a Lambda function using the AWS console or CLI:
   ```bash
   aws lambda create-function \
     --function-name sentiment-api \
     --runtime python3.9 \
     --handler sentiment_analysis_api.lambda_handler \
     --zip-file fileb://lambda_function.zip \
     --role your-lambda-execution-role
   ```

3. Set up API Gateway to expose your Lambda function

### Google Cloud Functions

1. Adapt your API for Cloud Functions (create a main.py file)
2. Deploy to Cloud Functions:
   ```bash
   gcloud functions deploy sentiment-api \
     --runtime python39 \
     --trigger-http \
     --allow-unauthenticated
   ```

## Option 3: Using DeepSeek R1 via Hugging Face API

If you prefer to use DeepSeek R1 through Hugging Face's API:

1. Create an account on [Hugging Face](https://huggingface.co/)
2. Get an API key from your profile settings
3. Modify the `analyze_sentiment_with_reasoner` function in `sentiment_analysis_api.py`:

```python
def analyze_sentiment_with_reasoner(texts, ticker):
    """
    Analyze sentiment using DeepSeek R1 via Hugging Face API.
    """
    if not texts:
        return 0  # Neutral sentiment if no texts
    
    # Concatenate texts with context
    combined_text = " ".join([t[:500] for t in texts])[:5000]
    
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-r1"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    prompt = f"""
    Analyze the sentiment of the following content about {ticker} stock and rate it 
    on a scale from -1 (extremely negative) to 1 (extremely positive), where 0 is neutral.
    
    Content: {combined_text}
    
    Provide your analysis in JSON format with a single 'score' field containing the numerical sentiment score.
    """
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        
        # Extract score from the response (format depends on the model output)
        # This may need to be adjusted based on actual model output format
        if isinstance(result, list) and len(result) > 0:
            text_response = result[0]["generated_text"]
            # Parse JSON from text response
            import re
            json_match = re.search(r'{\s*"score"\s*:\s*(-?\d+\.?\d*)\s*}', text_response)
            if json_match:
                return float(json_match.group(1))
        
        return 0  # Default to neutral on error
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0  # Default to neutral on error
```

## Option 4: Self-Hosting DeepSeek R1

For maximum control and potentially lower costs, you can self-host DeepSeek R1:

1. Set up a powerful server with GPU support (NVIDIA A100 or similar)
2. Install required dependencies:
   ```bash
   pip install deepseek-ai torch
   ```
3. Download the model:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model_name = "deepseek-ai/deepseek-coder-r1"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)
   ```
4. Implement the sentiment analysis function:
   ```python
   def analyze_with_self_hosted_deepseek(text, ticker):
       prompt = f"""
       Analyze the sentiment of the following content about {ticker} stock and rate it 
       on a scale from -1 (extremely negative) to 1 (extremely positive), where 0 is neutral.
       
       Content: {text}
       
       Provide your analysis in JSON format with a single 'score' field containing the numerical sentiment score.
       """
       
       inputs = tokenizer(prompt, return_tensors="pt")
       outputs = model.generate(
           inputs["input_ids"], 
           max_length=500,
           temperature=0.1
       )
       
       response = tokenizer.decode(outputs[0])
       # Extract score from response
       # Implementation depends on model output format
       # ...
       
       return score
   ```

## Connecting to QuantConnect

Once your API is deployed, update your QuantConnect algorithm with the endpoint URL:

1. Go to your QuantConnect algorithm
2. Set the API endpoint and key as parameters:
   ```csharp
   self.sentiment_api_url = self.GetParameter("SentimentAPIEndpoint")
   self.sentiment_api_key = self.GetParameter("SentimentAPIKey")
   ```
3. Ensure the `GetAISentiment` function in the algorithm is properly connected to your API

## Testing and Monitoring

1. Test your API with sample requests:
   ```bash
   curl "https://your-api-endpoint/api/sentiment?ticker=AAPL&days=5"
   ```

2. Set up monitoring using CloudWatch (AWS), Stackdriver (GCP), or similar

3. Create alerts for errors or timeouts

## Security Considerations

1. Use API keys to secure access to your sentiment API
2. Implement rate limiting to prevent abuse
3. Consider using HTTPS for all communications
4. Keep your API keys and credentials secure

## Cost Optimization

1. Implement caching to reduce API calls to news sources and the AI model
2. Use serverless for cost-effective scaling
3. Consider batch processing for multiple tickers

## Troubleshooting

If you encounter issues:

1. Check API response logs
2. Verify API keys are valid
3. Test API endpoints directly
4. Ensure proper error handling in your code

## Next Steps

- Implement more sophisticated news and social media data collection
- Add additional data sources (analyst reports, earnings call transcripts)
- Create a dashboard to monitor sentiment trends
- Explore advanced AI models for improved sentiment analysis