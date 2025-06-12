import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

API_KEY = '665d3cb0ce534d5da4d72bafc4fd7bc3'
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    url = (
        'https://newsapi.org/v2/everything?'
        f'q={ticker}&'
        'sortBy=publishedAt&'
        f'from={(datetime.now() - timedelta(days=20)).date()}&'
        f'to={datetime.now().date()}&'
        f'apiKey={API_KEY}'
    )

    response = requests.get(url)
    articles = response.json().get('articles', [])

    scores = []
    for article in articles:
        content = article['title'] + " " + (article.get('description') or "")
        score = analyzer.polarity_scores(content)['compound']
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0
