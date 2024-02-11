import pandas as pd
import newsapi
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

class NewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.newsapi_client = newsapi.NewsApiClient(api_key=self.api_key)
        self.categories = ['general', 'business', 'technology', 'entertainment', 'health', 'science', 'sports']
        self.news_with_cat = pd.DataFrame()

    def fetch_news_data(self):
        for category in self.categories:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            all_articles = self.newsapi_client.get_everything(q=category, from_param=from_date, to=to_date, language='en')
            df = pd.DataFrame(all_articles['articles'])
            df['category'] = category
            self.news_with_cat = pd.concat([self.news_with_cat, df], ignore_index=True)

    def preprocess_sentiment_data(self):
        self.news_with_cat['tokens'] = self.news_with_cat['title'].apply(lambda x: word_tokenize(x.lower()))

        stemmer = PorterStemmer()
        self.news_with_cat['stemmed_tokens'] = self.news_with_cat['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

        lemmatizer = WordNetLemmatizer()
        self.news_with_cat['lemmatized_tokens'] = self.news_with_cat['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

        sia = SentimentIntensityAnalyzer()
        self.news_with_cat['compound_sentiment'] = self.news_with_cat['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

    def classify_sentiment(self):
        self.news_with_cat['sentiment'] = "Positive"
        self.news_with_cat.loc[self.news_with_cat['compound_sentiment'] < 0, 'sentiment'] = 'Negative'

    def analyze_sentiment(self):
        self.fetch_news_data()
        self.preprocess_sentiment_data()
        self.classify_sentiment()
