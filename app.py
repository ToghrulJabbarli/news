import time
import requests
from decouple import config
from news_analyzer import NewsAnalyzer


def push_data_to_power_bi(api_url, data_rows):
    for index, data_row in enumerate(data_rows):
        try:
            response = requests.post(api_url, json=data_row)

            if response.status_code == 200:
                print(f"Row {index + 1} pushed successfully.")
            else:
                print(f"Error pushing row {index + 1}: {response.status_code}, {response.text}")

        except requests.RequestException as e:
            print(f"Request error for row {index + 1}: {e}")

        # Add a delay to avoid too frequent requests
        time.sleep(0.5)


def main():
    api_key = config('API_KEY')
    api_url_category_sentiment = config('API_URL_CATEGORY_SENTIMENT')
    api_url_total_sentiment = config('API_URL_TOTAL_SENTIMENT')
    news_analyzer = NewsAnalyzer(api_key)
    news_analyzer.analyze_sentiment()
    sentiment_df = news_analyzer.news_with_cat
    sentiment_df_final = sentiment_df.groupby('category')['sentiment'].value_counts().unstack(fill_value=0)
    sentiment_df_final = sentiment_df_final.rename_axis(columns=None).reset_index()
    sentiment_df_final_total = sentiment_df.groupby('sentiment').size()
    print(sentiment_df_final_total)

    data_rows = []
    data_rows_second = []
    for index, row in sentiment_df_final.iterrows():
        data_row = {
            'category': row['category'],
            'negative': row.get('Negative', 0),
            'positive': row['Positive']
        }
        data_rows.append(data_row)

    data_rows_sec = {
        'Negative': int(sentiment_df_final_total.get('Negative', 0)),
        'Positive': int(sentiment_df_final_total.get('Positive', 0))}
    data_rows_second.append(data_rows_sec)
    print(data_rows_second)

    push_data_to_power_bi(api_url_category_sentiment, data_rows)
    push_data_to_power_bi(api_url_total_sentiment, data_rows_second)


if __name__ == "__main__":
    main()
