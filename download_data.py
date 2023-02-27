import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv'

filename = 'data/full-corpus.csv'

def download():
    if not os.path.isfile(filename):
        print('Downloading data...')
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

    df = pd.read_csv(filename)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    print('The data has been successfully downloaded and saved.')


if __name__ == "__main__":
    download()
