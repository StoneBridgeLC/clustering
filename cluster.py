import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import requests


def get_data_from_api(url):
    res = requests.get(url)
    json_data = res.json()

    return pd.DataFrame(json_data)


def preprocessing(df):
    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)

    tfidf_vect = TfidfVectorizer(tokenizer=sp,
                                 ngram_range=(1, 2),
                                 min_df=0.1, max_df=0.85)

    ftr_vect = tfidf_vect.fit_transform(df['body'])

    return ftr_vect


def cluster(vect):
    dbs = DBSCAN(min_samples=6, eps=0.7)
    res = dbs.fit(vect)

    return res

