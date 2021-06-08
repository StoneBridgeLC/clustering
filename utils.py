import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from konlpy.tag import Mecab
import requests


def get_data_from_api(url):
    try:
        res = requests.get(url)
    except requests.exceptions.ConnectionError as e:
        print(e)
        return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None

    json_data = res.json()

    return pd.DataFrame(json_data)


def generate_tfidf(df, min_df=0.01, max_df=0.8):
    mecab = Mecab()
    print('Generate tf-idf vector...')
    tfidf_vect = TfidfVectorizer(tokenizer=mecab.morphs,
                                 ngram_range=(1, 2),
                                 min_df=min_df, max_df=max_df)

    ftr_vect = tfidf_vect.fit_transform(df['body'])
    print('tf-idf vector is generated.')

    return ftr_vect


def cluster_to_csv(vect, df, min_cluster_size=20):
    print('Start clustering....')
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    res = cluster.fit(vect)
    df['label'] = res.labels_
    print('Clustering is finished.')
