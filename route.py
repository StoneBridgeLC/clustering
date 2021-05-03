from flask import Flask, Response
from cluster import get_data_from_api, preprocessing, cluster


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    return '테스트!'


@app.route('/news')
def clustering():
    url = 'http://localhost:8080/news'
    news_df = get_data_from_api(url)
    vect = preprocessing(news_df)
    cluster_res = cluster(vect)

    print(news_df['body'][0])

    news_df['label'] = cluster_res.labels_

    return Response(news_df.to_json(orient="records", force_ascii=False), content_type='application/json; charset=utf-8')


if __name__ == '__main__':
    app.run(debug=True)

