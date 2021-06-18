import utils
import os
import time
import boto3
import schedule


def clustering():
    dataserver = os.environ['DATASERVER']
    news_df = utils.get_data_from_api(dataserver)
    ftr_vect = utils.generate_tfidf(news_df)
    utils.cluster_to_csv(ftr_vect, news_df)

    current_time = time.strftime('%Y%m%d')
    csvfile = f'./{current_time}.csv'
    news_df.to_csv(csvfile, sep=',', encoding='utf-8', index=None)

    s3 = boto3.client('s3')
    bucket_name = 'newsclusterstorage'

    try:
        s3.upload_file(csvfile, bucket_name, csvfile)
    except:
        print(f'Upload to s3 bucket is failed.')
    else:
        print(f'Result of clustering at {current_time} is uploaded')


clustering()
#schedule.every().day.at("16:07").do(clustering)

#while True:
#    schedule.run_pending()
#    time.sleep(1)
