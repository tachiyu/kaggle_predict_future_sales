from google.cloud import bigquery
import lightgbm as lgb
import numpy as np
import pandas as pd

CLIENT = bigquery.Client()
DATASET = CLIENT.dataset("predict_future_sales")
print("Client creating using default project: {}".format(CLIENT.project))
print("Target dataset: {}".format(DATASET.dataset_id))

# SQLからデータ作製し表示する関数
def get_data(sql):
    query_job = CLIENT.query(sql)
    return query_job.to_dataframe()

# SQLからデータ作製し保存する関数
def save_data(sql, destination_table):
    table_ref = DATASET.table(destination_table)
    job_config = bigquery.QueryJobConfig(destination=table_ref)
    # Start the query, passing in the extra configuration.
    query_job = CLIENT.query(sql, job_config=job_config)
    query_job.result()  # Waits for the query to finish
    print("Query results loaded to table {}".format(table_ref.path))
    
# lightGBMでpredictionをする関数
def predict(training, test):
    model = lgb.LGBMRegressor()
    model.fit(training.drop("item_cnt_month", axis=1), training.item_cnt_month)
    y_pred = model.predict(test)
    y_pred = np.clip(y_pred, 0, 20)
    return y_pred

# 任意のカラムを落とす関数
def drop_col(df, cols):
    for col in cols:
        df = df.drop(col, axis=1)
    return df

# RMSEを求める関数
def RMSE(y_pred, y):
    return np.sqrt(np.mean(np.power(y_pred - np.clip(y, 0, 20), 2)))

# cross validationする関数
def cross_validate(train):
    trainblocks = []
    valblocks = []
    for i in range(10):
        trainblocks.append(train[train.block < 33-i])
        valblocks.append(train[train.block == 33-i])    
    accuracies = []
    for training, val in zip(trainblocks, valblocks):
        accuracies.append(RMSE(predict(training, val.drop("item_cnt_month", axis=1)), val.item_cnt_month))
    return accuracies

# submission.csv を作る関数
def make_submission(train, test, filepath):
    y_pred = predict(train, test)
    pd.DataFrame({"ID":range(len(y_pred)), "item_cnt_month":y_pred}).to_csv(filepath, index=False)