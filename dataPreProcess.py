import pandas as pd
from sklearn.utils import shuffle

data_path = "data/Data.csv"
data = pd.read_csv(data_path)
print(data.shape)
data = shuffle(data)
#对数据进行分割，分成训练数据、测试数据和评估数据
train_data = data[1:150]
test_data = data[151:200]
dev_data = data[201:250]
print(data["label"][1])
print(data["review"][1])

# for index, row in data.iterrows():
#     print(row["label"], row["review"])
train_data.to_csv('data/train_data.csv')
test_data.to_csv('data/test_data.csv')
dev_data.to_csv('data/dev_data.csv')