#encoding:utf-8
#1.获取数据
#2.数据基本处理
#2.1数据分割
#3. 特征工程 - 标准化
#4. 机器学习--线性回归
#5. 模型评估

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np

def load_boston():
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
  data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  target = raw_df.values[1::2, 2]
  return data, target

def linear_model():
  # 1.获取数据
  boston = load_boston()
  #2.数据基本处理
  #2.1数据分割
  x_train, x_test, y_train, y_test = train_test_split(boston[0], boston[1], test_size=0.2)
  # 3. 特征工程 - 标准化
  transfer = StandardScaler()
  x_train = transfer.fit_transform(x_train)
  x_test = transfer.fit_transform(x_test)

  # 4. 机器学习--线性回归
  estimator = LinearRegression()
  estimator.fit(x_train, y_train)
  print("模型偏置时:", estimator.intercept_)
  print("模型系数是:", estimator.coef_)
  # 5. 模型评估
  #5.1预测值
  y_pre = estimator.predict(x_test)
  print('ypre:', y_pre)
  #5.2均方误差
  ret = mean_squared_error(y_test, y_pre)
  print("ret:", ret)

linear_model()