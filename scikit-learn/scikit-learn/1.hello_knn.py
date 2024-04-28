# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier

# https://www.bilibili.com/video/BV1a7411d7fk/?p=50&spm_id_from=pageDriver&vd_source=50fa37c761fffe263169f7cd7d21c8b5
# 1.构造数据
x = [[1], [2], [10], [20]]
y = [0, 0, 1, 1]
# 2.训练模型
estimator = KNeighborsClassifier(n_neighbors=4)
estimator.fit(x, y)
# 3.数据预测
ret = estimator.predict([[0]])
print(ret)

ret = estimator.predict([[5]])
print(ret)
