from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#1. 获取数据
iris = load_iris()

#2.基本数据处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
print(x_train)
print(y_train)
#3.特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)
#4.机器学习-KNN
#4.1 实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5)
#4.2模型训练
estimator.fit(x_train, y_train)
#5.模型评估
#5.1 预测值结果输出
y_pre = estimator.predict(x_test)
print("预测值:\r\n", y_pre)
print("预测值和实际值的对比是:\n",y_pre == y_test)
#5.2准确率计算
score = estimator.score(x_test,y_test)
print("准确率为:\n", score)