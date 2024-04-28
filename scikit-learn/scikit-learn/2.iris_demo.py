import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.data)
# print(iris['target'])
# print(iris.feature_names)
# print(iris.DESCR)

iris_d = pd.DataFrame(data=iris.data,
                      columns=['sepal length',
                               'sepal width',
                               'petal length',
                               'petal width'])

iris_d['target'] = iris.target

def iris_plot(data, col1, col2):
  sns.lmplot(x=col1,y=col2, data=data, hue='target', fit_reg=False)
  plt.show()

# iris_plot(iris_d,'sepal width', 'petal length',)
iris_plot(iris_d,'sepal length', 'petal width',)