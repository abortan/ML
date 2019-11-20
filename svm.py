from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split


iris=load_iris()
shape=iris.data.shape
target=iris.target
target_names=iris.target_names
feature=iris.feature_names
print('iris data','\n', iris.data)
print('iris shape','\n', shape)
print('iris target','\n', target)
print('target names','\n', target_names)
print('feature','\n', feature)

x_train, x_test, y_train, y_test = train_test_split(iris.data, target, test_size=0.2) 


print('x train','\n', x_train.shape) 
print('x test','\n',x_test.shape)

print('y train','\n',y_train) 

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print('true_value:      ',y_test)
print('predicted_value: ',predicted)

