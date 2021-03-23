#Importing the Libraries
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

#Loading the Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)

#Dimensions of the Dataset
print(dataset.shape)

#Take a look at the data
print(dataset.head())

#Statistical Summary
print(dataset.describe())

#Class Distribution
print(dataset.groupby('class').size())

#Univariate Plots - Box and Whisker Plots
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
pyplot.show()

#Histogram of the Variable
dataset.hist()
pyplot.show()

#Multivariate Plots
scatter_matrix(dataset)
pyplot.show()

#Creating a Validation Dataset and Splitting the Dataset
arr = dataset.values
X = arr[:, 0:4]
Y = arr[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

#Building Models
models = []
models.append(('lr', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('lda', LinearDiscriminantAnalysis()))
models.append(('knn', KNeighborsClassifier()))
models.append(('nb', GaussianNB()))
models.append(('svm', SVC(gamma = 'auto')))

#Evaluating the Created Models
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits = 10)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" %(name, cv_results.mean(), cv_results.std()))

#Comparing the Models
pyplot.boxplot(results, labels = names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#Making predictions on SVM
model = SVC(gamma = 'auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

#Evaluating our Predictions
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
