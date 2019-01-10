from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

def classify(vectors, labels, type="SVM"):
    # Random Splitting With Ratio 3 : 1
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels, test_size=0.25)

    # Initialize Model
    classifier = None
    if(type=="GNB"):
        classifier = GaussianNB()
        classifier.fit(train_vectors, train_labels)
    elif(type=="MNB"):
        classifier = MultinomialNB()
        classifier.fit(train_vectors, train_labels)
    elif(type=="KNN"):
        classifier = KNeighborsClassifier()
        classifier = GridSearchCV(classifier, dict(n_neighbors=[3,5,7,9]), cv=3)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="SVM"):
        classifier = SVC()
        classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10], 'gamma' :[0.001, 0.01, 0.1, 1]}, cv=3)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="DT"):
        classifier = DecisionTreeClassifier()
        params = {'criterion':['gini','entropy'],'max_depth':[5,10,20,50,100,200]}
        classifier = GridSearchCV(classifier, params, cv=3)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif(type=="RF"):
        classifier = RandomForestClassifier()
        classifier = GridSearchCV(classifier, {'n_estimators': [n for n in range(10,100,10)]}, cv=3)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    else:
        print("Wrong Classifier Type!")
        return

    print("Tuning .. Please be patient...")

    accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
    print("Training Accuracy:", accuracy)
    accuracy = accuracy_score(test_labels, classifier.predict(test_vectors))
    print("Test Accuracy:", accuracy)


def regress(vectors, labels):
    # Random Splitting With Ratio 3 : 1
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels, test_size=0.25)

    regressor = LinearRegression()
    regressor.fit(train_vectors, train_labels)
    print("Training Accuracy:", accuracy)
    accuracy = accuracy_score(test_labels, regressor.predict(test_vectors))
    print("Test Accuracy:", accuracy)
