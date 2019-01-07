from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

def classify(vectors, labels, type="SVM"):
    # Random Splitting With Ratio 3 : 1
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels, test_size=0.25)

    # Initialize Model
    classifier = None
    if(type=="GNB"):
        classifier = GaussianNB()
    elif(type=="MNB"):
        classifier = MultinomialNB()
    elif(type=="KNN"):
        classifier = KNeighborsClassifier()
        classifier = GridSearchCV(classifier, dict(n_neighbors=[3,5,7,9]), cv=3)
    elif(type=="SVM"):
        classifier = SVC()
        classifier = GridSearchCV(classifier, {'C':[0.001, 0.01, 0.1, 1, 10], 'gamma' :[0.001, 0.01, 0.1, 1]}, cv=3)
    elif(type=="DT"):
        classifier = DecisionTreeClassifier()
        params = {'criterion':['gini','entropy'],'max_depth':[5,10,20,50,100,200]}
        #classifier = GridSearchCV(classifier, params, cv=3)
    elif(type=="RF"):
        classifier = RandomForestClassifier()
        classifier = GridSearchCV(classifier, {'n_estimators': [n for n in range(10,100,10)]}, cv=3)
    elif(type=="SGD"):
        params = {'alpha': [10 ** x for x in range(-6, 1)], 'l1_ratio': [0, 0.1, 0.5, 0.9, 1]}
        classifier = SGDClassifier()
        classifier = GridSearchCV(classifier, params, cv=3)
    else:
        print("Wrong Classifier Type!")
        return

    print("Tuning .. Please be patient...")

    classifier.fit(train_vectors, train_labels)
    #classifier = classifier.best_estimator_
    accuracy = accuracy_score(test_labels, classifier.predict(test_vectors))
    print("Accuracy:", accuracy)
