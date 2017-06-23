import csv
import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    raw_data = load_data()
    X = raw_data[:,:2]
    y = raw_data[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    param_grid_linear = [{'kernel': ['linear'], 'C': [0.1,0.5,1,5,10,50,100]}]
    param_grid_polynomial = [{'kernel': ['poly'], 'C': [0.1,1,3], 'degree': [4,5,6], 'gamma': [0.1,0.5]}]
    param_grid_rbf = [{'kernel': ['rbf'], 'C': [0.1,0.5,1,5,10,50,100], 'gamma': [0.1,0.5,1,3,6,10]}]
    param_grid_log_reg = [{'C': [0.1,0.5,1,5,10,50,100]}]
    param_grid_knn = [{'n_neighbors': range(1,51), 'leaf_size': [5,10,15,20,25,30,35,40,45,50,55,60]}]
    param_grid_dec_trees = [{'max_depth': range(1,51), 'min_samples_split': [2,3,4,5,6,7,8,9,10]}]
    param_grid_rand_forest = [{'max_depth': range(1,51), 'min_samples_split': [2,3,4,5,6,7,8,9,10]}]
    scores = []
    results = []
    scores = get_scores1(X_train, X_test, y_train, y_test, param_grid_linear)
    results.append(['svm_linear', scores[0], scores[1]])
    
    scores = get_scores1(X_train, X_test, y_train, y_test, param_grid_polynomial)
    results.append(['svm_polynomial', scores[0], scores[1]])
    
    scores = get_scores1(X_train, X_test, y_train, y_test, param_grid_rbf)
    results.append(['svm_rbf', scores[0], scores[1]])

    scores = get_scores2(X_train, X_test, y_train, y_test, param_grid_log_reg, LogisticRegression())
    results.append(['logistic', scores[0], scores[1]])

    scores = get_scores2(X_train, X_test, y_train, y_test, param_grid_knn, KNeighborsClassifier())
    results.append(['knn', scores[0], scores[1]])

    scores = get_scores2(X_train, X_test, y_train, y_test, param_grid_dec_trees, DecisionTreeClassifier())
    results.append(['decision_tree', scores[0], scores[1]])

    scores = get_scores2(X_train, X_test, y_train, y_test, param_grid_rand_forest, RandomForestClassifier())
    results.append(['random_forest', scores[0], scores[1]])

    write_data(results)
    #print results


def get_scores1(X_train, X_test, y_train, y_test, grid):
    clf = GridSearchCV(SVC(C=1), grid, cv=5)
    clf.fit(X_train, y_train)
    #print clf.best_params_
    means = clf.cv_results_['mean_test_score']
    #stds = clf_linear.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #print '%0.3f (+/-%0.03f) for %r'% (mean, std*2, params)
    scores = []
    best_score = max(means)
    scores.append(best_score)
    #print best_score
        #print '\n'
    y_true, y_pred = y_test, clf.predict(X_test)
    #test_score = count1 / count2
    test_score = accuracy_score(y_true, y_pred)
    #print test
    #print len(y_true)
    #test_score = test / len(y_true)
    scores.append(test_score)
    return scores
    #print y_true
    #print y_pred
    #print test_score
    #write_data(raw_data)

def get_scores2(X_train, X_test, y_train, y_test, grid, algorithm):
    method = algorithm
    clf = GridSearchCV(method, grid, cv=5)
    clf.fit(X_train, y_train)
    #print gs.best_score_
    means = clf.cv_results_['mean_test_score']
    #stds = gs_linear.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        #print '%0.3f (+/-%0.03f) for %r'% (mean, std*2, params)
    scores = []
    best_score = max(means)
    scores.append(best_score)
    y_true, y_pred = y_test, clf.predict(X_test)
    test_score = accuracy_score(y_true, y_pred)
    scores.append(test_score)
    return scores

def load_data():
    input_file = open(sys.argv[1], 'rU')
    reader = csv.reader(input_file)
    next(reader, None)
    raw_data = numpy.asarray(list(reader), dtype=float)
    input_file.close()
    return raw_data

def write_data(data):
    output_file = open(sys.argv[2], 'wb')
    writer = csv.writer(output_file)
    writer.writerows(data)
    output_file.close()

if __name__ == "__main__":
    main()
