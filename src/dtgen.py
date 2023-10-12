"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import sys, csv, random, numpy as np, graphviz, sklearn2pmml
from distutils.dir_util import mkpath
from nyoka import skl_to_pmml
#from sklearn import tree, pipeline, ensemble
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from tabulate import tabulate

from .ConfigParsers.DtGenConfigParser import DtGenConfigParser


def read_dataset_from_csv(csv_file, delimiter, skip_header, outcome_col):
    attributes = []
    outcomes = []
    with open(csv_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(spamreader)
        for row in spamreader:
            outcome = row.pop(outcome_col)
            outcomes.append(outcome)
            attributes.append(row)
    return np.array(attributes, dtype = float), outcomes

def to_one_hot(outcomes, classes_name):
    try:
        x_one_hot = []
        n_unique_values = len(classes_name)
        for i in outcomes:
            next_el = [0] * n_unique_values
            next_el[classes_name.index(i) if isinstance(classes_name, list) else list(classes_name.keys()).index(i) ] = 1
            x_one_hot.append(next_el)
        return x_one_hot
    except ValueError as e:
        print(e)
        print(classes_name)
        exit()

def get_labels(outcomes, config):
    class_names = config.classes_name if isinstance(config.classes_name, list) else list(config.classes_name.keys())
    return [ class_names.index(o) for o in outcomes ]
    
def get_sets(dataset_file, config, fraction):
    if config.outcome_col is None:
        config.outcome_col = -1
    attributes, outcomes = read_dataset_from_csv(dataset_file, config.separator, config.skip_header, config.outcome_col)
    labels = np.array(get_labels(outcomes, config))
    labels.reshape((attributes.shape[0],)) 
    learning_set_size = int(np.ceil(attributes.shape[0] * fraction))
    learning_vectors_indexes = random.sample(range(int(attributes.shape[0])), learning_set_size)
    test_vectors_indexes = [x for x in range(attributes.shape[0]) if x not in learning_vectors_indexes]
    return list(attributes[learning_vectors_indexes, :]), list(labels[learning_vectors_indexes]), list(attributes[test_vectors_indexes, :]), list(labels[test_vectors_indexes])

def save_dataset_to_csv(filename, attributes_name, attributes, labels):
    original_stdout = sys.stdout
    with open(filename, "w") as file:
        sys.stdout = file
        print(*attributes_name, "Outcome", sep=";")
        for a, c  in zip(attributes, labels):
            print(*a, c, sep=";")
    sys.stdout = original_stdout  
    
def graphviz_export(model, attributes_name, classes_name, outputdir):
    if isinstance(model, RandomForestClassifier):
        for i, estimator in enumerate(model.estimators_):
            dot_data = export_graphviz(estimator, feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
            graph = graphviz.Source(dot_data)
            graph.render(directory = f"{outputdir}/export", filename = f'tree_{i}.gv')
    else:
        dot_data = export_graphviz(model, feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render(directory = f"{outputdir}/export", filename = 'tree.gv')
        
def dt_learner():
    pass

def rf_learner_w_random_search_cv(n_trees, x_train, y_train, n_iter = 150, cv = 3):
    search_grid = { #'n_estimators' : [int(x) for x in np.linspace(5, 200, num = 20)],
                    'max_features': ['auto', 'log2', 'sqrt'],
                    'criterion' : ["gini", "entropy", "log_loss"],
                    'max_depth': [int(x) for x in np.linspace(10, 50, num = 10)],
                    'min_samples_split': [int(x) for x in np.linspace(10, 100, num = 10)],
                    'min_samples_leaf': [int(x) for x in np.linspace(20, 100, num = 10)],
                    'bootstrap': [True, False]}
    rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(n_estimators = n_trees), param_distributions = search_grid, n_iter = n_iter, cv = cv, verbose=2, random_state = np.random.default_rng().integers(0, 100), n_jobs = -1)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)
    return rf_random.best_estimator_
    
def rf_learner_w_grid_search_cv(n_trees, x_train, y_train, cv = 3):
    search_grid = { #'n_estimators' : [int(x) for x in np.linspace(5, 200, num = 20)],
                    'max_features': ['auto', 'log2', 'sqrt'],
                    'criterion' : ["gini", "entropy", "log_loss"],
                    'max_depth': [int(x) for x in np.linspace(5, 20, num = 11)],
                    'min_samples_split': [int(x) for x in np.linspace(2, 15, num = 5)],
                    'min_samples_leaf': [int(x) for x in np.linspace(20, 140, num = 10)],
                    'bootstrap': [True, False]}
    rf_random = GridSearchCV(estimator = RandomForestClassifier(n_estimators = n_trees), param_grid = search_grid, cv = cv, verbose=2, n_jobs = -1)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)
    return rf_random.best_estimator_

def random_search_cv(clf, dataset, configfile, outputdir, fraction, ntrees, niter):
    config = DtGenConfigParser(configfile)
    x_train, y_train, x_test, y_test = get_sets(dataset, config, fraction)
    if clf == "dt":
        pass
    else:
        model = rf_learner_w_random_search_cv(ntrees, x_train, y_train, niter)
        data = [ [i, estimator.tree_.node_count, estimator.tree_.max_depth ] for i, estimator in enumerate(model.estimators_) ]
        print(tabulate(data, headers=["#", "#nodes", "depth"]))
            
    print(f"Classification accuracy: {sum(pred == ans for pred, ans in zip(model.predict(x_test), y_test)) / len(y_test)}")
    mkpath(outputdir)
    pmml_file = f"{outputdir}/classifier.pmml"
    print(f"Exporting PMML model to {pmml_file} ...")
    pipe = Pipeline([('clf', model)])
    skl_to_pmml(pipeline = pipe, col_names = config.attributes_name, pmml_f_name = pmml_file )
    training_set_csv = f"{outputdir}/training_set.csv"
    print(f"Exporting the training set to {training_set_csv} ...")
    save_dataset_to_csv(training_set_csv, config.attributes_name, x_train, y_train)
    test_dataset_csv = f"{outputdir}/test_set.csv"
    print(f"Exporting the test set to {test_dataset_csv} ...")
    save_dataset_to_csv(test_dataset_csv, config.attributes_name, x_test, y_test)
    print(f"Exporting graphviz draws of learned trees to {outputdir}/export ...")
    graphviz_export(model, config.attributes_name, list(config.classes_name.values()) if isinstance(config.classes_name, dict) else config.classes_name, outputdir)
    print("Done!")

def grid_search_cv(clf, dataset, configfile, outputdir, fraction, ntrees):
    config = DtGenConfigParser(configfile)
    x_train, y_train, x_test, y_test = get_sets(dataset, config, fraction)
    if clf == "dt":
        pass
    else:
        model = rf_learner_w_grid_search_cv(ntrees, x_train, y_train)
        data = [ [i, estimator.tree_.node_count, estimator.tree_.max_depth ] for i, estimator in enumerate(model.estimators_) ]
        print(tabulate(data, headers=["#", "#nodes", "depth"]))
            
    print(f"Classification accuracy: {sum(pred == ans for pred, ans in zip(model.predict(x_test), y_test)) / len(y_test)}")
    mkpath(outputdir)
    pmml_file = f"{outputdir}/classifier.pmml"
    print(f"Exporting PMML model to {pmml_file} ...")
    pipe = Pipeline([('clf', model)])
    skl_to_pmml(pipeline = pipe, col_names = config.attributes_name, pmml_f_name = pmml_file )
    training_set_csv = f"{outputdir}/training_set.csv"
    print(f"Exporting the training set to {training_set_csv} ...")
    save_dataset_to_csv(training_set_csv, config.attributes_name, x_train, y_train)
    test_dataset_csv = f"{outputdir}/test_set.csv"
    print(f"Exporting the test set to {test_dataset_csv} ...")
    save_dataset_to_csv(test_dataset_csv, config.attributes_name, x_test, y_test)
    print(f"Exporting graphviz draws of learned trees to {outputdir}/export ...")
    graphviz_export(model, config.attributes_name, list(config.classes_name.values()) if isinstance(config.classes_name, dict) else config.classes_name, outputdir)
    print("Done!")
    
     
def dtgen(clf, dataset, configfile, outputdir, fraction, depth, predictors, criterion, min_sample_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha, disable_bootstrap):
    config = DtGenConfigParser(configfile)
    x_train, y_train, x_test, y_test = get_sets(dataset, config, fraction)
    if clf == "dt":
       model = DecisionTreeClassifier(max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha).fit(x_train, y_train)
       print(model.tree_.node_count())
       print(model.tree_.max_dept)
    elif clf == "rf":
        model = RandomForestClassifier(n_estimators = predictors, max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha, bootstrap = not disable_bootstrap, n_jobs = -1, verbose = 1).fit(x_train, y_train)
        data = [ [i, estimator.tree_.node_count, estimator.tree_.max_depth ] for i, estimator in enumerate(model.estimators_) ]
        print(tabulate(data, headers=["#", "#nodes", "depth"]))
            
    print(f"Classification accuracy: {sum(pred == ans for pred, ans in zip(model.predict(x_test), y_test)) / len(y_test)}")
    mkpath(outputdir)
    pmml_file = f"{outputdir}/classifier.pmml"
    print(f"Exporting PMML model to {pmml_file} ...")
    pipe = Pipeline([('clf', model)])
    skl_to_pmml(pipeline = pipe, col_names = config.attributes_name, pmml_f_name = pmml_file )
    training_set_csv = f"{outputdir}/training_set.csv"
    print(f"Exporting the training set to {training_set_csv} ...")
    save_dataset_to_csv(training_set_csv, config.attributes_name, x_train, y_train)
    test_dataset_csv = f"{outputdir}/test_set.csv"
    print(f"Exporting the test set to {test_dataset_csv} ...")
    save_dataset_to_csv(test_dataset_csv, config.attributes_name, x_test, y_test)
    print(f"Exporting graphviz draws of learned trees to {outputdir}/export ...")
    graphviz_export(model, config.attributes_name, list(config.classes_name.values()) if isinstance(config.classes_name, dict) else config.classes_name, outputdir)
    print("Done!")
    