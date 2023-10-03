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
from sklearn import tree, pipeline, ensemble
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
    return attributes[learning_vectors_indexes, :], labels[learning_vectors_indexes], attributes[test_vectors_indexes, :], labels[test_vectors_indexes]

def save_dataset_to_csv(filename, attributes_name, attributes, labels):
    original_stdout = sys.stdout
    with open(filename, "w") as file:
        sys.stdout = file
        print(*attributes_name, "Outcome", sep=";")
        for a, c  in zip(attributes, labels):
            print(*a, c, sep=";")
    sys.stdout = original_stdout  
    
def graphviz_export(model, attributes_name, classes_name, outputdir):
    if isinstance(model, (ensemble.RandomForestClassifier, ensemble.BaggingClassifier)):
        for i in range(len(model.estimators_)):
            dot_data = tree.export_graphviz(model.estimators_[i], feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
            graph = graphviz.Source(dot_data)
            graph.render(directory = f"{outputdir}/export", filename = f'tree_{i}.gv')
    else:
        dot_data = tree.export_graphviz(model, feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render(directory = f"{outputdir}/export", filename = 'tree.gv')
    
def dtgen(clf, dataset, configfile, outputdir, fraction, depth, predictors, criterion, min_sample_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha, disable_bootstrap):
    config = DtGenConfigParser(configfile)
    learning_attributes, learning_labels, test_attributes, test_labels = get_sets(dataset, config, fraction)
    if clf == "dt":
        model = tree.DecisionTreeClassifier(max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha).fit(learning_attributes, learning_labels)
    elif clf == "rf":
        model = ensemble.RandomForestClassifier(n_estimators = predictors, max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha, bootstrap = not disable_bootstrap, n_jobs = -1, verbose = 1).fit(list(learning_attributes), list(learning_labels))
    elif clf == "wc":
        st = tree.DecisionTreeClassifier( max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha).fit(learning_attributes, learning_labels)
        model = ensemble.RandomForestClassifier( n_estimators = predictors, max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha, bootstrap = not disable_bootstrap, n_jobs = -1, verbose = 1).fit(list(learning_attributes), list(learning_labels))
        for i in range(len(model.estimators_)):
            model.estimators_[i] = st
    print(f"Classification accuracy: {sum(pred == ans for pred, ans in zip(model.predict(test_attributes), test_labels)) / len(test_labels)}")
    mkpath(outputdir)
    pmml_file = f"{outputdir}/{clf}_{predictors}.pmml" if predictors > 1 else f"{outputdir}/{clf}.pmml"
    print(f"Exporting PMML model to {pmml_file} ...")
    pipe = pipeline.Pipeline([('clf', model)])
    skl_to_pmml(pipeline = pipe, col_names = config.attributes_name, pmml_f_name = pmml_file )
    training_set_csv = f"{outputdir}/training_set.csv"
    print(f"Exporting the training set to {training_set_csv} ...")
    save_dataset_to_csv(training_set_csv, config.attributes_name, learning_attributes, learning_labels)
    test_dataset_csv = f"{outputdir}/test_set.csv"
    print(f"Exporting the test set to {test_dataset_csv} ...")
    save_dataset_to_csv(test_dataset_csv, config.attributes_name, test_attributes, test_labels)
    print("Done!")
    