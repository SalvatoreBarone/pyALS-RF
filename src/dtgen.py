"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

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
import sys, random, numpy as np, graphviz
from distutils.dir_util import mkpath
from nyoka import skl_to_pmml
from sklearn import tree, pipeline, ensemble
import sklearn2pmml

def to_one_hot(x):
    x_one_hot = []
    x_unique = list(set(x))
    n_unique_values = len(x_unique)
    for i in x:
        next_el = [0] * n_unique_values
        next_el[x_unique.index(i)] = 1
        x_one_hot.append(next_el)
    return x_one_hot

def get_sets(dataset_file, separator, outcome_col, skip_header, fraction):
    data = np.genfromtxt(dataset_file, delimiter = separator, skip_header = 1 if skip_header else 0)
    learning_set_size = int(np.ceil(data.shape[0] * fraction))
    learning_vectors_indexes = random.sample(range(int(data.shape[0])), learning_set_size)
    test_vectors_indexes = [x for x in range(data.shape[0]) if x not in learning_vectors_indexes]
    if outcome_col is None:
        outcome_col = -1
    classes = np.copy(data[:, outcome_col])
    n_classes = len(set(classes))
    attributes = np.delete(data, outcome_col, axis = 1)
    classes_one_hot = np.array(to_one_hot(classes))
    classes.reshape((attributes.shape[0],))
    classes_one_hot.reshape((attributes.shape[0], n_classes))
    attributes_name = [ f"attribute_{i}" for i in range(int(attributes.shape[1]))]
    classes_name = [ f"class_{i}" for i in range(n_classes)]
    learning_attributes = attributes[learning_vectors_indexes, :]
    learning_labels = classes[learning_vectors_indexes]
    # learning_classes_one_hot = classes_one_hot[learning_vectors_indexes, :]
    test_attributes = attributes[test_vectors_indexes, :]
    test_labels = classes[test_vectors_indexes]
    test_labels_one_hot = classes_one_hot[test_vectors_indexes, :]
    # print(f"Number of attributes: {int(attributes.shape[1])}")
    # print(f"Number of classes: {n_classes}")
    # print(f"Learning set size is ceil({data.shape[0]} * {fraction}) = {learning_set_size}")
    # print(f"Testing set size is {len(test_vectors_indexes)}")

    return attributes_name, classes_name, learning_attributes, learning_labels, test_attributes, test_labels, test_labels_one_hot

def save_test_dataset_to_csv(filename, attributes_name, test_attributes, classes_name, test_labels_one_hot):
    original_stdout = sys.stdout
    with open(filename, "w") as file:
        sys.stdout = file
        print(*attributes_name, *classes_name, sep=",")
        for a, c  in zip(test_attributes, test_labels_one_hot):
            print(*a, *c, sep=",")
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
    

def dtgen(clf, dataset, outputdir, separator, outcome, skip_header, fraction, depth, predictors, criterion, min_sample_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha, disable_bootstrap):
    attributes_name, classes_name, learning_attributes, learning_labels, test_attributes, test_labels, test_labels_one_hot = get_sets(dataset, separator, outcome, skip_header, fraction)
    if clf == "dt":
        model = tree.DecisionTreeClassifier(
                max_depth = depth,
                criterion = criterion,
                min_samples_split = min_sample_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                min_impurity_decrease = min_impurity_decrease,
                ccp_alpha = ccp_alpha).fit(learning_attributes, learning_labels)
    elif clf == "rf":
        model = ensemble.RandomForestClassifier(
                n_estimators = predictors,
                max_depth = depth,
                criterion = criterion,
                min_samples_split = min_sample_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                min_impurity_decrease = min_impurity_decrease,
                ccp_alpha = ccp_alpha,
                bootstrap = not disable_bootstrap,
                n_jobs = -1,
                verbose = 1).fit(list(learning_attributes), list(learning_labels))
    elif clf == "bag":
        model = ensemble.BaggingClassifier(
                n_estimators = predictors,
                bootstrap = not disable_bootstrap,
                n_jobs = -1,
                verbose = 1).fit(list(learning_attributes), list(learning_labels))
    elif clf == "wc":
        st = tree.DecisionTreeClassifier(
                max_depth = depth,
                criterion = criterion,
                min_samples_split = min_sample_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                min_impurity_decrease = min_impurity_decrease,
                ccp_alpha = ccp_alpha).fit(learning_attributes, learning_labels)
        model = ensemble.RandomForestClassifier(
                n_estimators = predictors,
                max_depth = depth,
                criterion = criterion,
                min_samples_split = min_sample_split,
                min_samples_leaf = min_samples_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                min_impurity_decrease = min_impurity_decrease,
                ccp_alpha = ccp_alpha,
                bootstrap = not disable_bootstrap,
                n_jobs = -1,
                verbose = 1).fit(list(learning_attributes), list(learning_labels))
        for i in range(len(model.estimators_)):
            model.estimators_[i] = st
        
    print(f"Classification accuracy: {sum(pred == ans for pred, ans in zip(model.predict(test_attributes), test_labels)) / len(test_labels)}")
    mkpath(outputdir)
    pipe = sklearn2pmml.PMMLPipeline([("classifier", model)])
    sklearn2pmml.sklearn2pmml(pipe, f"{outputdir}/{clf}_{predictors}.pmml" if predictors > 1 else f"{outputdir}/{clf}.pmml", with_repr = True)
    save_test_dataset_to_csv(f"{outputdir}/test_dataset_4_pyALS-rf.csv", attributes_name, test_attributes, classes_name, test_labels_one_hot)
    graphviz_export(model, attributes_name, classes_name, outputdir)