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
import sys, csv, random, numpy as np, graphviz, joblib, math, logging
from distutils.dir_util import mkpath
from nyoka import skl_to_pmml
#from sklearn2pmml.pipeline import PMMLPipeline
#from sklearn2pmml import sklearn2pmml
from tqdm import tqdm
#from sklearn import tree, pipeline, ensemble
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from tabulate import tabulate
from .ConfigParsers.DtGenConfigParser import DtGenConfigParser
from .scikit.RandonForestClassifierMV import RandomForestClassifierMV
from .Model.Classifier import *
from .plot import boxplot

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
    logger = logging.getLogger("pyALS-RF")
    if config.outcome_col is None:
        config.outcome_col = -1
    attributes, outcomes = read_dataset_from_csv(dataset_file, config.separator, config.skip_header, config.outcome_col)
    logger.info(f"Read {len(attributes)} feature vectors and {len(outcomes)} labels")
    x_train, x_test, y_train, y_test = train_test_split(attributes, np.array(get_labels(outcomes, config)).reshape((attributes.shape[0],)), train_size = fraction)
    logger.info(f"Training sets is {len(x_train)} feature vectors and {len(y_train)} labels")
    logger.info(f"Testing sets is {len(x_test)} feature vectors and {len(y_test)} labels")
    return list(x_train), list(y_train), list(x_test), list(y_test)

def save_dataset_to_csv(filename, attributes_name, attributes, labels):
    original_stdout = sys.stdout
    with open(filename, "w") as file:
        sys.stdout = file
        print(*attributes_name, "Outcome", sep=";")
        for a, c  in zip(attributes, labels):
            print(*a, c, sep=";")
    sys.stdout = original_stdout  
    
def graphviz_export(model, attributes_name, classes_name, outputdir):
    if isinstance(model, (RandomForestClassifier, RandomForestClassifierMV)):
        for i, estimator in enumerate(model.estimators_):
            dot_data = export_graphviz(estimator, feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
            graph = graphviz.Source(dot_data)
            graph.render(directory = f"{outputdir}/export", filename = f'tree_{i}.gv')
    else:
        dot_data = export_graphviz(model, feature_names = attributes_name, class_names = classes_name, filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render(directory = f"{outputdir}/export", filename = 'tree.gv')
        
def print_clf(clf : DecisionTreeClassifier):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=values[i]
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=values[i],
                )
            )
        

def print_nodes(model):
    if isinstance(model, (RandomForestClassifier, RandomForestClassifierMV)):
        for estimator in model.estimators_:
            print_clf(estimator)
    elif isinstance(model, DecisionTreeClassifier):
        print_clf(model)



def plot_redundancy(model,  x_test, y_test, outfile):
    samples_error = { i: [] for i in range(model.n_classes_) }
    for x, y in tqdm( zip(x_test, y_test), total=len(y_test), desc="Collecting redundancy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        outcome =  model.predict_proba(np.array(x).reshape((1, -1)))[0]
        for i in range(model.n_classes_):
            if i != y:
                samples_error[i].append(np.ceil( (outcome[y] - outcome[i]) / 2))
    boxplot([ list(v) for v in samples_error.values()], "Classes", r"$E_{p_i}$", outfile, (model.n_classes_, 4), False)
    

            
        
def save_model(outputdir, config, model, best_params, x_train, y_train, x_test, y_test, cross_validate):
    logger = logging.getLogger("pyALS-RF")
    acc = sum( model.predict(np.array(x).reshape((1, -1))) == y for x, y in tqdm( zip(x_test, y_test), total=len(y_test), desc="Computing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False, ))[0]
    logger.info(f"Classification accuracy: {acc}/{len(y_test)}*100={acc / len(y_test) * 100}")
    mkpath(outputdir)
    dump_file = f"{outputdir}/classifier.joblib"
    pmml_file = f"{outputdir}/classifier.pmml"
    params_file = f"{outputdir}/best_parameters.json5"
    training_set_csv = f"{outputdir}/training_set.csv"
    test_dataset_csv = f"{outputdir}/test_set.csv"

    logger.info(f"Saving training parameters to {params_file}")
    with open(params_file, "w") as f:
        json5.dump(best_params, f, indent=2)
    logger.info(f"Exporting the training set to {training_set_csv} ...")
    save_dataset_to_csv(training_set_csv, config.attributes_name, x_train, y_train)
    logger.info(f"Exporting the test set to {test_dataset_csv} ...")
    save_dataset_to_csv(test_dataset_csv, config.attributes_name, x_test, y_test)
    logger.info(f"Exporting graphviz draws of learned trees to {outputdir}/export ...")
    graphviz_export(model, config.attributes_name, list(config.classes_name.values()) if isinstance(config.classes_name, dict) else config.classes_name, outputdir)

    logger.info(f"Dumping to {dump_file} ...")
    joblib.dump(model, dump_file)

    logger.info(f"Exporting PMML model to {pmml_file} ...")
    model.fake() #! This is vital! Call this function right before the PMML export
    skl_to_pmml(pipeline = Pipeline([('rfc', model)]), col_names = config.attributes_name, pmml_f_name = pmml_file )
    # pipeline = PMMLPipeline([("classifier", model)])
    # sklearn2pmml(pipeline, pmml_file, with_repr = True)
    model = joblib.load(dump_file) #! after calling the fake() method you have no choice but reloading the model from file...
    logger.info("Done PMML export!")
    
    logger.info("Computing redundancy...")
    plot_redundancy(model,  x_test, y_test, f"{outputdir}/error_boxplot.pdf")
    logger.info("Done computing redundancy!")
    
    if cross_validate:
        logger.info("Performing model crossvalidation...")
        models_crossvalidation(config, model, x_test, y_test, pmml_file, test_dataset_csv)
        logger.info("Done model crossvalidation!")

def models_crossvalidation(config, model, x_test, y_test, pmml_file, test_dataset_csv):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Performing model debugging and validation...")
    classifier = Classifier(cpu_count())
    classifier.pmml_parser(pmml_file, config)
    classifier.read_test_set(test_dataset_csv)
    acc_pyals = 0
    acc_scikit = 0
    mismatches = []
    for x, y, x_prime, y_prime in tqdm(zip(classifier.x_test, classifier.y_test, x_test, y_test), total = len(y_test), desc="Testing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        assert all( math.isclose(i, j, rel_tol=1e-5) for i, j in zip(x, x_prime)), f"Error reading attributes from {test_dataset_csv}. Read {x}, shuld be {x_prime}. {[i == j for i, j in zip(x, x_prime)]}"
        assert y == y_prime, f"Error reading labels. Read {y}, should be {y_prime}"
        score_1 = classifier.get_score(x)
        score_2 = classifier.get_score(x_prime)
        rho_1 = model.predict_proba(np.array(x).reshape((1, -1)))
        rho_2 = model.predict_proba(np.array(x_prime).reshape((1, -1)))
        assert all(i == j for i, j in zip(score_1, score_2)), f"Error in scores: {score_1} {score_2}"
        assert all(i == j for i, j in zip(rho_1[0], rho_2[0])), f"Error in rho: {rho_1} {rho_2}"
        if any(i != int(j) for i, j in zip(score_1, rho_1[0])):
            logger.warn(f"Error in model response: {score_1} {rho_1[0]}")
        outcome_1, draw_1 = classifier.predict(x)
        outcome_2, draw_2 = classifier.predict(x_prime)
        assert all(i == j for i, j in zip(outcome_1, outcome_2)), f"Error in outcome: Got {outcome_1} and {outcome_2}. {[i == j for i, j in zip(outcome_1, outcome_2)]}"
        assert draw_1 == draw_2, f"Draw mismatch. Got {draw_1} and {draw_2}"
        draw_scikit, _ = classifier.check_draw(rho_1[0].tolist())
        if np.argmax(rho_1) == y and not draw_scikit:
            acc_scikit += 1
        if np.argmax(outcome_1) == y and not draw_1:
            acc_pyals += 1
        if (np.argmax(outcome_1) != np.argmax(rho_1)) and (draw_1 != draw_scikit):
            mismatches.append((', '.join(str(s) for s in score_1), draw_1, ', '.join(str(s) for s in outcome_1), np.argmax(outcome_1), ', '.join(f'{q:.2f}' for q in rho_1[0]), np.argmax(rho_1), y))
    if mismatches:
        logger.info(f'Mismatches in models:\n{tabulate(mismatches, headers=["Score", "Draw", "Outcome", "argmax", "Scikit Rho", "argmax", "Label"])}')
        logger.info(f"{len(mismatches)} mismatches")
    logger.info(f"Accuracy of the pyALS model: {acc_pyals / len(y_test) * 100}%")
    logger.info(f"Accuracy of the scikit-learn model: {acc_scikit / len(y_test)*100}%")

def training_with_parameter_tuning(clf, tuning, dataset, configfile, outputdir, fraction, ntrees, niter, cross_validate):
    logger = logging.getLogger("pyALS-RF")
    config = DtGenConfigParser(configfile)
    x_train, y_train, x_test, y_test = get_sets(dataset, config, fraction)
    search_grid = { 'max_features': [None, 'log2', 'sqrt'],
                    'criterion' : ["gini", "entropy", "log_loss"],
                    'max_depth': [int(x) for x in np.linspace(6, 100, 2)],
                    'min_samples_split': [int(x) for x in np.linspace(10, 100, 1)],
                    'min_samples_leaf': [int(x) for x in np.linspace(10, 100, 1)],
                    'ccp_alpha' : np.arange(0.01, 0.5, 0.02),
                    'bootstrap': [True, False]}
    estimator = RandomForestClassifierMV(n_estimators = ntrees)
    if clf == "dt":
        pass
    else:
        if tuning == "random":
            rf_random = RandomizedSearchCV(estimator = estimator, param_distributions = search_grid, n_iter = niter, cv = 3, verbose = 1, random_state = np.random.default_rng().integers(0, 100), n_jobs = -1)
        else:
            rf_random = GridSearchCV(estimator = estimator, param_grid = search_grid, cv = 3, verbose = 1, n_jobs = -1)
        rf_random.fit(x_train, y_train)
        logger.info(f'Best parameters:\n{tabulate([ [k, v] for k, v in rf_random.best_params_.items()])}')
        data = [ [i, estimator.tree_.node_count, estimator.tree_.max_depth ] for i, estimator in enumerate(rf_random.best_estimator_.estimators_) ]
        logger.info(f'DTs Stats.:\n{tabulate(data, headers=["tree", "#nodes", "depth"])}\n')
        save_model(outputdir, config, rf_random.best_estimator_, rf_random.best_params_, x_train, y_train, x_test, y_test, cross_validate)
        
def dtgen(clf, dataset, configfile, outputdir, fraction, depth, predictors, criterion, min_sample_split, min_samples_leaf, max_features, max_leaf_nodes, min_impurity_decrease, ccp_alpha, disable_bootstrap, cross_validate):
    logger = logging.getLogger("pyALS-RF")
    config = DtGenConfigParser(configfile)
    x_train, y_train, x_test, y_test = get_sets(dataset, config, fraction)
    if clf == "dt":
        model = DecisionTreeClassifier(max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha).fit(x_train, y_train)
        logger.info(f"Node count: {model.tree_.node_count()}")
        logger.info(f"Depth: {model.tree_.max_dept}")
    elif clf == "rf":
        model = RandomForestClassifierMV(n_estimators = predictors, max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha, bootstrap = not disable_bootstrap, n_jobs = -1, verbose = 0).fit(x_train, y_train)
        #model = RandomForestClassifier(n_estimators = predictors, max_depth = depth, criterion = criterion, min_samples_split = min_sample_split, min_samples_leaf = min_samples_leaf, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, ccp_alpha = ccp_alpha, bootstrap = not disable_bootstrap, n_jobs = -1, verbose = 1).fit(x_train, y_train)
        data = [ [i, estimator.tree_.node_count, estimator.tree_.max_depth ] for i, estimator in enumerate(model.estimators_) ]
        logger.info(f'DTs Stats.:\n{tabulate(data, headers=["Tree", "#nodes", "depth"])}')
    #Todo Store training parameters on file (in place of None)
    save_model(outputdir, config, model, None, x_train, y_train, x_test, y_test, cross_validate)    
    