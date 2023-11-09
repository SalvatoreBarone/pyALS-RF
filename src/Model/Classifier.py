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
import numpy as np, pandas as pd, random, json5, joblib, logging
from xml.etree import ElementTree
from anytree import Node, RenderTree, AsciiStyle
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from pyalslib import list_partitioning
from .DecisionTree import *
from .rank_based import softmax, giniImpurity

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from ..scikit.RandonForestClassifierMV import RandomForestClassifierMV

class Classifier:
    __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_4'}

    def __init__(self, ncpus = None, use_espresso = False):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        self.ncpus = min(ncpus, cpu_count()) if ncpus is not None else cpu_count()
        self.use_espresso = use_espresso
        self.args = None
        self.p_tree = None
        self.pool = None
        self.als_conf = None

    def __deepcopy__(self, memo=None):
        classifier = Classifier(self.als_conf)
        classifier.trees = copy.deepcopy(self.trees)
        classifier.model_features = copy.deepcopy(self.model_features)
        classifier.model_classes = copy.deepcopy(self.model_classes)
        classifier.classes_name = copy.deepcopy(self.classes_name)
        classifier.ncpus = self.ncpus
        classifier.p_tree = None
        classifier.args = None
        classifier.pool = None
        classifier.als_conf = None
        return classifier
    
    @staticmethod
    def get_xmlns_uri(elem):
        if elem.tag[0] == "{":
            uri, ignore, tag = elem.tag[1:].partition("}")
        else:
            uri = None
        return uri
    
    def parse(self, model_source : str, dataset_description = None):
        if model_source.endswith(".pmml"):
            self.pmml_parser(model_source, dataset_description)
        elif model_source.endswith(".joblib"):
            self.joblib_parser(model_source, dataset_description)

    def pmml_parser(self, pmml_file_name, dataset_description = None):
        logger = logging.getLogger("pyALS-RF")
        logger.debug(f"Parsing {pmml_file_name}")
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        tree = ElementTree.parse(pmml_file_name)
        root = tree.getroot()
        self.__namespaces["pmml"] = Classifier.get_xmlns_uri(root)
        self.get_features_and_classes_from_pmml(root)
        if dataset_description is not None:
            self.classes_name = dataset_description.classes_name
        else:
            self.classes_name = self.model_classes
        segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        if segmentation is not None:
            for tree_id, segment in enumerate(segmentation.findall("pmml:Segment", self.__namespaces)):
                logger.debug(f"Parsing tree {tree_id}... ")
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
                tree = self.get_tree_model_from_pmml(str(tree_id), tree_model_root)
                self.trees.append(tree)
            logger.debug("\rDone")
        else:
            tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find(
                "pmml:Node", self.__namespaces)
            tree = self.get_tree_model_from_pmml("0", tree_model_root)
            self.trees.append(tree)
        logger.debug(f"Done parsing {len(self.trees)} trees")
        self.ncpus = min(self.ncpus, len(self.trees))

    def joblib_parser(self, joblib_file_name, dataset_description):
        logger = logging.getLogger("pyALS-RF")
        logger.debug(f"Parsing {joblib_file_name}")
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        model = joblib.load(joblib_file_name)
        self.classes_name = dataset_description.classes_name
        self.model_classes = dataset_description.classes_name
        self.model_features = [ {"name": f, "type": "double" } for f in dataset_description.attributes_name ]
        if isinstance(model, (RandomForestClassifier, RandomForestClassifierMV)):
            for i, estimator in enumerate(model.estimators_):
                logger.debug(f"Parsing tree_{i}")
                root_node = self.get_tree_model_from_joblib(estimator)
                self.trees.append(DecisionTree(f"tree_{i}", root_node, self.model_features, self.model_classes, self.use_espresso))
        elif isinstance(model, DecisionTreeClassifier):
            root_node = self.get_tree_model_from_joblib(model)
            self.trees.append(DecisionTree(f"tree_0", root_node, self.model_features, self.model_classes, self.use_espresso))
        logger.debug(f"Done parsing {len(self.trees)} trees")
        self.ncpus = min(self.ncpus, len(self.trees))

    def dump(self):
        print("Features:")
        for f in self.model_features:
            print("\tName: ", f["name"], ", Type: ", f["type"])
        print("\n\nClasses:")
        for c in self.model_classes:
            print("\tName: ", c)
        print("\n\nTrees:")
        for t in self.trees:
            t.dump()
            
    def read_test_set(self, dataset_csv):
        self.dataframe = pd.read_csv(dataset_csv, sep = ";")
        attribute_name = list(self.dataframe.keys())[:-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        name_matches = [ a == f for a, f in zip(attribute_name, f_names) ]
        assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
        self.x_test = self.dataframe.loc[:, self.dataframe.columns != "Outcome"].values.tolist()
        self.y_test = sum(self.dataframe.loc[:, self.dataframe.columns == "Outcome"].values.tolist(), [])
        self.x_val = self.x_test
        self.y_val = self.y_test
        
    def read_training_set(self, dataset_csv):
        self.dataframe = pd.read_csv(dataset_csv, sep = ";")
        attribute_name = list(self.dataframe.keys())[:-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        assert attribute_name == f_names, f"{attribute_name} != {f_names}"
        self.x_train = self.dataframe.loc[:, self.dataframe.columns != "Outcome"].values.tolist()
        self.y_train = sum(self.dataframe.loc[:, self.dataframe.columns == "Outcome"].values.tolist(), [])
        
    def split_test_dataset(self):
        validation_set = random.choices(range(len(self.x_test)), k = len(self.x_test) // 2)
        self.x_val = [ self.x_test[i] for i in range(len(self.x_test)) if i in validation_set ]
        self.y_val = [ self.y_test[i] for i in range(len(self.y_test)) if i in validation_set ]
        self.x_test = [ self.x_test[i] for i in range(len(self.x_test)) if i not in validation_set ]
        self.y_test = [ self.y_test[i] for i in range(len(self.y_test)) if i not in validation_set ]
        self.args = [[t, self.x_test, False] for t in self.p_tree] if self.p_tree is not None else None
    
    def brace4ALS(self, als_conf):
        if self.als_conf is None:
            self.als_conf = als_conf
            for t in self.trees:
                t.brace4ALS(als_conf)

    def reset_nabs_configuration(self):
        self.set_nabs({f["name"]: 0 for f in self.model_features})

    def reset_assertion_configuration(self):
        if self.als_conf is None:
            for t in self.trees:
                t.reset_assertion_configuration()

    def set_nabs(self, nabs):
        for tree in self.trees:
            tree.set_nabs(nabs)

    def set_assertions_configuration(self, configurations):
        if self.als_conf is None:
            for t, c in zip(self.trees, configurations):
                t.set_assertions_configuration(c)

    def set_first_stage_approximate_implementations(self, configuration):
        if self.als_conf is None:
            for t, c in zip(self.trees, configuration):
                t.set_first_stage_approximate_implementations(c)

    def get_num_of_trees(self):
        return len(self.trees)

    def get_total_bits(self):
        return sum(t.get_total_bits() for t in self.trees)

    def get_total_retained(self):
        return sum(t.get_total_retained() for t in self.trees)

    def get_als_cells_per_tree(self):
        return [len(t.get_graph().get_cells()) for t in self.trees]

    def get_als_dv_upper_bound(self):
        ub = []
        for t in self.trees:
            ub.extend(iter(t.get_als_dv_upper_bound()))
        return ub

    def get_assertions_configuration(self):
        return [t.get_assertions_configuration() for t in self.trees] if self.als_conf is not None else []

    def get_assertions_distance(self):
        return [t.get_assertions_distance() for t in self.trees] if self.als_conf is not None else []

    def get_current_required_aig_nodes(self):
        return [t.get_current_required_aig_nodes() for t in self.trees] if self.als_conf is not None else []

    def get_num_of_first_stage_approximate_implementations(self):
        return [len(t.get_first_stage_approximate_implementations()) - 1 for t in self.trees] if self.als_conf is not None else []

    def get_struct(self):
        return [tree.get_struct() for tree in self.trees]

    def enable_mt(self):
        self.p_tree = list_partitioning(self.trees, self.ncpus)
        self.args = [[t, self.x_test, False] for t in self.p_tree]
        self.pool = Pool(self.ncpus)

    def evaluate_test_dataset(self, use_pruning = False):
        logger = logging.getLogger("pyALS-RF")
        if self.args is None:
            logger.warn("Multi-threading is disabled. To enable it, call the enable_mt() member of the Classifier class")
            accuracy = 0
            for x, y in tqdm(zip(self.x_test, self.y_test), total=len(self.y_test), desc="Computing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
                outcome, draw = self.predict(x, use_pruning)
                if not draw and np.argmax(outcome) == y:
                    accuracy += 1
            return accuracy / len(self.y_test) * 100
        
        for a in self.args:
            a[2] = use_pruning
        outcomes = self.pool.starmap(Classifier.compute_score, self.args)
        return sum(np.argmax(score := [sum(s) for s in zip(*scores)]) == y and not self.check_draw(score)[0] for scores, y in zip(zip(*outcomes), self.y_test)) / len(self.y_test) * 100
    
    def get_assertion_activation(self, use_training_data):
        samples = self.x_train if use_training_data else self.x_val
        labels = self.y_train if use_training_data else self.y_val
        activity_by_sample = []
        nclasses = len(self.model_classes)
        ntrees = len(self.trees)
        for x, y in tqdm( zip(samples, labels), total=len(labels), desc="Computing assertions' activation...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            outcome = {"x" : x, "y": str(y), "redundancy" : 0, "rho": np.zeros((nclasses,), dtype=int), "Ig" : 0, "outcomes" : {}}
            for t in self.trees:
                predicted_class, active_assertion, mask = t.get_assertion_activation(x)
                cost = len(active_assertion.split("and"))
                outcome["rho"] += mask
                outcome["outcomes"][t.name] = {"assertion" : active_assertion, "cost": cost, "correct" : predicted_class == str(y)}
            outcome["Ig"] = giniImpurity(softmax(outcome["rho"]))
            outcome["redundancy"] = int(sum(i["correct"] for i in outcome["outcomes"].values()) - np.ceil(ntrees/2))
            outcome["rho"] = outcome["rho"].tolist()
            activity_by_sample.append(outcome) 
        return activity_by_sample    
    
    def set_pruning(self, pruning):
        for t in self.trees:
            t.set_pruning(pruning, self.use_espresso)
            
    def get_assertions_cost(self):
        return sum( t.get_assertions_cost() for t in self.trees )
    
    def get_pruned_assertions_cost(self):
        return sum( t.get_pruned_assertions_cost() for t in self.trees )

    @staticmethod
    def compute_score(trees, x_test, pruning):
        scores = []
        for x in x_test:
            outcomes = [ t.visit(x, pruning) for t in trees ]
            scores.append([sum(s) for s in zip(*outcomes)])
        return scores

    def get_score(self, x, use_pruning = False):
        outcomes = [ t.visit(x, use_pruning) for t in self.trees ]
        return [sum(s) for s in zip(*outcomes)]
    
    def check_draw(self, scores):
        m = np.max(scores)
        return (m <= (len(self.trees) // len(self.model_classes))) or np.sum(s == m for s in scores) > 1, m
    
    def predict(self, x, use_pruning=False):
        score = self.get_score(x, use_pruning)
        draw, max_score = self.check_draw(score)
        return [int(s == max_score) for s in score] , draw
    
    def predict_dump(self, index: int, outfile: str, use_pruning=False):
        score = self.get_score(self.x_test[index], use_pruning)
        draw, max_score = self.check_draw(score)
        outcome = [int(s == max_score) for s in score]
        data = {
            "score": score,
            "draw": int(draw),
            "outcome": dict(zip(self.classes_name, outcome)),
            "trees": {t.name: {"outcome": {k: int(v) for k, v in zip(self.classes_name, t.visit(self.x_test[index], use_pruning))}} for t in self.trees}}
        with open(outfile, "w") as f:
            json5.dump(data, f, indent=2)
    
    def get_features_and_classes_from_pmml(self, root):
        for child in root.find("pmml:DataDictionary", self.__namespaces).findall('pmml:DataField', self.__namespaces):
            if child.attrib["optype"] == "continuous":
                # the child is PROBABLY a feature
                self.model_features.append({
                    "name": child.attrib['name'].replace('-', '_'),
                    "type": "double" if child.attrib['dataType'] == "double" else "int"})
            elif child.attrib["optype"] == "categorical":
                # the child PROBABLY specifies model-classes
                for element in child.findall("pmml:Value", self.__namespaces):
                    self.model_classes.append(element.attrib['value'].replace('-', '_'))

    def get_tree_model_from_pmml(self, tree_name, tree_model_root, id=0):
        tree = Node(f"Node_{tree_model_root.attrib['id']}" if "id" in tree_model_root.attrib else f"Node_{id}", feature="", operator="", threshold_value="", boolean_expression="")
        self.get_tree_nodes_from_pmml_recursively(tree_model_root, tree, id)
        return DecisionTree(tree_name, tree, self.model_features, self.model_classes, self.use_espresso)

    def get_tree_nodes_from_pmml_recursively(self, element_tree_node, parent_tree_node, id=0):
        children = element_tree_node.findall("pmml:Node", self.__namespaces)
        assert len(children) == 2, f"Only binary trees are supported. Aborting. {children}"
        for child in children:
            boolean_expression = parent_tree_node.boolean_expression
            if boolean_expression:
                boolean_expression += " & "
            predicate = None
            if compound_predicate := child.find("pmml:CompoundPredicate", self.__namespaces):
                predicate = next(item for item in compound_predicate.findall("pmml:SimplePredicate", self.__namespaces) if item.attrib["operator"] != "isMissing")
            else:
                predicate = child.find("pmml:SimplePredicate", self.__namespaces)
            if predicate is not None:
                feature = predicate.attrib['field'].replace('-', '_')
                operator = predicate.attrib['operator']
                threshold_value = predicate.attrib['value']
                if operator in ('equal', 'lessThan', 'greaterThan'):
                    parent_tree_node.feature = feature
                    parent_tree_node.operator = operator
                    parent_tree_node.threshold_value = threshold_value
                    boolean_expression += parent_tree_node.name
                else:
                    boolean_expression += f"~{parent_tree_node.name}"
            if child.find("pmml:Node", self.__namespaces) is not None:
                new_tree_node = Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent = parent_tree_node, feature = "", operator = "", threshold_value = "", boolean_expression = boolean_expression)
                self.get_tree_nodes_from_pmml_recursively(child, new_tree_node, id + 1)
            else:
                Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent = parent_tree_node, score = child.attrib['score'].replace('-', '_'), boolean_expression = boolean_expression)
                
    def get_tree_model_from_joblib(self, clf : DecisionTreeClassifier):
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        values = clf.tree_.value
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        
        root_node = Node("Node_0", feature="", operator="", threshold_value="", boolean_expression="")
        stack = [(0, 0, root_node)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            current_node_id, depth, current_node = stack.pop()
            node_depth[current_node_id] = depth
            # If the left and right child of a node is not the same we have a split node
            is_split_node = children_left[current_node_id] != children_right[current_node_id]
            if is_split_node:
                # If a split node, append left and right children and depth to `stack` so we can loop through them
                current_node.feature = self.model_features[feature[current_node_id]]["name"]
                #* sklearn only supports the <= (less or equal), which is not supported by pyALS-rf.
                #* For this reason, boolean expressions are generated by reversing the comparison condition
                current_node.operator = 'greaterThan' 
                current_node.threshold_value =threshold[current_node_id]
                boolean_expression = current_node.boolean_expression
                if len(boolean_expression) > 0:
                    boolean_expression += " & "
                child_l = Node(f"Node_{children_left[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}~{current_node.name}")
                stack.append((children_left[current_node_id], depth + 1, child_l))
                child_r = Node(f"Node_{children_right[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}{current_node.name}")
                stack.append((children_right[current_node_id], depth + 1, child_r))
            else:
                current_node.score = self.model_classes[np.argmax(values[current_node_id])]
                is_leaves[current_node_id] = True
        return root_node
                
