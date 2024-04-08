"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>, Antonio Emmanuele <antonio.emmanuele@unina.it>

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
from numpy import ndarray
from anytree import Node, RenderTree, AsciiStyle
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from pyalslib import list_partitioning
from .RegressorTree import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
#from ..scikit.RandonForestClassifierMV import RandomForestClassifierMV
import xgboost as xgb
import re

class Regressor:
    __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_4'}

    def __init__(self, ncpus = None, use_espresso = False, learning_rate: float = None):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        self.ncpus = min(ncpus, cpu_count()) if ncpus is not None else cpu_count()
        self.use_espresso = use_espresso
        self.als_conf = None
        self.pool = Pool(self.ncpus)
        self.learning_rate = learning_rate
        # Multiplicative constant to mult to the combination of all tree outputs.
        # For RF is equal to 1 / (number of trees) and for xgb is eq to 1
        # The other costant in per_tree_weight that 
        self.sum_weight = 0
        
    def __del__(self):
        self.pool.close()
    
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
            if self.learning_rate == None:
                assert 1 == 0, "invalid parser format"
            else:
                # For xgboost classifier the dataset description can be used as model feature, a list
                # with [{"name", "value"}]
                self.booster_joblib_parser(model_source, dataset_description)
            # self.joblib_parser(model_source, dataset_description)
        
        # The weight of the first model is always 1.
        tree_weight_map = [(self.trees[0], 1)]
        self.ncpus = min(self.ncpus, len(self.trees))
        # If the learning rate is none then the model is a rf and 
        # the sum weight is 1/ num_tree 
        # and the tree weight is 1.
        if self.learning_rate == None:
            self.sum_weight     = 1/len(self.trees)
            # Generate a mapping among each tree and its weight
            tree_weight_map     = [(self.trees[idx], 1) for idx in range(1,len(self.trees))]
        # Otherwise it is an XGB model so the sum weight is 1 and the tree weight
        # is equal to the learning rate.
        else:
            self.sum_weight = 1
            # Generate a mapping among each tree and its weight
            tree_weight_map     = [(self.trees[idx], self.learning_rate) for idx in range(1,len(self.trees))]
        self.p_tree = list_partitioning(self.trees, self.ncpus)
        self.args = [[t, None] for t in self.p_tree]

    def pmml_parser(self, pmml_file_name, dataset_description = None):
        logger = logging.getLogger("pyALS-RF")
        logger.debug(f"Parsing {pmml_file_name}")
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        tree = ElementTree.parse(pmml_file_name)
        root = tree.getroot()
        self.__namespaces["pmml"] = Regressor.get_xmlns_uri(root)
        self.get_features_and_classes_from_pmml(root)

        if dataset_description is not None:
            self.classes_name = dataset_description.classes_name
        else:
            self.classes_name = self.model_classes
        segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        # exit(1)
        if segmentation is not None:
            for tree_id, segment in enumerate(segmentation.findall("pmml:Segment", self.__namespaces)):
                print(f"Tree Id {tree_id} Segment {segment}")
                logger.debug(f"Parsing tree {tree_id}... ")
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
                tree = self.get_tree_model_from_pmml(str(tree_id), tree_model_root)
                #print(tree.dump())
                self.trees.append(tree)
                logger.debug(f"Done parsing tree {tree_id}")
        else:
            tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find(
                "pmml:Node", self.__namespaces)
            tree = self.get_tree_model_from_pmml("0", tree_model_root)
            # exit(1)
            self.trees.append(tree)
        logger.debug(f"Done parsing {pmml_file_name}")
        

    # def joblib_parser(self, joblib_file_name, dataset_description):
    #     logger = logging.getLogger("pyALS-RF")
    #     logger.info(f"Parsing {joblib_file_name}")
    #     self.trees = []
    #     self.model_features = []
    #     self.model_classes = []
    #     self.classes_name = []
    #     model = joblib.load(joblib_file_name)
    #     self.classes_name = dataset_description.classes_name
    #     self.model_classes = dataset_description.classes_name
    #     self.model_features = [ {"name": f, "type": "double" } for f in dataset_description.attributes_name ]
    #     if isinstance(model, (RandomForestClassifier, RandomForestClassifierMV)):
    #         for i, estimator in enumerate(model.estimators_):
    #             logger.debug(f"Parsing tree_{i}")
    #             root_node = self.get_tree_model_from_joblib(estimator)
    #             self.trees.append(DecisionTree(f"tree_{i}", root_node, self.model_features, self.model_classes, self.use_espresso))
    #             logger.debug(f"Done parsing tree_{i}")
    #     elif isinstance(model, DecisionTreeClassifier):
    #         root_node = self.get_tree_model_from_joblib(model)
    #         self.trees.append(DecisionTree(f"tree_0", root_node, self.model_features, self.model_classes, self.use_espresso))
    #     logger.info(f"Done parsing {joblib_file_name}")

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
        self.x_test = self.dataframe.loc[:, self.dataframe.columns != "Outcome"].values
        self.y_test = self.dataframe.loc[:, self.dataframe.columns == "Outcome"].values
        for arg in self.args:
            arg[1] = self.x_test
    
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

    # def get_score(self, x):
    #     outcomes = [ t.visit(x) for t in self.trees ]
    #     return [sum(s) for s in zip(*outcomes)]
    
    @staticmethod
    def check_draw(scores):
        r = np.sort(np.array(scores, copy=True))[::-1]
        return r[0] == r[1], r[0]
    
    @staticmethod
    def compute_score(trees : list, x_test : ndarray):
        assert len(np.shape(x_test)) == 2
        # Return the sum of predictions of the assigned trees for each sample in x_test
        # Each prediction is multiplied by the weight of the specific tree, associated by the struct 
        # p_trees
        return [np.sum([t.visit(x) for t in trees]) for x in x_test ]
    
    # The prediction is :
    # sum_weight * (prediction_tree[0] * tree[0].weight + prediction_tree[1] * tree[1].weight + ..)
    # If the model is a RF then:
    # sum_weight = 1/num_trees
    # weight[i] = 1
    # If the model is XGB:
    # sum_weight = 1
    # weight[i] = learning_rate if i != 0 else 1
    def predict(self, x_test : ndarray):
        if len(np.shape(x_test)) == 1:
            np.reshape(x_test, np.shape(x_test)[0])
        # The args are the tree per cpu and the totality of samples.
        args = [[t, x_test] for t in self.p_tree]
        # evals contains the sum of predictions for each tree in each cpu.
        # If the trees are 8 and the cpu are 4 then evals contains (4,len(x_test)) 
        # Each row is composed by the sum of values per CPU, so the np.sums sum these predictions 
        # for each sample while the final division outputs the mean per sample.
        return np.sum(self.pool.starmap(Regressor.compute_score, args), axis = 0) * self.sum_weight


    def evaluate_test_dataset(self):
        outcomes = np.sum(self.pool.starmap(Regressor.compute_score, self.args), axis = 0)
        return np.sum(tuple( np.argmax(o) == y[0] and not Regressor.check_draw(o)[0] for o, y in zip(outcomes, self.y_test))) / len(self.y_test) * 100
    
    def predict_dump(self, index: int, outfile: str):
        score = self.predict(self.x_test[index])
        draw, max_score = Regressor.check_draw(score)
        outcome = [int(s == max_score) for s in score]
        data = {
            "score": score,
            "draw": int(draw),
            "outcome": dict(zip(self.classes_name, outcome)),
            "trees": {t.name: {"outcome": {k: int(v) for k, v in zip(self.classes_name, t.visit(self.x_test[index]))}} for t in self.trees}}
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
        return RegressorTree(tree_name, tree, self.model_features, self.model_classes, self.use_espresso)
    
    def get_tree_nodes_from_pmml_recursively(self, element_tree_node, parent_tree_node, id=0):
        print(f"Invoked for node {id}")
        children = element_tree_node.findall("pmml:Node", self.__namespaces)
        assert len(children) == 2, f"Only binary trees are supported. Aborting. {children}"
        for child in children:
            boolean_expression = parent_tree_node.boolean_expression
            if boolean_expression:
                boolean_expression += " and "
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
                    boolean_expression += f"not {parent_tree_node.name}"
            if child.find("pmml:Node", self.__namespaces) is not None:
                new_tree_node = Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent = parent_tree_node, feature = "", operator = "", threshold_value = "", boolean_expression = boolean_expression)
                self.get_tree_nodes_from_pmml_recursively(child, new_tree_node, id + 1)
            else:
                print(f"Expression: {boolean_expression}")
                print(f"Score {child.attrib['score']}")
                if "id" in child.attrib:
                    print(f"ID in child, Continuing with node {child.attrib['id']}")
                else :
                    print(f"ID not in child, Continuing with node {id}")

                Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent = parent_tree_node, score = child.attrib['score'], boolean_expression = boolean_expression)
                
    # def get_tree_model_from_joblib(self, clf : DecisionTreeClassifier):
    #     # Initializza il numero di nodi
    #     n_nodes = clf.tree_.node_count
    #     # Prendi il figlio di sinistra.
    #     children_left = clf.tree_.children_left
    #     # prendi il figlio di destra.
    #     children_right = clf.tree_.children_right
    #     # Prendi la feature.
    #     feature = clf.tree_.feature
    #     # Prendi la threshold.
    #     threshold = clf.tree_.threshold
    #     # Prendi il valore dell'albero.
    #     values = clf.tree_.value
    #     node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    #     is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    #     # Prendi il primo nodo ((fittizio)
    #     root_node = Node("Node_0", feature="", operator="", threshold_value="", boolean_expression="")
    #     stack = [(0, 0, root_node)]  # start with the root node id (0) and its depth (0)
    #     # Finchè ci sono nodi sullo stack
    #     while len(stack) > 0:
    #         # Prendi il nodo, il suo id, la sua profondità, e l'oggetto nodo.
    #         # `pop` ensures each node is only visited once
    #         current_node_id, depth, current_node = stack.pop()
    #         node_depth[current_node_id] = depth
    #         # If the left and right child of a node is not the same we have a split node
    #         is_split_node = children_left[current_node_id] != children_right[current_node_id]
    #         if is_split_node:
    #             # If a split node, append left and right children and depth to `stack` so we can loop through them
    #             current_node.feature = self.model_features[feature[current_node_id]]["name"]
    #             #* sklearn only supports the <= (less or equal), which is not supported by pyALS-rf.
    #             #* For this reason, boolean expressions are generated by reversing the comparison condition
    #             current_node.operator = 'greaterThan' 
    #             current_node.threshold_value =threshold[current_node_id]
    #             boolean_expression = current_node.boolean_expression
    #             if len(boolean_expression) > 0:
    #                 boolean_expression += " and "
    #             child_l = Node(f"Node_{children_left[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}not {current_node.name}")
    #             stack.append((children_left[current_node_id], depth + 1, child_l))
    #             child_r = Node(f"Node_{children_right[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}{current_node.name}")
    #             stack.append((children_right[current_node_id], depth + 1, child_r))
    #         else:
    #             current_node.score = self.model_classes[np.argmax(values[current_node_id])]
    #             is_leaves[current_node_id] = True
    #     return root_node
    
    # This function takes as input the joblib file of a xgb regressors
    # and parses the model obtaining the pyALS-regressor        
    def booster_joblib_parser(self, joblib_file_name, model_features):
        # Load the model
        xgb_regressor = joblib.load(joblib_file_name)
        # Obtain the booster
        booster = xgb_regressor.get_booster()
        # Obtain the booster and its "string-dump"
        booster_dump = booster.get_dump()
        # Obtain the model features
        #self.model_features = [{"name" : f"f{i}", "type":"double"}  for i in range(0,250)]
        self.model_features = model_features
        # For each tree
        weight = 1 
        for i, tree in enumerate(booster_dump):
            print(f"Albero {i+1}:\n{tree}\n")
            # Obtain a list of lines per tree
            tree_in_lines = [line.strip('\t') for line in tree.split('\n') if line.strip('\t')]
            print(f"Albero {i+1}:\n{tree_in_lines}\n")
            # Get the tree from the joblib
            root_node = self.booster_tree_joblib_parser(tree_in_lines)
            tree = RegressorTree(name = f"Tree_{i}", root_node = root_node, features = self.model_features, classes = self.model_classes, use_espresso = self.use_espresso, weight = weight)
            #tree.dump()
            self.trees.append(tree)
            weight = self.learning_rate
            #exit(1)
        
    # Taken a line such as User '0:[f249<-0.642892063] yes=1,no=2,missing=1', this function returns 
    # the integer before :, i.e. the node id, which is in this case 0.
    @staticmethod
    def booster_get_node_id(node_line):
        match = re.search(r"(\d+):", node_line)
        if match:
            return int(match.group(1))  
        else:
            assert 1 == 0, "Invalid xgb string"
    
    # Given a line of a boosting tree returns true if the associated node is a leaf.
    # since the string of a leaf contains leaf={leaf_value} then the function simply
    # searchs for string in the line
    @staticmethod 
    def booster_is_leaf(node_line):
        return "leaf" in node_line
    
    # Each node in the tree is rappresented as a string ( the entire tree) is a list of string (node_lines).
    # this function returns the index on the list where the node_id is equal to req_id
    @staticmethod
    def booster_find_node(node_lines, req_id):
        for index, line in enumerate(node_lines):
            if Regressor.booster_get_node_id(line) == req_id:
                return index
            
    # Parse a joblib xgboost tree using its dump.
    # nodes is a list of string, where each string is a tree node.
    # The function parses the tree by tree levels:
    # while explored_nodes < len(nodes):
    #   node = explore_node_by_level
    def booster_tree_joblib_parser(self, nodes):
        first_non_leaf  = nodes[0]
        # The first id
        base_start = 0
        # The level used during the exploration
        current_level   = 1
        # The number of leaves in the precedent level.
        leaves_precedent_level = 0
        feature, thd = Regressor.booster_extract_feature_thd(nodes[0])
        # Initialize the first node, operator is always lessThan.
        tree = Node(f"Node_{0}", feature = feature, operator="lessThan", threshold_value=thd, boolean_expression="")
        # The bitmask of leaves of the previous level
        # the leaves bitmask contains : 
        # The identifier of a node, True if the node is a leaf, the pointer to the node.
        leaves_bitmask = [(0, False, tree)]
        while base_start < len(nodes) - 1: # BS starts from 0, so, if the ub is 13 then base_start will reach 12
            print(f"BS {base_start} leaves {len(nodes)}")
            base_start, current_level, leaves_precedent_level, leaves_bitmask = Regressor.explore_level(nodes, 
                                                                                                        current_level = current_level,
                                                                                                        base_start = base_start, 
                                                                                                        leaves_precedent = leaves_precedent_level, 
                                                                                                        leaves_mask = leaves_bitmask)
            
        
        # Return the pointer to the root node.
        return tree

    # Given the line of a node, which is not a leaf, extract the 
    # feature and the thd.
    @staticmethod
    def booster_extract_feature_thd(node_line):
        pattern = r'\[(.+)<(-?\d+\.\d+)\]'
        match = re.search(pattern, node_line)
        if match:
            feature = match.group(1)
            thd = float(match.group(2))
            return feature, thd 
        else:
            assert 1 == 0, " Error during parsing"
    
    # Given a leaf node, extract the value of the leaf that comes after
    # the value leaf={leaf_score}.
    @staticmethod
    def booster_extract_score(leaf_line):
        parts = leaf_line.split('=')
        score = parts[1].strip()
        return float(score)
    
    # Explore a specific level of the tree
    # nodes:            list of strings each one rappresent a specific node.
    # current_level:    level being explored.
    # leaves_precedent: number of leaves in the precedent explored level.
    # leaves mask:      Mask of leaves.
    @staticmethod
    def explore_level(nodes, current_level, base_start, leaves_precedent, leaves_mask):

        # To make the association with a father node each leaf has a counter 
        def get_father_node(f_slots, l_mask):
            for idx in range(0,len(f_slots)):
                # The Node in the slot is the father and the leaf is a 
                # left children
                if f_slots[idx] == 2:
                    f_slots[idx] = f_slots[idx] - 1
                    return l_mask[idx][2], True
                # Otherwise it is a right children
                elif f_slots[idx] == 1:
                    f_slots[idx] = f_slots[idx] - 1
                    return l_mask[idx][2], False

        # The number of nodes in a level can be seen as 2 * number of non leafs of previous level
        nodes_per_level = 0
        for idx in range(0,len(leaves_mask)):
            if leaves_mask[idx][1] == False:
                nodes_per_level += 1
        nodes_per_level = 2 * nodes_per_level
        # lines in level is the numebr of node lines in the specific regressor.
        lines_in_level = {}
        # Maximum number of while iterations
        ub = base_start + nodes_per_level
        # New mask for leaves node in this level
        new_leaves_mask = []
        # Counter of leaves in this level.
        new_leaves_predecent = 0
        # Available slots for children father association, 
        # each children node is associated with the first available father e.g. the first node
        # having 2 available slots.
        father_slots = [2 if l[1] == False else 0 for l in leaves_mask]
        # Identify all the nodes and all the leaves
        while base_start < ub:
            # First update the base start to the next node
            base_start += 1
            print(f"BS {base_start} UB {ub}")
            # Identify the node 
            idx = Regressor.booster_find_node(node_lines = nodes, req_id = base_start)
            # Save the line 
            lines_in_level.update({Regressor.booster_get_node_id(nodes[idx]) : nodes[idx]})
            # Obtain the father node:
            father_node, is_left = get_father_node(father_slots, leaves_mask) 
            # Left is true in xgb
            if is_left: 
                if father_node.boolean_expression == "":
                    boolean_expression = father_node.name
                else:
                    boolean_expression = father_node.boolean_expression + " and " + father_node.name
            else:
                if father_node.boolean_expression == "":
                    boolean_expression = "not " + father_node.name
                else:
                    boolean_expression = father_node.boolean_expression + " and not " + father_node.name

            # Now update the leaves mask
            is_leaf = Regressor.booster_is_leaf(nodes[idx])    
            if not is_leaf:
                # Extract the feature and thd.
                feature, thd = Regressor.booster_extract_feature_thd(nodes[idx])
                # Generate the node.
                tree = Node(f"Node_{Regressor.booster_get_node_id(nodes[idx])}", feature = feature, parent = father_node, operator = "lessThan", threshold_value = thd, boolean_expression=boolean_expression)
                print(f"Node: {tree.name} Feature: {tree.feature} Thd: {tree.threshold_value} Boolean {tree.boolean_expression}")
                # Generate the new mask.
                new_leaves_mask.append((Regressor.booster_get_node_id(nodes[idx]) , is_leaf, tree))
            else:                
                # Increase the counter of leaves
                new_leaves_predecent += 1
                # Extract the score
                score = Regressor.booster_extract_score(nodes[idx])
                # Generate the node.
                tree = Node(f"Node_{Regressor.booster_get_node_id(nodes[idx])}", feature = "", operator = "", parent = father_node, threshold_value = "", boolean_expression = boolean_expression, score = score)
                print(f"Leaf: {tree.name} Score: {tree.score} Boolean {tree.boolean_expression}") 
                # Generate the new mask.
                new_leaves_mask.append((Regressor.booster_get_node_id(nodes[idx]) , is_leaf, tree))
            
        return base_start, current_level + 1, new_leaves_predecent, new_leaves_mask
    