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
from numpy import ndarray
from anytree import Node, RenderTree, AsciiStyle
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from pyalslib import list_partitioning
from .DecisionTree import *
from .rank_based import softmax, giniImpurity

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from ..scikit.RandonForestClassifierMV import RandomForestClassifierMV

import time # To remove
from concurrent.futures import ThreadPoolExecutor, as_completed

""" This function was used during the inference testing.
    The main idea was to transform each boolean espression into an actual python boolean function.
    In this way, the inference time would have been increasingly improved removing the need to use the 
    eval function to evaluate boolean functions.
"""
def generate_boolean_function(expr, func_name):
    func_code = f"""
def {func_name}(minterms):
    return {expr}
"""
    exec(func_code, globals()) 
    return globals()[func_name] # Update the set of functions adding the boolean function evaluation.

def extract_nodes_from_assertion(assertion):
    """
    Extract all nodes (including 'not') from the given assertion function,
    excluding logical operators like 'and'.

    :param assertion: A string containing the logical assertion function.
    :return: A list of nodes including their negations (e.g., 'not Node_0').
    """
    # Define the regular expression pattern for matching 'not Node_X' or 'Node_X'
    pattern = r"(?:not\s+Node_\d+|Node_\d+)"
    
    # Find all matches
    matches = re.findall(pattern, assertion)
    
    # Return the list of nodes
    return matches

def clean_node_names_from_not(nodes):
    """
    Remove 'not' prefixes from a list of nodes.

    :param nodes: A list of nodes (e.g., ['not Node_0', 'Node_1']).
    :return: A list of cleaned node names (e.g., ['Node_0', 'Node_1']).
    """
    # Remove 'not ' from each node if it exists
    cleaned_nodes = [node.replace("not ", "") for node in nodes]
    
    return cleaned_nodes

class Classifier:
    __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_4'}

    def __init__(self, ncpus = None, use_espresso = False):
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        self.ncpus = min(ncpus, cpu_count()) if ncpus is not None else cpu_count()
        self.use_espresso = use_espresso
        self.als_conf = None
        
    def __del__(self):
        self.thd_pool.shutdown(wait = True)
        self.pool.close()
    
    @staticmethod
    def get_xmlns_uri(elem):
        if elem.tag[0] == "{":
            uri, ignore, tag = elem.tag[1:].partition("}")
        else:
            uri = None
        return uri
    
    """ This function takes in input the path of the stored model and the dataset_description.
        Depending on the format in which the model is stored (either PMML or Joblib) a different parser
        is invoked.
        At the end of the function the number of cpus used during inferences is setted. 
    """
    def parse(self, model_source : str, dataset_description = None):
        self.pool = Pool(self.ncpus)
        self.thd_pool = ThreadPoolExecutor(max_workers = self.ncpus)
        if model_source.endswith(".pmml"):
            self.pmml_parser(model_source, dataset_description)
        elif model_source.endswith(".joblib"):
            self.joblib_parser(model_source, dataset_description)
        self.ncpus = min(self.ncpus, len(self.trees))
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
        self.__namespaces["pmml"] = Classifier.get_xmlns_uri(root)
        self.get_features_and_classes_from_pmml(root)
        # Save other parameters
        if dataset_description is not None:
            self.classes_name = dataset_description.classes_name
            # In case the dataset is splitted (copied into the outdir folder during training)
            # then use the original separator
            if dataset_description.separated_training : 
                self.csv_separator = dataset_description.separator
            # Otherwise use the standard ";"
            else:
                self.csv_separator = ";"
            self.out_column = dataset_description.outcome_col
        else:
            self.classes_name = self.model_classes
            self.csv_separator = ";"
            self.out_column = -1
        segmentation = root.find("pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        if segmentation is not None:
            for tree_id, segment in enumerate(segmentation.findall("pmml:Segment", self.__namespaces)):
                logger.debug(f"Parsing tree {tree_id}... ")
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find("pmml:Node", self.__namespaces)
                tree = self.get_tree_model_from_pmml(str(tree_id), tree_model_root)
                self.trees.append(tree)
                logger.debug(f"Done parsing tree {tree_id}")
        else:
            tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find(
                "pmml:Node", self.__namespaces)
            tree = self.get_tree_model_from_pmml("0", tree_model_root)
            self.trees.append(tree)
        logger.debug(f"Done parsing {pmml_file_name}")


    def joblib_parser(self, joblib_file_name, dataset_description):
        logger = logging.getLogger("pyALS-RF")
        logger.info(f"Parsing {joblib_file_name}")
        self.trees = []
        self.model_features = []
        self.model_classes = []
        self.classes_name = []
        model = joblib.load(joblib_file_name)
        self.classes_name = dataset_description.classes_name
        self.model_classes = dataset_description.classes_name
        self.csv_separator = dataset_description.separator
        self.out_column = dataset_description.outcome_col
        self.model_features = [ {"name": f, "type": "double" } for f in dataset_description.attributes_name ]
        if isinstance(model, (RandomForestClassifier, RandomForestClassifierMV)):
            for i, estimator in enumerate(model.estimators_):
                logger.debug(f"Parsing tree_{i}")
                root_node = self.get_tree_model_from_joblib(estimator)
                self.trees.append(DecisionTree(f"tree_{i}", root_node, self.model_features, self.model_classes, self.use_espresso))
                logger.debug(f"Done parsing tree_{i}")
        elif isinstance(model, DecisionTreeClassifier):
            root_node = self.get_tree_model_from_joblib(model)
            self.trees.append(DecisionTree(f"tree_0", root_node, self.model_features, self.model_classes, self.use_espresso))
        logger.info(f"Done parsing {joblib_file_name}")

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

        
    """ 
        This is an additional function added to support QAT (Quantization Aware Training).
        It simply changes the underlying data type of each decision box FOR EACH TREE.
    """
    def set_thds_type(self, type = "int16"):
        for tree in self.trees:
            tree.set_box_data_type(type)

    def read_test_set(self, dataset_csv, no_header = False):
        self.dataframe = pd.read_csv(dataset_csv, sep = self.csv_separator)
        # Todo : Remove the assumption of the last column being the label
        attribute_name = list(self.dataframe.keys())[:-1]
        out_col = self.dataframe.keys()[-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        if not no_header:
            #name_matches = [ a == f for a, f in zip(attribute_name, f_names) ] # SUBSTITUTED FOR TIC TAC TOE ENDGAME
            name_matches = [ a.replace('-', '_') == f.replace('-', '_') for a, f in zip(attribute_name, f_names) ]
            assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
            self.x_test = self.dataframe.loc[:, self.dataframe.columns != out_col].values
            self.y_test = self.dataframe.loc[:, self.dataframe.columns == out_col].values
            for arg in self.args:
                arg[1] = self.x_test
        else: # TEMPORARY, TO GENERALIZE IN FUTURE
            self.x_test = self.dataframe.iloc[:, : -1].values
            self.y_test = self.dataframe.iloc[:, -1].values
            for arg in self.args:
                arg[1] = self.x_test
            # print(self.y_test)
            # exit(1)

    def read_training_set(self, dataset_csv, no_header = False):
        self.dataframe = pd.read_csv(dataset_csv, sep = self.csv_separator)
        # Todo : Remove the assumption of the last column being the label
        attribute_name = list(self.dataframe.keys())[:-1]
        out_col = self.dataframe.keys()[-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        if not no_header:
            #name_matches = [ a == f for a, f in zip(attribute_name, f_names) ] # SUBSTITUTED FOR TIC TAC TOE ENDGAME
            name_matches = [ a.replace('-', '_') == f.replace('-', '_') for a, f in zip(attribute_name, f_names) ]
            assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
            self.x_train = self.dataframe.loc[:, self.dataframe.columns != out_col].values
            self.y_train = self.dataframe.loc[:, self.dataframe.columns == out_col].values
            for arg in self.args:
                arg[1] = self.x_test
        else: # TEMPORARY, TO GENERALIZE IN FUTURE
            self.x_train = self.dataframe.iloc[:, : -1].values
            self.y_train = self.dataframe.iloc[:, -1].values
            for arg in self.args:
                arg[1] = self.x_test
            # print(self.y_test)
            # exit(1)

    def brace4ALS(self, als_conf):
        if self.als_conf is None:
            self.als_conf = als_conf
            for t in self.trees:
                t.brace4ALS(als_conf)
    
    """ This function resets the number of approximated bits for each feature. """
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
    def compute_score(trees : list[DecisionTree], x_test : ndarray, disable_tqdm = True):
        assert len(np.shape(x_test)) == 2
        return np.array( [ np.sum( [t.visit(x) for t in trees ], axis = 0) for x in tqdm(x_test, desc = "Evaluating score", disable = disable_tqdm) ] )
    
    def predict(self, x_test : ndarray, disable_tqdm = False):
        if len(np.shape(x_test)) == 1:
            np.reshape(x_test, np.shape(x_test)[0])
        args = [[t, x_test, disable_tqdm] for t in self.p_tree]
        return np.sum(self.pool.starmap(Classifier.compute_score, args), axis = 0)
    
    """ Given  a set of classes per each tree this function returns the set of leaf indexes for each tree.
        For instance:
            1 - class_list =  [[c_0, c_3], [c_2, c_4]]  c_0 and c_3 are the classes required for a specific tree 
                            while c_2 and c_4 are the classes for the second tree ( the one having index 1)
        This function returns the set of leaves in a related to a specific class.
    """
    def get_leaf_indexes_by_class_list(self, class_per_tree):
        leaves = {}
        for tree_idx, tree_classes in enumerate(class_per_tree):
            tree_leaves = self.trees[tree_idx].get_leaves_for_classes(tree_classes)
            leaves.update({tree_idx: tree_leaves})
        return leaves
    
    """ Identical to the previous function, it considers only the classes and leaves which are not present in the
        set of classes per tree. In other words this method returns for each tree the leaves idxs of classes not 
        present in tree's classes list specified in class_per_tree.
    """
    def get_leaf_indexes_not_in_class_list(self, class_per_tree):
        leaves = {}
        for tree_idx, tree_classes in enumerate(class_per_tree):
            tree_leaves = self.trees[tree_idx].get_leaves_idx_not_in_class(tree_classes)
            leaves.update({tree_idx: tree_leaves})
        return leaves

    """ These functions works with leaf indexes instead of boolean functions.
    """
    """ Given a set of leaves obtained using the  compute_leaves_idx this function returns the number of votes for each class. """
    def get_votes_vectors_by_leaves_idx(self, leaves, y):
        votes_vector = [ [0 for t in self.model_classes] for x in y]
        for tree_id, tree_leaves in enumerate(leaves):
            for sample_id, single_tree_leaf in enumerate(tree_leaves):
                # If the prediction truly happened
                if single_tree_leaf >= 0:
                    votes_vector[sample_id][int(self.trees[tree_id].leaves[single_tree_leaf]["class"])] += 1
        return votes_vector
    
    def get_leaves_costs_by_leaves_idx(self, leaves):
        leavesCosts = [[0 for t in range(len(self.trees))] for s in range(len(leaves[0]))]
        for treeId, treeLeaves in enumerate(leaves):
            for sampleId, leafId in enumerate(treeLeaves):
                leafMinterm = self.trees[treeId].leaves[leafId]["sop"]
                leafCost =  len(re.findall(r'Node_\d+', leafMinterm))
                leavesCosts[sampleId][treeId] = leafCost
        return leavesCosts
    
    """ Function used to obtain the set of class labels from a set of leaves indexes.
    """
    def get_class_labels_by_leaves_idx(self, leaves, y):
        votes_vectors = self.get_votes_vectors_by_leaves_idx(leaves, y)
        class_labels_draw = []
        class_labels = []
        for vote in votes_vectors:
            label = np.argmax(vote)
            if not Classifier.check_draw(vote)[0]:
                class_labels_draw.append(label)
                class_labels.append(label)
            else:
                class_labels_draw.append(label)
                class_labels.append(-1)
        #class_labels = [np.argmax(vote) if not Classifier.check_draw(vote)[0] else -1 for vote in votes_vectors]
        return class_labels, class_labels_draw
    
    def get_accuracy_from_labels(self, class_labels, class_labels_draw, y):
        correctly_classified_draw = 0
        correctly_classified_no_draw = 0
        for y_pred, y_pred_draw, y_true in zip(class_labels, class_labels_draw, y):
            if y_pred_draw == int(y_true):
                # If it was not a draw then increase both counters.
                if y_pred != -1:
                    correctly_classified_draw += 1
                    correctly_classified_no_draw += 1
                # Otherwise it was a draw, so increase only the draw counter.
                else:
                    correctly_classified_no_draw += 1
        return 100 * correctly_classified_draw / len(y), 100 * correctly_classified_no_draw / len(y)
    
    """ Given a set of leaves obtained using the  compute_leaves_idx this function returns the number of votes for each class
        and the accuracy w.r.t the oracle y.
    """
    def get_accuracy_by_leaves_idx(self, leaves, y):
        votes_vector = self.get_votes_vectors_by_leaves_idx(leaves, y)
        correctly_classified_draw = 0
        correctly_classified_no_draw = 0
        for vote, correct_classess in zip(votes_vector, y):
            #print(f"{np.argmax(vote)} {correct_classess} ")        
            if np.argmax(vote) == int(correct_classess):
                if not Classifier.check_draw(vote)[0]:
                    correctly_classified_draw += 1
                    correctly_classified_no_draw += 1
                else:
                    correctly_classified_no_draw += 1
        return votes_vector, 100 * correctly_classified_draw / len(y), 100 * correctly_classified_no_draw / len(y)
    
    """ Function used to import an ensemble pruning configuration, espressed as a list of trees to remove"""
    def prune_trees(self,pruning_conf):
        new_trees = []
        pruned_trees = []
        
        for t_idx, t in enumerate(self.trees):
            if t_idx not in pruning_conf:
                new_trees.append(t)
        for t_idx in pruning_conf:
            pruned_trees.append(self.trees[t_idx])
        
        self.trees = new_trees
        self.p_tree = list_partitioning(self.trees, self.ncpus)
        self.args = [[t, None] for t in self.p_tree]
        self.pruned_trees = pruned_trees
    
    """  
        Given the set of predicted classes per each tree and the set of true values
        this function returns the 
    """
    def get_class_rep_per_tree(self, per_tree_classes, y):
        per_tree_correct = []
        #per_tree_incorrect = []
        for tree_ in per_tree_classes:
            correct = 0
            wrong = 0
            for y_pred, y_true in zip(tree_classes, y):
                if y_pred == y_true:
                    correct += 1
                # else:
                #     wrong += 1
            per_tree_correct.append(correct)
            #per_tree_incorrect.append(wrong)
        return per_tree_correct#, per_tree_incorrect
        

    """ Given a set of leaves obtained using the  compute_leaves_idx this function returns the corresponding classes.
    """
    def transform_leaves_into_classess(self, leaves):
        classes = []
        for tree_id, tree_leaves in enumerate(leaves):
            tree_classes = []
            for single_tree_leaf in tree_leaves:
                # If the prediction truly happened
                if single_tree_leaf >= 0:
                    tree_classes.append(int(self.trees[tree_id].leaves[single_tree_leaf]["class"]))
                else:
                    tree_classes.append(-1)
            classes.append(tree_classes)
        return classes
    
    """ Given the set of predicted classes per each tree, this function returns the accuracy per each tree. """
    @staticmethod
    def get_per_tree_accuracy(per_tree_correct, nro_samples):
        per_tree_accuracy = []
        for corr in per_tree_correct:
            per_tree_accuracy.append(per_tree_correct / nro_samples * 100)
        return per_tree_accuracy
    
    @staticmethod
    def compute_indexes(trees : list[DecisionTree], X : ndarray, disable_tqdm = True):
        assert len(np.shape(X)) == 2
        # logger = logging.getLogger("pyALS-RF")
        to_ret = np.array( [t.visit_by_leaf_idx(X) for t in tqdm(trees, disable = disable_tqdm) ])
        # logger.info(to_ret)
        return to_ret
        #return np.array( [t.visit_by_leaf_idx(X) for t in tqdm(trees, disable = disable_tqdm) ])
    
    """ I implemented this function to allow compatibility with other tools.
        In pratice, a pyALSRF pruning configuration highly depends on the parsing process (i.e. to establish)
        node names which corresponds to different decision boxes.
        So for this reason this function takes in input a list of assertion functions (leaves) and returns the list of directions 
        required to reach that specific leaf.
        In practice, a direction is a set of 0 and 1 which indicate (if 0) that the right node is taken from the current node, and 
        if 1 that the left node is taken from the current node.
        Additionally, this function returns also the operators of each node in order to flip the direction in case other implementations
        do not support the operator or uses a different tree-logic. 
    """
    def transform_assertion_into_directions(self, pruning_conf):
        directions_per_tree = [[] for t in self.trees]
        for _, tree_id, assertion in pruning_conf:
            tree_idx = int(tree_id)
            dirs, ops = self.__assertion_into_dir(tree_idx, assertion)
            directions_per_tree[tree_idx].append((dirs, ops))
        return directions_per_tree
        
    def __assertion_into_dir(self, tree_idx, assertion):
        nodes_list = extract_nodes_from_assertion(assertion)
        dirs    = []
        ops     = []
        for node in nodes_list:
            if "not" in node:
                dirs.append(0)
            else:
                dirs.append(1)
            box_name = clean_node_names_from_not([node])[0]
            db : DecisionBox = self.trees[tree_idx].get_db_from_name(box_name)
            assert db != None, f"Fatal error, searching non existent box ! Name {box_name}"
            ops.append(db.get_str_op())
        return dirs, ops
            
    """ Given a set of input vectors X get the corresponding leaf index for each X. Returns a vector len(tree) X len(X) """
    def compute_leaves_idx(self, X, disable_tqdm = True):
        args = [[t, X, disable_tqdm] for t in self.p_tree]
        # logger = logging.getLogger("pyALS-RF")
        # logger.info(f"Executing {len(args)}")
        #exit(1)
        lists  = self.pool.starmap(Classifier.compute_indexes, args)
        # logger.info(returns)
        # exit(1)
        #lists = np.array(returns)
        final_list = []
        for p_t_leaves in lists:
            for tree_leaf in p_t_leaves:
                final_list.append(tree_leaf)
        return np.array(final_list)
    

    def evaluate_test_dataset(self):
        outcomes = np.sum(self.pool.starmap(Classifier.compute_score, self.args), axis = 0)
        return np.sum(tuple( np.argmax(o) == y[0] and not Classifier.check_draw(o)[0] for o, y in zip(outcomes, self.y_test))) / len(self.y_test) * 100
    
    def evaluate_accuracy(self, X, y, disable_tqdm = False):
        outcomes = self.predict(X, disable_tqdm = disable_tqdm)
        return np.sum(tuple( np.argmax(o) == y[0] and not Classifier.check_draw(o)[0] for o, y in zip(outcomes, y))) / len(y) * 100
    
            
    def predict_dump(self, index: int, outfile: str):
        score = self.predict(self.x_test[index])
        draw, max_score = Classifier.check_draw(score)
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
    

    def __adeguate_feature_names(features):
        to_ret = []
        for f in features:
            to_ret.append(f.replace('-', '_'))
        print(to_ret)
        return to_ret
    
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
                    boolean_expression += " and "
                child_l = Node(f"Node_{children_left[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}not {current_node.name}")
                stack.append((children_left[current_node_id], depth + 1, child_l))
                child_r = Node(f"Node_{children_right[current_node_id]}", parent = current_node, feature = "", operator = "", threshold_value = "", boolean_expression = f"{boolean_expression}{current_node.name}")
                stack.append((children_right[current_node_id], depth + 1, child_r))
            else:
                current_node.score = self.model_classes[np.argmax(values[current_node_id])]
                is_leaves[current_node_id] = True
        return root_node

    # Replace Tree decision boxes with faulted boxes.    
    # Returns the set of stored decision boxes per each different tree.        
    def inject_tree_boxes_faults_fb(self, faults_per_tree):
        stored_boxes = {}
        for tree_name in faults_per_tree.keys():
            for tree in self.trees:
                if tree.name == tree_name:
                    old_boxes = tree.replace_db_with_fb(faults_per_tree[tree_name])
                    stored_boxes.update({tree_name : old_boxes})
        return stored_boxes
    
    # Fix the assertion functions
    # This function simply changes specific assertion functions for each 
    # tree by setting specific assertions to True or false. 
    def inject_bns_faults_ws(self, faults_per_tree):
        old_bns_per_tree = {}
        for tree_name in faults_per_tree.keys():
            for tree in self.trees:
                if tree.name == tree_name:
                    old_bns = tree.inj_fault_assertion_functions(faults_per_tree[tree_name])
                old_bns_per_tree.update(old_bns)
        return old_bns_per_tree
    
    # Identical, without bns savings
    def inject_bns_faults(self, faults_per_tree):
        for tree_name in faults_per_tree.keys():
            for tree in self.trees:
                if tree.name == tree_name:
                    tree.inj_fault_assertion_functions_ws(faults_per_tree[tree_name])
    
    # Restore functions..
    def restore_dbs(self, old_dbs):
        for tree in self.trees:
            if tree.name in old_dbs.keys():
                tree.restore_db(old_dbs[tree.name])

    def restore_bns(self, old_bns):
        for tree in self.trees:
            tree.boolean_networks = copy.deepcopy(old_bns[tree.name])

    def store_bns(self):
        bns = {}
        for tree in self.trees:
            bns.update({ tree.name : copy.deepcopy(tree.boolean_networks)})
        return bns
    
    # VISIT TESTS
    """ 
        Returns a list, where each elements contains the set of pointers to 
        boolean functions for each different tree.
    """
    def get_bns_functions(self):
        # Save the boolean functions
        bns_fns_tree = []
        for tree in self.trees:
            new_bns = tree.get_bns_functions()
            bns_fns = []
            for bn_id, bn in enumerate(new_bns):
                bns_fns.append(generate_boolean_function(bn, f"bn_{tree.name}_{bn_id}"))
            bns_fns_tree.append(bns_fns)
        return bns_fns_tree
    
    """ 
        This function is used when inference is performed by ""linearizing"" the features
        vector with the decision boxes.
        Therefore, this function, returns the offsets for each tree in the feature linearized
        vector, in order to be able to isolate samples during BNs evaluations.
    """
    def get_dbs_offset_in_samples(self):
        # Save tree starts and endings
        self.list_starts = []
        self.list_ends   = []
        start_boxes = 0
        for tree_id, tree in enumerate(self.trees):
            boxes_tree  = len(tree.decision_boxes)
            end_boxes   = start_boxes + boxes_tree
            self.list_starts.append(start_boxes)
            self.list_ends.append(end_boxes)
            tree.set_end_start_sample(start_boxes, end_boxes)
            start_boxes += boxes_tree

        return self.list_starts, self.list_ends
    
    """ 
        For each tree, this function instantiates an array of thresholds for each comparator.
        Used to perform the linear comparison with the DBs.
    """
    def instantiate_dbs_vectors(self):
        self.dbs_thd_lin = np.array([np.float64(box["box"].threshold) for tree in self.trees for box in tree.decision_boxes])
    
    """ 
        This function generates for each x sample a linear vector to be directly compared with self.dbs_thd_lin.
        IT IS STRONGLY ADVIDE TO EXECUTE THIS FUNCTION JUST ONCE AND NOT DURING INFERENCES.
        Execution time is with this approach EXTREMELY reduced.
    """
    def linearize_samples(self, x_test: np.ndarray):
        samples_refactorized = [np.array([s[tree.attrbutes_name.index(box["box"].feature_name)]  for tree in self.trees for box in tree.decision_boxes]) for s in x_test]
        return samples_refactorized
    
    @staticmethod
    def linearize_samples_static(trees : DecisionTreeClassifier, x_test: np.ndarray):
        samples_refactorized = [np.array([s[tree.attrbutes_name.index(box["box"].feature_name)]  for tree in trees for box in tree.decision_boxes]) for s in x_test]
        return samples_refactorized
    
    """ 
        Perform on single core a linear visit on the test samples.
        This function is better than the test_sample variants when the size of vectors is low.
        Future investigation is required to test with multiprocessing alternatives.
    """
    def visit_acc_iv(self, samples, bns_fns_tree):
        box_outs = [s > self.dbs_thd_lin for s in samples]
        # Initialize to 0 for each class
        sample_preds = [np.array([0 for i in range(len(self.trees[0].boolean_networks))]) for s in samples]
        # For each output
        for idx, sample in enumerate(box_outs):
            #start_boxes = 0
            preds_per_tree = []
            # For each tree
            for tree_id, tree in enumerate(self.trees):
                tree_boxes = sample[self.list_starts[tree_id]: self.list_ends[tree_id]]
                preds_per_tree.append(np.array([bn(tree_boxes) for bn in bns_fns_tree[tree_id]]))
            #sample_preds.append(np.sum(np.array(preds_per_tree), axis = 0))
            sample_preds[idx] = np.sum(np.array(preds_per_tree), axis = 0)
        return sample_preds

    """ 
        This function splits the test samples into different cores, executes inferences like in visit_acc_iv
        and then uses multithreading to evaluate bns.
        The reason of multithreading stands behind the fact that bns are dynamically compiled ( no more eval),
        for the joy of Filippo.
        However, with simpler models, this function performs slightly worse than visit_acc_iv.
        While further investigation is required, it is recomended to use this function with larger datasets,
        with an high number of inferences ( like doing LCOR or PS) as the function easily outperforms
        its single core variant. 
    """
    def visit_acc_multhd_samples(self, samples, bns_fns_tree):
        box_outs = [s > self.dbs_thd_lin for s in samples]
        box_ids  = np.arange(len(box_outs))
        partitioned_boxes = np.array_split(box_ids, self.ncpus)
        mthd_preds = [np.array([0 for i in range(len(self.trees[0].boolean_networks))]) for s in samples]
        parallel_args = [[bns_fns_tree, self.list_starts, self.list_ends, box_outs, pb, mthd_preds] for pb in partitioned_boxes]
        self.thd_pool.map(evaluate_bns_mthd_per_sample, parallel_args)    
        return mthd_preds
    
#    return np.array( [ np.sum( [t.visit(x) for t in trees ], axis = 0) for x in tqdm(x_test, desc = "Evaluating score", disable = disable_tqdm) ] )
    @staticmethod
    def visit_per_tree(trees, linearized_box_outs):
        #minterms 
        # for box_idx, box in enumerate(tree.decision_boxes): minterms={box["box"].name : sample[tree.start: tree.end]}]
        return [np.sum([[ int(eval(a["sop"], {box["box"].name : sample[tree.start: tree.end][box_idx] for box_idx, box in enumerate(tree.decision_boxes)})) for a in tree.boolean_networks ] for tree in trees ],axis = 0) for sample in linearized_box_outs]
        

    def get_dbs_vectors(self):
        for tree in self.trees:
            tree.regenerate_asserions()
            exit(1)
        bns_tree = self.get_bns_functions()
        partitioning_indexes = list_partitioning([i for i in range(0, len(self.trees))], self.ncpus)
        list_starts, list_ends = self.get_dbs_offset_in_samples()
        # multicore_bns = [[bns_tree[idx] for idx in indexes] for indexes in partitioning_indexes]
        # multicore_starts = [[list_starts[idx] for idx in indexex ]for indexex in partitioning_indexes]
        # multicore_ends   = [[list_ends[idx] for idx in indexex] for indexex in partitioning_indexes]  
        self.instantiate_dbs_vectors()
        Node_ = self.dbs_thd_lin
        samples = self.x_test[0 : -1]
        samples_refactorized = self.linearize_samples(samples)

        start_time = time.time()
        box_outs = [s > Node_ for s in samples_refactorized]
        sample_preds = self.visit_acc_iv(samples = samples_refactorized, bns_fns_tree = bns_tree)
        end_time = (time.time() - start_time) * 1000
        print(f"Parallel time {end_time:.2f} ms")

        print(f"Executing parallel with Multithread")        
        start_time  = time.time()
        mthd_preds = self.visit_acc_multhd_samples(samples = samples_refactorized, bns_fns_tree = bns_tree)
        end_time = (time.time() - start_time) * 1000
        
        print(f"Multicore with Parallel time {end_time:.2f} ms")
        print("Executing Multicore")
        start_time = time.time()
        multicore_outs = self.predict(samples, disable_tqdm = True)
        end_time = (time.time() - start_time) * 1000
        print(f"Multicore time {end_time:.2f} ms")

        print("Executing Multicore v.2")
        start_time = time.time()
        box_outs = [s > self.dbs_thd_lin for s in samples_refactorized]
        multicore_outs_v2 = np.sum(self.pool.starmap(Classifier.visit_per_tree, [ [tree, box_outs] for tree in self.p_tree]), axis = 0)
        end_time = (time.time() - start_time) * 1000
        print(f"Multicore v.2 time {end_time:.2f} ms")

        # Validate visiting results
        for out_parallel, mouts in zip(sample_preds, multicore_outs):
            if not np.array_equal(out_parallel, mouts):
                print("Non validati")
                exit(1)
        print("Output uguali per parallel")  
        for out_parallel, mouts in zip(multicore_outs, mthd_preds):
            if not np.array_equal(out_parallel, mouts):
                print(out_parallel)
                print(mouts)
                print("Non validati")
                exit(1)
        print("Output uguali per Mthd Parallel")

    """ 
    This function executes the prediction function using IV.
    Firstly, features FOR ALL THE SAMPLES are alligned to that of DBS 
    (i.e. features are iterated and setted against the thresholds of DBS) 
    and then they are sequentially compared.
    Later, in parallel, boolean functions are evaluated for portions of the test samples
    by leveraging the evaluate_bns_mthd_per_sample.
    """
    def predict_iv(self, x_test: np.ndarray):
        return 0

# mthd_preds = []
""" 
This function assumes that the test sample is partitioned using the list partitioning
For each input sample evaluates the output of ALL the decision trees in the ensemble 
produding the output matrix for each sample
"""
def evaluate_bns_mthd_per_sample(args):
    bns = args[0]
    starts = args[1]
    ends = args[2]
    box_outs = args[3]
    box_out_indedex = args[4]
    out_preds = args[5]
    for sample_id in box_out_indedex:
        preds_per_tree = []
        for tree_idx, bns_tree in enumerate(bns):
            tree_dbs = box_outs[sample_id][starts[tree_idx]:ends[tree_idx]]
            preds_per_tree.append(np.array([bn(tree_dbs) for bn in bns_tree]))
        out_preds[sample_id] = np.sum(preds_per_tree, axis = 0)
    
# @staticmethod
# def pls_be_fast(args):
#     Node_ = args[0]
#     trees = args[1]
#     bns_tree = args[2]
#     list_starts = args[3]
#     list_ends = args[4]
#     sample_preds = args[5]
#     x_tst = args[6]
#     samples_refactorized = [np.array([s[tree.attrbutes_name.index(box["box"].feature_name)]  for tree in trees for box in tree.decision_boxes]) for s in x_tst]
#     box_outs = [s > Node_ for s in samples_refactorized]
#     # For each output
#     for sample in box_outs:
#         #start_boxes = 0
#         preds_per_tree = []
#         # For each tree
#         for tree_id, tree in enumerate(trees):
#             tree_boxes = sample[list_starts[tree_id]:list_ends[tree_id]]
#             preds_per_tree.append(np.array([bn(tree_boxes) for bn in bns_tree[tree_id]]))
#         sample_preds.append(np.sum(np.array(preds_per_tree), axis = 0))
    