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

def generate_boolean_function(expr, func_name):
    func_code = f"""
def {func_name}(minterms):
    return {expr}
"""
    exec(func_code, globals()) 
    return globals()[func_name]

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
    
    def parse(self, model_source : str, dataset_description = None):
        if model_source.endswith(".pmml"):
            self.pmml_parser(model_source, dataset_description)
        elif model_source.endswith(".joblib"):
            self.joblib_parser(model_source, dataset_description)
        self.ncpus = min(self.ncpus, len(self.trees))
        self.p_tree = list_partitioning(self.trees, self.ncpus)
        self.args = [[t, None] for t in self.p_tree]
        self.pool = Pool(self.ncpus)
        self.thd_pool = ThreadPoolExecutor(max_workers = self.ncpus)

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
            
    def read_test_set(self, dataset_csv):
        self.dataframe = pd.read_csv(dataset_csv, sep = self.csv_separator)
        # Todo : Remove the assumption of the last column being the label
        attribute_name = list(self.dataframe.keys())[:-1]
        out_col = self.dataframe.keys()[-1]
        assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        f_names = [ f["name"] for f in self.model_features]
        name_matches = [ a == f for a, f in zip(attribute_name, f_names) ]
        assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
        self.x_test = self.dataframe.loc[:, self.dataframe.columns != out_col].values
        self.y_test = self.dataframe.loc[:, self.dataframe.columns == out_col].values
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
    def compute_score(trees : list[DecisionTree], x_test : ndarray, disable_tqdm = True):
        assert len(np.shape(x_test)) == 2
        return np.array( [ np.sum( [t.visit(x) for t in trees ], axis = 0) for x in tqdm(x_test, desc = "Evaluating score", disable = disable_tqdm) ] )
    
    def predict(self, x_test : ndarray, disable_tqdm = False):
        if len(np.shape(x_test)) == 1:
            np.reshape(x_test, np.shape(x_test)[0])
        args = [[t, x_test, disable_tqdm] for t in self.p_tree]
        return np.sum(self.pool.starmap(Classifier.compute_score, args), axis = 0)
    
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
    
    def get_dbs_vectors(self):
        bns_tree = self.get_bns_functions()
        partitioning_indexes = list_partitioning([i for i in range(0, len(self.trees))], self.ncpus)
        list_starts, list_ends = self.get_dbs_offset_in_samples()
        # multicore_bns = [[bns_tree[idx] for idx in indexes] for indexes in partitioning_indexes]
        # multicore_starts = [[list_starts[idx] for idx in indexex ]for indexex in partitioning_indexes]
        # multicore_ends   = [[list_ends[idx] for idx in indexex] for indexex in partitioning_indexes]  
        self.instantiate_dbs_vectors()
        Node_ = self.dbs_thd_lin
        samples = self.x_test[0 : int(len(self.x_test)/2)]
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
    