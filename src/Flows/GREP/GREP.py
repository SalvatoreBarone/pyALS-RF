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
import logging, numpy as np
from multiprocessing import cpu_count
from tabulate import tabulate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ...Model.Classifier import *
from ...Model.DecisionTree import *
from ...plot import boxplot
from enum import Enum

class GREP:
    
    class CostCriterion:
        depth = 1,      # higher the depth higher the cost
        activity = 2,  # lower the frequency of activation higher the cost
        combined = 3    # both the previous, combined; thus, leaves with the same costs in terms of depth but with lower frequency of activations cost more!
        
    def __init__(self, classifier : Classifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()):
        self.classifier = classifier
        self.pruning_set_fraction = pruning_set_fraction
        self.max_loss = max_loss
        self.min_resiliency = min_resiliency
        self.ncpus = min(ncpus, len(self.classifier.trees))
        
    def store_pruning_conf(self, outfile : str):
        with open(outfile, "w") as f:
            json5.dump(self.pruning_configuration, f, indent=2)
                
    def backup_bns(self):
        self.bns_backup = { t.name : copy.deepcopy(t.boolean_networks) for t in self.classifier.trees }
        
    def restore_bns(self):
        for t in self.classifier.trees:
            t.boolean_networks = self.bns_backup[t.name]

    def split_test_dataset(self, pruning_set_fraction : float = 0.5):
        self.x_pruning, self.x_validation, self.y_pruning, self.y_validation = train_test_split(self.classifier.x_test, self.classifier.y_test, train_size = pruning_set_fraction)       
    
    def evaluate_accuracy(self):
        outcomes = np.sum(self.pool.starmap(Classifier.compute_score, self.args_evaluate_validation), axis = 0)
        return np.sum(np.argmax(o) == y and not self.classifier.check_draw(o)[0] for o, y in zip(outcomes, self.y_validation)) / len(self.y_validation) * 100
    
    def evaluate_redundancy(self):
        logger = logging.getLogger("pyALS-RF")
        self.initial_redundancy = [] # keeps the initial redundancy of each sample (it's easier to sort a list of tuples)
        self.samples_info = { GREP.sample_to_str(x) : { "r": 0, "leaves" : []} for x in self.x_pruning} # for each sample, keeps its residual redundancy and the list of prunable minterm
        self.leaves_info = {} # for each leaf, of each tree, keeps the list of sample belonging to that leaf
        self.accuracy_pruning_set = 0
        # compute the tree visit for each of the sample
        logger.info("Computing the pruning capabilities")
        tree_visiting_outcomes = self.pool.starmap(GREP.compute_redundancy, self.args_evaluate_pruning) 
        logger.info("Done")
        # then, for each sample
        for x, y, (visiting_outcomes) in tqdm(zip(self.x_pruning, self.y_pruning, zip(*tree_visiting_outcomes)), total=len(self.y_pruning), desc="Computing redundancy", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            visiting_outcomes = [i[0] for i in visiting_outcomes]
            logger.debug(f"Visiting outcomes: {visiting_outcomes}")
            # compure the score of that sample
            score = np.sum(i[3] for i in visiting_outcomes)
            logger.debug(f"Score: {score}") 
            # check if the prediction results in correct classification
            if np.argmax(score) == y and not self.classifier.check_draw(score)[0]:
                self.accuracy_pruning_set += 1
                 # in this case compute the number of prunable leaves
                r = np.sort(np.array(score, copy=True))[::-1]
                self.samples_info[GREP.sample_to_str(x)]["r"] = (r[0] - r[1] - 1) // 2
                logger.debug(f"Initial redundancy: {self.samples_info[GREP.sample_to_str(x)]['r']}")
                self.initial_redundancy.append((x, self.samples_info[GREP.sample_to_str(x)]["r"]))
                # among all leaves resulting from trees visiting, the leaves that can be pruned are those resulting in correct classification
                self.samples_info[GREP.sample_to_str(x)]["leaves"] = [ i[:3] for i in visiting_outcomes if np.argmax(i[3]) == y ]
                logger.debug(f"Candidate leaves: {self.samples_info[GREP.sample_to_str(x)]['leaves']}")
                # for each of the leaves resulting in correct classification, the list of samples resulting in that leaf is updated
                # in order to have the complete list of samples activating each leaf in the forest
                for leaf in self.samples_info[GREP.sample_to_str(x)]["leaves"]: #*note that leaf is actually a tuple (tree name, leaf)
                    if leaf not in self.leaves_info:
                        self.leaves_info[leaf] = {"cost" : 0.0, "samples" : []}
                    self.leaves_info[leaf]["samples"].append(x)
                    logger.debug(f"Adding {x} to the list of samples resulting in {leaf}")
        self.initial_redundancy.sort(key=lambda x: x[1], reverse = True)
        logger.info(f"Found {len(self.leaves_info)} candidate leaves")
        self.accuracy_pruning_set = self.accuracy_pruning_set * 100 / len(self.y_pruning)     
        logger.info(f"Accuracy (on the pruning set): {self.accuracy_pruning_set}%")
        
    def sort_leaves_by_cost(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        # compute the cost of each leaf first, based on depth and activations
        for leaf, info in self.leaves_info.items():
            literals = len(leaf[2].split("and"))
            activations = len(info["samples"])
            if cost_criterion == GREP.CostCriterion.depth:
                info["cost"] = literals
            elif cost_criterion == GREP.CostCriterion.activity:
                info["cost"] = 1 / activations
            elif cost_criterion == GREP.CostCriterion.combined:
                info["cost"] = literals / activations # leaves with the same costs in terms of literals but with less activity cost more!
            logger.debug(f"Cost of {leaf} is {literals}/{activations}={info['cost']}")
        # now, for each of the activing sample, sort the list of leaves based on their cost
        for info in self.samples_info.values():
            leaves_and_their_cost = [ (leaf, self.leaves_info[leaf]["cost"]) for leaf in info["leaves"] ]
            logger.debug(f"Sorting leaves\n{leaves_and_their_cost}")
            leaves_and_their_cost.sort(key=lambda x: x[1], reverse = True)
            info["leaves"] = [ l[0] for l in leaves_and_their_cost]
            logger.debug(f"Sorted leaves\n{info['leaves']}")
    
    def redundancy_boxplot(self, outfile):
        boxplot([ i[1] for i in self.initial_redundancy ], "", "Redundancy", outfile, figsize = (2, 4), annotate = False, integer_only= True)
            
    def get_cost(self):
        return sum( GREP.get_bns_cost(t) for t in self.classifier.trees )
    
        
    @staticmethod
    def sample_to_str(x):
        return ';'.join(str(i) for i in x.tolist())
    
    @staticmethod
    def get_cost_criterion(criterion : str):
        return { "depth" : GREP.CostCriterion.depth, "activity" : GREP.CostCriterion.activity, "combined" : GREP.CostCriterion.combined}[criterion]
    
    @staticmethod
    def tree_visit_with_leaf(tree : DecisionTree, attributes):
        boxes_output = tree.get_boxes_output(attributes)
        prediction_as_one_hot = np.array([eval(a["sop"], boxes_output) for a in tree.boolean_networks ], dtype=int)
        for class_name, assertions in tree.class_assertions.items():
            for leaf in assertions:
                if eval(leaf, boxes_output):
                    return tree.name, class_name, leaf, prediction_as_one_hot
      
    @staticmethod          
    def get_bns_cost(tree : DecisionTree):
        literal_cost = 0
        for network in tree.boolean_networks:
            for minterm in network["minterms"]:
                literal_cost += len(minterm.split(" and "))
        return literal_cost
    
    @staticmethod
    def set_pruning_conf(classifier : Classifier, pruning_conf):
        for t in classifier.trees:
            GREP.set_pruning(t, pruning_conf)
    
    @staticmethod
    def set_pruning(tree : DecisionTree, pruning_configuration, use_espresso : bool = False):
        logger = logging.getLogger("pyALS-RF")
        nl = '\n'
        #tree.boolean_networks = []
        logger.debug(f"Setting pruning configuration for {tree.name}")
        for bn, (class_name, assertions) in zip(tree.boolean_networks, tree.class_assertions.items()):
            pruned = [assertion for class_label, tree_name, assertion in pruning_configuration if tree_name == tree.name and class_label == class_name ] 
            kept_assertions = [ assertion for assertion in assertions if assertion not in pruned ]
            logger.debug(f"Pruning on tree {tree.name}, class {class_name}: {len(kept_assertions)} leaves kept out of {len(bn['minterms'])}")          
            kept_assertions, sop, hdl_expression = tree.define_boolean_expression(kept_assertions, use_espresso)
            bn['minterms'] = kept_assertions
            bn['sop'] = sop
            bn['hdl_expression'] = hdl_expression
            #tree.boolean_networks.append({"class" : class_name, "minterms" : kept_assertions, "sop" : sop, "hdl_expression" : hdl_expression})
        logger.debug(f'Tree {tree.name} pruning configuration:\n{tabulate([[bn["class"], f"{nl}".join(bn["minterms"]), bn["sop"].replace(" or ", f" or{nl}"), bn["hdl_expression"].replace(" or ", f" or{nl}")] for bn in tree.boolean_networks], headers=["class", "minterms", "SoP", "HDL"], tablefmt="grid")}')    

    @staticmethod
    def compute_redundancy(trees, dataset):
        return [[ GREP.tree_visit_with_leaf(t, x) for t in trees ] for x in dataset ]
    
    def compare(self):
        data = []
        self.restore_bns()
        exact_outcome = np.sum(self.pool.starmap(Classifier.compute_score, self.args_evaluate_validation), axis = 0)
        GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
        pruned_outcome = np.sum(self.pool.starmap(Classifier.compute_score, self.args_evaluate_validation), axis = 0)
        for eo, po, x, y in zip(exact_outcome, pruned_outcome, self.x_validation, self.y_validation):
            if np.argmax(eo) != np.argmax(po) or Classifier.check_draw(eo) != Classifier.check_draw(po):
                data.append((x, y, eo, Classifier.check_draw(eo), po, Classifier.check_draw(po)))
        if data:
            print(tabulate(data, headers = ["Sample", "Class", "O.out", "O.draw", "P.out", "P.draw"], showindex="always"))
            
    def trim(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        logger.info(f"Test set: {len(self.classifier.x_test)} samples")
        logger.info(f"Pruning set fraction: {self.pruning_set_fraction}")
        self.split_test_dataset(self.pruning_set_fraction)
        logger.info(f"Pruning set: {len(self.x_pruning)} samples")
        logger.info(f"Validation set: {len(self.x_validation)} samples")
        self.p_tree = self.classifier.p_tree
        self.args_evaluate_pruning = [[t, self.x_pruning] for t in self.p_tree]
        self.args_evaluate_validation = [[t, self.x_validation] for t in self.p_tree]
        self.pool = self.classifier.pool
        self.baseline_accuracy = self.evaluate_accuracy()
        logger.info(f"Baseline accuracy (on validation set) : {self.baseline_accuracy}%")
        self.original_cost = self.get_cost()
        logger.info(f"Original cost: {self.original_cost}")
        self.accuracy = self.baseline_accuracy
        self.loss = 0
        logger.info("Performing Boolean networks backup")
        self.backup_bns()
        self.evaluate_redundancy()
        self.sort_leaves_by_cost(cost_criterion)
        self.pruning_configuration = []