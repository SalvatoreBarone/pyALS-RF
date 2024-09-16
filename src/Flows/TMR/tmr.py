"""
Copyright 2021-2024 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it>
This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging, joblib, numpy as np
from multiprocessing import cpu_count, Pool
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP
import time
import csv 
import os
import json5

# Get the accuracy of a single decision tree on class c ( x_set contains only c samples).
def tree_accuracy( tree, x_set, c ):
    correctly = 0
    for x in x_set:
        if np.argmax(tree.visit(x)) == c: 
            correctly += 1
    return 100 * correctly / len(x_set)

class TMR(GREP):
    
    def __init__(
                    self, classifier : Classifier,
                    pruning_set_fraction : float = 0.5,
                    ncpus : int = cpu_count(),
                    out_path : str = "./",
                    flow : str = "pruning",
                    experiment_iteration : int = 0
                ):
        super().__init__(classifier, pruning_set_fraction, max_loss = 5.0, min_resiliency = 0, ncpus = ncpus)
        self.out_path = os.path.join(out_path, "tmr") 
        logger = logging.getLogger("pyALS-RF")
        logger.info(f"Out Path  {self.out_path}")
        if os.path.exists(self.out_path) == False:
            logger.info("Generating output path")
            os.makedirs(self.out_path)
        self.flow = flow
        self.pool = Pool(self.ncpus)
        self.pruning_configuration = []
        self.exp_it = experiment_iteration
    
    # # Predispose the  trim operation by initializing internal values.
    # def predispose_trim(self):
    #     super().trim(GREP.CostCriterion.depth) # cost_criterion is useless.


    # Approximate the function.
    def approx(self, report = True):
        # Initialize the logger 
        logger = logging.getLogger("pyALS-RF")
        # Initialize the output paths 
        report_path = os.path.join(self.out_path, "tmr.csv")
        pruning_path =  os.path.join(self.out_path, f"pruning_configuration_{self.exp_it}.json5")
        flow_store_path = os.path.join(self.out_path, ".flow.json5")

        # Split the dataset for the evaluation
        self.split_test_dataset(self.pruning_set_fraction)
        # Get the tree indexes for classes.
        self.trees_per_class = {}
        comp_time = time.time()
        # For each class, take the tree with the minimum number of classifications
        for c in self.classifier.model_classes:
            logger.debug(f"[TMR] Isolating pruning samples for class {c}")
            # Get the pruning set samples associated to the specific class. 
            pruning_set_classes_x = [ ]
            for x,y in zip(self.x_pruning, self.y_pruning): 
                #print(f" Y on pruning is {y[0]} and c is {c} {y[0] == int(c)} {type(y[0])} {type(c)}")
                if y[0] == int(c):
                    pruning_set_classes_x.append(x)
            # Find the accuracy of each single tree for the specific class.
            #acc = tree_accuracy(self.classifier.trees[0], pruning_set_classes_x, int(c)) 
            logger.debug(f"[TMR] Finding best trees for class {c}")
            accvec_args = [(t, pruning_set_classes_x, int(c)) for t in self.classifier.trees]        
            acc_vec = self.pool.starmap(tree_accuracy, accvec_args)
            # Get the best 3 different trees
            idx_couples = list(enumerate(acc_vec))
            idx_couples_sorted = sorted(idx_couples, key=lambda x: x[1], reverse=True)
            best_trees = [idx for idx, value in idx_couples_sorted[:3]]
            self.trees_per_class.update({int(c) : best_trees})
            logger.debug(f"[TMR] Pruning remaining trees for class {c}")
            # Prune the trees not present in the configuration 
            for tree_idx in range(0, len(self.classifier.trees)):
                # If the tree is not in the 
                if not tree_idx in best_trees:
                    # Remove all the leaves associated with the class
                    for l in self.classifier.trees[tree_idx].leaves:
                        # If the leaf is associated to the class
                        if l["class"] == c:
                            # append the leaf to the pruning configuration  
                            # The pruning configuration is : Class Label - Tree Idx - SOP 
                            self.pruning_configuration.append((c, str(tree_idx), l["sop"]))
        
        def tree_in_class(tree_idx):
            # for each class
            for c in self.classifier.model_classes:
                # If the tree having tree_idx is considered for a specific class
                # Return true
                if tree_idx in self.trees_per_class[int(c)]:
                    return True
            # The tree is not in a class, return False 
            return False
        
        def leaf_in_pruning(leaf_cfg):
            # For each leaf in pruning configuration 
            for already_pruned in self.pruning_configuration:
                # If the leaf is in the pruning cfg return true
                if already_pruned[0] == leaf_cfg[0] and already_pruned[1] == leaf_cfg[1] and already_pruned[2] == leaf_cfg[2]:
                    return True
            # else false
            return False
        # Now, for each tree not present in any best tree configuration, simply prune every leaf.
        for t in range(len(self.classifier.trees)):
            # If the tree is not in a class
            if not tree_in_class(t):
                # For each leaf
                for leaf in self.classifier.trees[t].leaves:
                    leaf_prune_cfg = (l["class"], str(t), l["sop"])
                    # If the leaf is not in the pruning cfg
                    if not leaf_in_pruning(leaf_prune_cfg):
                        self.pruning_configuration.append(leaf_prune_cfg)
        # Evaluate the overall computational time.
        comp_time = time.time() - comp_time
        # Now evaluate the accuracy of the pruned tree and save results
        # Arguments for evaluating accuracy 
        self.args_evaluate_validation = [[t, self.x_validation] for t in self.classifier.p_tree]
        self.pool = self.classifier.pool
        self.baseline_accuracy = self.evaluate_accuracy()[0]
        GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
        pruned_acc, nro_draw = self.evaluate_accuracy_tmr()
        logger.info(f'Baseline acc {self.baseline_accuracy} Pruned Accuracy {pruned_acc}')
        logger.info(f'N.ro pruned leaves {len(self.pruning_configuration)}  N.ro draw {nro_draw} Computational Time {comp_time}')
        # Save the pruning configuration
        self.store_pruning_conf(pruning_path)
        # Save the flow
        with open(f"{flow_store_path}", "w") as f:
            json5.dump(self.flow, f, indent=2)
        # Save the report 
        if report :
            csv_header = [  "N.ro pruned leaves",
                            "Pruned Accuracy", 
                            "Base Accuracy",
                            "N.ro draw conditions",
                            "Computational Time",
                            "Exp. It."]
            csv_body = [ 
                        len(self.pruning_configuration),
                        pruned_acc,
                        self.baseline_accuracy,
                        nro_draw,
                        comp_time,
                        self.exp_it
                        ]
            add_csv_head = not os.path.exists(report_path)
            with open(report_path, 'a') as f:
                writer = csv.writer(f)
                if add_csv_head: 
                    writer.writerow(csv_header)
                writer.writerow(csv_body)
    
    
    # Evaluate the accuracy of a tmr approximated model. 
    def evaluate_accuracy_tmr(self):
        correctly_classified_no_draw = 0
        number_draws = 0
        for x_val, y_val in zip(self.x_validation,self.y_validation):
            correct_class = y_val[0]
            trees_for_class = self.trees_per_class[correct_class]
            # Visit all the trees 
            local_counter = 0
            # Evaluate the number of correct classification over the selected trees
            for tree in trees_for_class:
                if np.argmax(self.classifier.trees[tree].visit(x_val)) == correct_class: 
                    local_counter += 1
            # If there are at least two trees then the classification is correct.
            if local_counter >= 2:
                # Check draw conditions 
                correctly_classified_no_draw += 1
                # Local counter for draw conditions
                local_counter_draw = 0
                # For each other TMR ( i.e. for each other class)
                for wrong_class, other_trees in self.trees_per_class.items():
                    # For other TRMS
                    if wrong_class != correct_class:
                        # Check if the other tree correctly vote for their class
                        for ot in other_trees:
                            pred_vec = self.classifier.trees[ot].visit(x_val)
                            # When trees are approximated then maybe no class is obtained, so the argmax will return 0.
                            # However 0 is the index of a class, resulting in a misclassification.
                            if np.argmax(pred_vec) == wrong_class and np.max(pred_vec) > 0: 
                                local_counter_draw += 1
                                # print(f" The predicted class is {np.argmax(self.classifier.trees[ot].visit(x_val))}")
                                # print(f" THe vector {self.classifier.trees[ot].visit(x_val)}")

                        # If at least two trees vote for their class, we have a draw condition.
                        if local_counter_draw >= 2 : 
                            correctly_classified_no_draw -= 1
                            number_draws += 1
                            break                  
        return correctly_classified_no_draw / len(self.x_validation) * 100, number_draws
    

    # def trim(self, report, report_path):
    #     logger = logging.getLogger("pyALS-RF")
    #     scores_idx = 0      # Index used for the scores list
    #     pruned_leaves = 0   # Number of pruned leaves
    #     final_acc = 0
    #     nro_candidates = len(self.leaf_scores)
    #     comp_time = time.time()
    #     while self.loss <= self.max_loss and len(self.leaf_scores) > scores_idx:

    #         tentative = copy.deepcopy(self.pruning_configuration) # save the pruning conf.
    #         leaf_id = self.leaf_scores[scores_idx][0] # Save the leaf id to try.  
    #         tentative.append(leaf_id)  # append the element with the best value.
        
    #         GREP.set_pruning_conf(self.classifier, tentative) # Set the pruning configuration
    #         self.accuracy = self.evaluate_accuracy() # Evaluate the accuracy
    #         loss = self.baseline_accuracy - self.accuracy # compute the loss
    #         if loss <= self.max_loss:   # If the loss is acceptable
    #             self.loss = loss        # Update the loss 
    #             final_acc = self.accuracy  # Save the real accuracy
    #             self.pruning_configuration.append(leaf_id) # Insert the leaf into the pruning configuration
    #             pruned_leaves += 1  # Increase the number of pruned leaves
    #             logger.info(f'Actual acc {self.accuracy} Base {self.baseline_accuracy}')
    #             logger.info(f'Idx {scores_idx} Max LEN {len(self.leaf_scores)}')
    #             logger.info(f'Actual loss {self.loss}')
    #             # scores_idx = 0 # Reset the score index
    #             # scores = self.compute_leaves_score(self.corr_per_leaf) # Just need to recompute the scores and sort them.
    #             # self.leaf_scores = sorted(scores.items(), key=lambda x: x[1],reverse = True) # Sort scores
    #         else :
    #             self.tabu_leaves.append(self.leaf_scores[scores_idx][0]) # Remove the leaf
    #         scores_idx += 1
    #     comp_time = time.time() - comp_time
    #     logger.info(f'Total N.ro Leaves {len(self.corr_per_leaf)} N.ro pruned leaves {len(self.pruning_configuration)}')
    #     logger.info(f' N.ro candidates {nro_candidates}  N.ro pruned leaves {pruned_leaves}')
    #     logger.info(f' Pruned Accuracy {final_acc} Base Accuracy{self.baseline_accuracy}')
    #     logger.info(f' Max loss {self.max_loss} Pruned Loss {self.loss} ')
    #     logger.info(f' Computational time {comp_time} [s]')
        
    #     # Save the report 
    #     if report :
    #         csv_header = [  "Total number of leaves" , "N.ro pruned leaves",
    #                         "N.ro candidates Iteration"," N.ro pruned leaves Iteration",
    #                         "Scores Idx", "Scores remainings elems",
    #                         "Pruned Accuracy", "Base Accuracy",
    #                         "Max Loss", "Pruned Loss",
    #                         "Computational Time"]
    #         csv_body = [ len(self.corr_per_leaf),len(self.pruning_configuration),
    #                      nro_candidates, pruned_leaves,
    #                      scores_idx, len(self.leaf_scores),
    #                      final_acc, self.baseline_accuracy,
    #                      self.max_loss, self.loss[0],
    #                      comp_time]
    #         with open(report_path, 'w') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(csv_header)
    #             writer.writerow(csv_body)


