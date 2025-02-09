"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it>

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
import logging, copy
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .GREPSK import GREPSK
from sklearn.metrics import accuracy_score
import time
import os
import pandas as pd

class ResiliencyBasedGREPSK(GREPSK):

    def __init__(self, classifier : RandomForestClassifier, pruning_set_fraction : float = 0.5, min_eta : int = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()) -> None:
        super().__init__(classifier, pruning_set_fraction, ncpus)
        self.min_eta = min_eta

    def split_pruning_validation_set(self, X, y, validation_size = 0.5):
        super().split_pruning(X,y)
        

    def trim(self, cost_criterion : GREPSK.CostCriterion):
        self.logger.info("Trimming the model...")
        self.logger.info(f"Evaluating accuracy on the testing set...")
        original_classes_test = self.classifier.predict(self.x_test) 
        self.baseline_accuracy_test = accuracy_score(self.y_test, original_classes_test) * 100.0
        self.logger.info(f"Accuracy on the testing set is {self.baseline_accuracy_test}")
        start = time.time()
        # Initialize the error resiliency of the pruning samples.
        self.evaluate_error_resiliency()
        
        
        self.logger.info("Initiating algorithm ")
        # Prune until the minimum accuracy is found.
        while len(self.red_vec) > 0 :
            # The redundancy vector is ordered such that the first sample
            # is the one with the highest redundancy.
            self.red_vec = self.red_vec[1:]
            # Get the next sample.
            considered_sample = self.x_pruning_correct[0]
            self.x_pruning_correct = self.x_pruning_correct[1:]
            self.x_pruning_correct_leaves = self.x_pruning_correct_leaves[1:]
            considered_class = self.predicted_classes[0]
            self.predicted_classes = self.predicted_classes[1:]
            
            # Get leaves
            considered_leaves = self.classifier.apply([considered_sample])[0]
            pred_epis = self.computed_episVector(considered_class, considered_leaves)
            old_epi = GREPSK.compute_sample_res_from_EpiVector(pred_epis)

            while True: 
                # Prune the best leaf
                tree_to_prune, leaf_to_prune = self.get_best_leaf(considered_leaves, cost_criterion)
                if leaf_to_prune > 0:
                    parent_node, sibling, sibling_id, old_value = self.prune_leaf(tree_to_prune, leaf_to_prune)
                else:
                    break
                # Get the best Leaves
                considered_leaves = self.classifier.apply([considered_sample])[0]
                new_epi_vec =  self.computed_episVector(considered_class, considered_leaves)
                new_epi = GREPSK.compute_sample_res_from_EpiVector(new_epi_vec)
                eta = old_epi - new_epi

                if eta <= self.min_eta:
                    # Append the newly found pruned node and its tree
                    self.pruning_configuration.append((tree_to_prune, leaf_to_prune))
                    self.removed_boxes += 2
                    self.removed_and_nodes += self.leaf_dephts[tree_to_prune][leaf_to_prune] # Depth of the original leaf -1 + 1 of the sibling
                    
                    # Update the error resiliency by adding the new node 
                    # and evaluating the new R.
                    # R should be evaluated on the entire pruning set
                    if len(self.x_pruning_correct) > 0:
                        self.update_error_resiliency(tree_to_prune, leaf_to_prune, parent_node)
                else:
                    self.restore_pruned_leaf(tree_id=tree_to_prune, parent_node=parent_node, pruned_leaf=leaf_to_prune, sibling=sibling, sibling_id=sibling_id, old_value=old_value)
                    break
        end = time.time()
        self.delta_trimming = end - start
        self.used_criterion = cost_criterion
        self.logger.info(f"Trimming completed in {self.delta_trimming} seconds.")
        self.node_savings = self.origina_node_cost - self.removed_boxes
        self.logger.info(f"Original Nodes : {self.origina_node_cost} Removed Boxes : {self.removed_boxes} Savings: {self.node_savings}")
        self.logger.info(f"Evaluating pruned accuracy on the testing set")
        pruning_classes = self.classifier.predict(self.x_test)
        self.pruned_accuracy_test = accuracy_score(self.y_test, pruning_classes) * 100.0
        self.logger.info(f"Accuracy on the testing set is {self.pruned_accuracy_test}")
        self.loss_test = self.baseline_accuracy_test - self.pruned_accuracy_test
        self.logger.info(f"Loss on the testing set is {self.loss_test}")

    def dump_report(self, report_path):
        report_file = os.path.join(report_path, "report.csv")
        # Update stats.
        sol_summary = {
                "Algo"              : "loss_based",
                "LeafStrategy"      : GREPSK.CostCriterion.crit_to_str(self.used_criterion),
                "MinEta"           : self.min_eta,

                "Baseline Acc XTest" : self.baseline_accuracy_test,
                "Pruned Acc XTest"   : self.pruned_accuracy_test,
                "Loss XTest"         : self.loss_test,

                "Original Nodes"     : self.origina_node_cost,
                "Removed Nodes"      : self.removed_boxes,
                "Original Ands"      : self.original_and_nodes,
                "Removed Ands"       : self.removed_and_nodes,
                "Comp Time [s]"         : self.delta_trimming
            }
        add_header = not os.path.exists(report_file)
        df = pd.DataFrame(sol_summary, index=[0]).to_csv(report_file, index = False, header = add_header, mode = "a")
        self.logger.info(f"Summary CSV updated! Please check {report_file}")
        