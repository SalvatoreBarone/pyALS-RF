"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it>
                    Salvatore Barone <salvatore.barone@unina.it>

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

class LossBasedGREPSK(GREPSK):
    # Adds a validation fraction to split the pruning set.
    # The validation is used to evaluate the accuracy
    def __init__(self, classifier : RandomForestClassifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()) -> None:
        super().__init__(classifier, pruning_set_fraction, ncpus)
        self.max_loss = max_loss

    def split_pruning_validation_set(self, X, y, validation_size = 0.5):
        super().split_pruning(X,y)
        self.x_pruning, self.x_validation, self.y_pruning, self.y_validation, self.idx_prun, self.idx_validation = train_test_split(self.x_pruning, self.y_pruning, self.idx_prun, test_size=0.5) 
        # self.x_validation = self.x_test 
        # self.y_validation = self.y_test
        # self.idx_validation = self.idx_test

    def trim(self, cost_criterion : GREPSK.CostCriterion):
        self.logger.info("Trimming the model...")
        self.logger.info(f"Evaluating accuracy on the validation set...")
        original_classes_val = self.classifier.predict(self.x_validation)
        self.baseline_accuracy_validation = accuracy_score(self.y_validation, original_classes_val) * 100.0
        self.logger.info(f"Accuracy on the validation set is {self.baseline_accuracy_validation}")
        self.logger.info(f"Evaluating accuracy on the testing set...")
        original_classes_test = self.classifier.predict(self.x_test) 
        self.baseline_accuracy_test = accuracy_score(self.y_test, original_classes_test) * 100.0
        self.logger.info(f"Accuracy on the testing set is {self.baseline_accuracy_validation}")
        start = time.time()
        # Initialize the error resiliency of the pruning samples.
        self.evaluate_error_resiliency()
        tollerance_counter = 0
        tollerance_thds = int(len(self.x_pruning) * 0.1)
        # Prune until the minimum accuracy is found.
        while True:
            # The redundancy vector is ordered such that the first sample
            # is the one with the highest redundancy.
            considered_sample = self.redundancy_vector[-1]
            # Sample is a tuple (sample_idx, redundancy, per_tree_leaves)
            considered_leaves = considered_sample[2]
            self.logger.debug(f"Considering sample {considered_sample}")
            tree_to_prune, leaf_to_prune = self.get_best_leaf(considered_leaves, cost_criterion)
            # Prune.
            parent_node, sibling, sibling_id = self.prune_leaf(tree_to_prune, leaf_to_prune)
            # Evaluate accuracy.
            pruning_classes = self.classifier.predict(self.x_validation)
            pruned_accuracy = accuracy_score(self.y_validation, pruning_classes) * 100.0
            accuracy_loss = self.baseline_accuracy_validation - pruned_accuracy
            if accuracy_loss <= self.max_loss:
                # Append the newly found pruned node and its tree
                self.pruning_configuration.append((tree_to_prune, leaf_to_prune))
                self.removed_boxes += 1
                # Update the error resiliency by adding the new node 
                # and evaluating the new R.
                # R should be evaluated on the entire dataset.
                self.evaluate_error_resiliency()
                self.pruned_accuracy_validation = pruned_accuracy
                self.loss_validation = accuracy_loss
                tollerance_counter = 0
            else:
                self.restore_pruned_leaf(tree_id=tree_to_prune, parent_node=parent_node, pruned_leaf=leaf_to_prune, sibling=sibling, sibling_id=sibling_id)
                tollerance_counter += 1 
                if tollerance_counter >= tollerance_thds:
                    break # Stop the algorithm
        end = time.time()
        self.delta_trimming = end - start
        self.used_criterion = cost_criterion
        self.logger.info(f"Trimming completed in {self.delta_trimming} seconds.")
        self.node_savings = self.origina_node_cost - self.removed_boxes
        self.logger.info(f"Original Nodes : {self.origina_node_cost} Removed Boxes : {self.removed_boxes} Savings: {self.node_savings}")
        self.logger.info(f"Evaluating pruned accuracy on the testing validation")
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
                
                "Baseline Acc XVal" : self.baseline_accuracy_validation,
                "Pruned Acc XVal"   : self.pruned_accuracy_validation,
                "Loss XVal"         : self.loss_validation,

                "Baseline Acc XTest" : self.baseline_accuracy_test,
                "Pruned Acc XTest"   : self.pruned_accuracy_test,
                "Loss XTest"         : self.loss_test,

                "Original Nodes"     : self.origina_node_cost,
                "Removed Nodes"      : self.removed_boxes,
                "Savings"            :self.node_savings,
                "Comp Time [s]"         : self.delta_trimming
            }
        add_header = not os.path.exists(report_file)
        df = pd.DataFrame(sol_summary, index=[0]).to_csv(report_file, index = False, header = add_header, mode = "a")
        self.logger.info(f"Summary CSV updated! Please check {report_file}")
        