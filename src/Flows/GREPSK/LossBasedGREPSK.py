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
import logging, copy
from multiprocessing import cpu_count
from tqdm import tqdm
from ...Model.Classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .GREPSK import GREPSK

class LossBasedGREPSK(GREPSK):
    
    def __init__(self, classifier : RandomForestClassifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()) -> None:
        super().__init__(classifier, pruning_set_fraction, max_loss, min_resiliency, ncpus)
    
    # Split the dataset into training, validation, and testing sets
    def split_test_dataset(self, mode, pruning_set_fraction : float = 0.5):
        # 2/3 for pruning and 1/3 for validation.
        
        # # Define the proportions
        # train_size = 0.6  # 60% for training
        # val_size = 0.2    # 20% for validation
        # test_size = 0.2   # 20% for testing

        # # First split: train and temp (val + test)
        # X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        #     X, y, indices, test_size=(1 - train_size), random_state=42
        # )

        # # Second split: temp into validation and test
        # X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        #     X_temp, y_temp, idx_temp, test_size=(test_size / (test_size + val_size)), random_state=42
        # )

    
    def trim(self, cost_criterion : GREPSK.CostCriterion):
        pass
        # super().trim(cost_criterion)
        # logger = logging.getLogger("pyALS-RF")
        # self.pruning_configuration = []
        # for x, _ in tqdm(self.initial_redundancy, total = len(self.initial_redundancy), desc="Loss-based hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        #     actual_redundancy = self.samples_info[GREP.sample_to_str(x)]["r"]
        #     active_leaves = self.samples_info[GREP.sample_to_str(x)]["leaves"]
        #     for tree_name, class_name, leaf in tqdm(active_leaves, total = len(active_leaves), desc="Evaluating leaves...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        #         tentative = copy.deepcopy(self.pruning_configuration)
        #         leaf_id = (class_name, tree_name, leaf)
        #         if leaf_id not in tentative:
        #             tentative.append(leaf_id)
        #             logger.debug(f"Tentative configuration: {tentative}")
        #             GREP.set_pruning_conf(self.classifier, tentative)
        #             self.accuracy = self.evaluate_accuracy()
        #             loss = self.baseline_accuracy - self.accuracy
        #             logger.debug(f"Resulting loss: {self.loss}")
        #             if loss < self.max_loss:
        #                 self.loss = loss
        #                 self.pruning_configuration.append(leaf_id)
        #                 logger.debug(f"Adding {leaf_id} to the list of pruned assertions. Current loss is {self.loss}% (max. {self.max_loss}%). Cost now is {self.get_cost()}.")            
        #                 # TODO: the redundancy of affected samples shuld be update in self.initial_redundancy, and it sould be re-sorted
        # final_cost = self.get_cost()
        # logger.info(f"Pruned {len(self.pruning_configuration)} leaves")
        # logger.info(f"Accuracy loss: {self.loss}")
        # logger.info(f"Final cost: {final_cost}. Expected saving is {(1 - final_cost / self.original_cost) * 100}%")
