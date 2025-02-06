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
import logging
from multiprocessing import cpu_count
from tqdm import tqdm
from ...Model.Classifier import Classifier
from .GREP import GREP

class ResiliencyBasedGREP(GREP):
    def __init__(self, classifier : Classifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count()):
        super().__init__(classifier, pruning_set_fraction, max_loss, min_resiliency, ncpus)
    
    def trim(self, cost_criterion : GREP.CostCriterion):
        super().trim(cost_criterion)
        logger = logging.getLogger("pyALS-RF")
        self.pruning_configuration = []
        for x, _ in tqdm(self.initial_redundancy, total = len(self.initial_redundancy), desc="Redundancy-based hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            actual_redundancy = self.samples_info[GREP.sample_to_str(x)]["r"]
            active_leaves = self.samples_info[GREP.sample_to_str(x)]["leaves"]
            if actual_redundancy > self.min_resiliency:
                for tree_name, class_name, leaf in tqdm(active_leaves, total = len(active_leaves), desc="Evaluating leaves...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
                    leaf_id = (class_name, tree_name, leaf)
                    samples = self.leaves_info[(tree_name, class_name, leaf)]["samples"]
                    residual_redundancy = [ self.samples_info[GREP.sample_to_str(x)]["r"] for x in samples ]
                    if leaf_id not in self.pruning_configuration and all( r > (self.min_resiliency + 1) for r in residual_redundancy ):
                        self.pruning_configuration.append(leaf_id)
                        GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
                        self.accuracy = self.evaluate_accuracy()
                        self.loss = self.baseline_accuracy - self.accuracy
                        logger.debug(f"Resulting loss: {self.loss}")
                        logger.debug(f"Adding {leaf_id} to the list of pruned assertions. Current loss is {self.loss}% (max. {self.max_loss}%)")
                        actual_redundancy -= 1
                        self.update_redundancy(samples)
                        if actual_redundancy < self.min_resiliency:
                            break
        final_cost = self.get_cost()
        logger.info(f"Pruned {len(self.pruning_configuration)} leaves")
        logger.info(f"Accuracy loss: {self.loss}")
        logger.info(f"Final cost: {final_cost}. Expected saving is {(1 - final_cost / self.original_cost) * 100}%")

    def update_redundancy(self, samples):
        logger = logging.getLogger("pyALS-RF")
        for x in samples:
            self.samples_info[GREP.sample_to_str(x)]["r"] -= 1
            #TODO: self.initial_redundancy has to be updated as well, then re-sorted
            logger.debug(f"\tDecreasing resiliency for sample {x}. Residual redundancy: {self.samples_info[GREP.sample_to_str(x)]['r']}. Cost now is {self.get_cost()}.")
        