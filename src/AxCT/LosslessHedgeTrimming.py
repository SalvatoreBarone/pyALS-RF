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
import json5, logging, copy
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ..Model.Classifier import Classifier
from .HedgeTrimming import HedgeTrimming

class LosslessHedgeTrimming(HedgeTrimming):
    # TODO: soglie di ridondanza come parametro e criterio di stop dell'algoritmo
    def __init__(self, classifier: Classifier, use_training_data: bool = False, min_redundancy : int = 0, max_loss_perc : float = 1.0) -> None:
        super().__init__(classifier, use_training_data)
        self.min_redundancy = min_redundancy
        self.max_loss_perc = max_loss_perc
        
    def compute_candidates(self):
        self.candidate_assertions = []
        logger = logging.getLogger("pyALS-RF")
        for class_label, trees in self.pruning_table.items():
            for tree_name, assertions in trees.items():
                for assertion, samples in assertions.items():
                    if all(self.redundancy_table[sample] > 0 for sample in samples ):
                        cost = len(assertion.split("and")) / len(samples)
                        candidate = (class_label, tree_name, assertion, cost)
                        self.candidate_assertions.append(candidate)
                        logger.debug(f"Adding {candidate} to pruning-candidate")
        logger.debug("Performing cost-based sorting for pruning-candidates")
        self.candidate_assertions.sort(key=lambda x: x[3], reverse = True)
    
    def trim(self):
        self.compute_candidates()
        logger = logging.getLogger("pyALS-RF")
        # TODO pruned assertion è incrementale, ogni elemento della lista viene costruito a partire da quello immediatamente precedente
        self.pruned_assertions = []
        #with logging_redirect_tqdm():
            #for class_label, tree_name, assertion, cost in tqdm(self.candidate_assertions, total=len(self.candidate_assertions), desc="Lossless hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        for class_label, tree_name, assertion, cost in self.candidate_assertions:
            samples = self.pruning_table[class_label][tree_name][assertion]
            approximable = all( self.redundancy_table[sample] > self.min_redundancy for sample in samples )
            if approximable:
                candidate = (class_label, tree_name, assertion, cost) 
                
                tentative = copy.deepcopy(self.pruned_assertions)
                tentative.append((class_label, tree_name, assertion, cost))
                self.classifier.set_pruning(tentative)
                self.accuracy = self.classifier.evaluate_test_dataset(True)
                self.loss = self.baseline_accuracy - self.accuracy
                if self.loss < self.max_loss_perc:
                    logger.debug(f"Adding {candidate} to the list of pruned assertions")
                    for sample in samples:
                        self.redundancy_table[sample] -= 1
                        logger.debug(f"\tDecreasing resiliency for sample {sample}. Residual redundancy: {self.redundancy_table[sample]}")
                    self.pruned_assertions.append(candidate)
                    
    def store(self, outputdir : str):
        HedgeTrimming.store(self, outputdir)
        candidate_assertions_json = f"{outputdir}/candidate_assertions.json5"
        with open(candidate_assertions_json, "w") as f:
            json5.dump(self.candidate_assertions, f, indent=2)