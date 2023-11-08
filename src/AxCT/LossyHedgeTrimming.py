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
import json5, copy
from tqdm import tqdm
from ..Model.Classifier import Classifier
from .LosslessHedgeTrimming import LosslessHedgeTrimming

class LossyHedgeTrimming(LosslessHedgeTrimming):
    def __init__(self, classifier: Classifier, use_training_data: bool = False, max_loss_perc : float = 5.0) -> None:
        super().__init__(classifier, use_training_data)
        assert 0.0 < max_loss_perc < 100.0, "Maximum allowed accuracy loss must be in (0, 100)"
        self.max_loss_perc = max_loss_perc
        self.loss = 0.0
        
    def trim(self):
        self.compute_candidates()
        self.pruned_assertions = []
        for class_label, tree_name, assertion, cost in tqdm(self.candidate_assertions, total=len(self.candidate_assertions), desc="Lossy hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            #samples = self.pruning_table[class_label][tree_name][assertion]
            #approximable = all([ self.redundancy_table[sample] > 0 for sample in samples ])
            #if approximable:
            # TODO setta la configurazione di pruning
            tentative = copy.deepcopy(self.pruned_assertions)
            tentative.append((class_label, tree_name, assertion, cost))
            self.classifier.set_pruning(tentative)
            # TODO valuta la perdita di accuratezza
            self.loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset(True)
            # TODO prosegui se la perdita è sotto soglia (con soglia parametrica)
            if self.loss > self.max_loss_perc:
                tqdm.write(f"Adding {assertion} to the list of prunable assertions (Class: {class_label}, Tree: {tree_name}, Cost: {cost}, Acc.Loss: {loss}%)")
                self.pruned_assertions.append((class_label, tree_name, assertion, cost))
