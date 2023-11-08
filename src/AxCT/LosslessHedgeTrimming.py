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
import json5
from tqdm import tqdm
from ..Model.Classifier import Classifier
from .HedgeTrimming import HedgeTrimming

class LosslessHedgeTrimming(HedgeTrimming):
    def __init__(self, classifier: Classifier, use_training_data: bool = False) -> None:
        super().__init__(classifier, use_training_data)
        
    def compute_candidates(self):
        self.candidate_assertions = []
        for class_label, trees in self.pruning_table.items():
            for tree_name, assertions in trees.items():
                for assertion, samples in assertions.items():
                    approximable = all([ self.redundancy_table[sample] > 0 for sample in samples ])
                    literals = len(assertion.split("and"))
                    if approximable:
                        self.candidate_assertions.append((class_label, tree_name, assertion, literals / len(samples)) )
        self.candidate_assertions.sort(key=lambda x: x[3], reverse = True)
    
    def trim(self):
        self.compute_candidates()
        self.pruned_assertions = []
        for class_label, tree_name, assertion, cost in tqdm(self.candidate_assertions, total=len(self.candidate_assertions), desc="Lossless hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            samples = self.pruning_table[class_label][tree_name][assertion]
            approximable = all([ self.redundancy_table[sample] > 0 for sample in samples ])
            if approximable:
                for sample in samples:
                    self.redundancy_table[sample] -= 1
                self.pruned_assertions.append((class_label, tree_name, assertion, cost))
        self.classifier.set_pruning(self.pruned_assertions)
        self.loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset(True)
    
    def store(self, outputdir : str):
        HedgeTrimming.store(self, outputdir)
        candidate_assertions_json = f"{outputdir}/candidate_assertions.json5"
        with open(candidate_assertions_json, "w") as f:
            json5.dump(self.candidate_assertions, f, indent=2)