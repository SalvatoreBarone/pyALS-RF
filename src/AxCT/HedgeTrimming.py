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
from ..Model.Classifier import *
class HedgeTrimming:
    def __init__(self, classifier : Classifier, use_training_data : bool = False) -> None:
        self.classifier = classifier
        self.use_training_data = use_training_data
        self.get_pruning_table()
        self.pruned_assertions = []
        
    def get_pruning_table(self):
        self.active_assertions = self.classifier.get_assertion_activation(self.use_training_data)
        self.redundancy_table = {}
        self.pruning_table = { c : {t.name : {} for t in self.classifier.trees } for c in self.classifier.model_classes }
        for m in self.active_assertions:
            for tree, path in m["outcomes"].items():
                if path["correct"]:
                    if path["assertion"] not in self.pruning_table[m["y"]][tree]:
                        self.pruning_table[m["y"]][tree][path["assertion"]] = []
                    sample_id = ';'.join([str(x) for x in m["x"]])
                    self.pruning_table[m["y"]][tree][path["assertion"]].append(sample_id)
                    self.redundancy_table[sample_id] = m["redundancy"]
                    
    def redundancy_histogram(self):
        hist = {}
        for r in self.redundancy_table.values():
            if r not in hist:
                hist[r] = 0
            hist[r] += 1
        for k in hist:
            hist[k] = hist[k] * 100 / len(self.redundancy_table)
        return hist
    
    def store(self, outputdir : str):
        active_assertions_json = f"{outputdir}/active_assertion.json5"
        redundancy_json = f"{outputdir}/redundancy.json5"
        pruning_json = f"{outputdir}/pruning.json5"
        pruned_assertions_json = f"{outputdir}/pruned_assertions.json5"
        with open(active_assertions_json, "w") as f:
            json5.dump(self.active_assertions, f, indent=2)
        with open(redundancy_json, "w") as f:
            json5.dump(self.redundancy_table, f, indent=2)
        with open(pruning_json, "w") as f:
            json5.dump(self.pruning_table, f, indent=2)
        with open(pruned_assertions_json, "w") as f:
            json5.dump(self.pruned_assertions, f, indent=2)
            
    def trim(self):
        pass
        