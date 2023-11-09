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
import json5, logging
from tabulate import tabulate
from ..Model.Classifier import *
class HedgeTrimming:
    def __init__(self, classifier : Classifier, use_training_data : bool = False) -> None:
        self.classifier = classifier
        self.use_training_data = use_training_data
        self.baseline_accuracy = self.classifier.evaluate_test_dataset()
        self.get_pruning_table()
        self.pruned_assertions = []
        
    def get_pruning_table(self):
        logger = logging.getLogger("pyALS-RF")
        active_assertions = self.classifier.get_assertion_activation(self.use_training_data)
        self.redundancy_table = {}
        self.pruning_table = { c : {t.name : {} for t in self.classifier.trees } for c in self.classifier.model_classes }
        for activity in active_assertions:
            sample_id = ';'.join([str(x) for x in activity["x"]])
            self.redundancy_table[sample_id] = activity["redundancy"]
            for tree, path in activity["outcomes"].items():
                if path["correct"]:
                    if path["assertion"] not in self.pruning_table[activity["y"]][tree]:
                        self.pruning_table[activity["y"]][tree][path["assertion"]] = []
                    self.pruning_table[activity["y"]][tree][path["assertion"]].append(sample_id)
                    logger.debug(f"Adding {sample_id} (which redundancy is {self.redundancy_table[sample_id]}) to the pruning table for class {activity['y']}, tree {tree}, assertion {path['assertion']}")
                    
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
        #redundancy_json = f"{outputdir}/redundancy.json5"
        pruning_json = f"{outputdir}/pruning_table.json5"
        pruned_assertions_json = f"{outputdir}/pruned_assertions.json5"
        # with open(redundancy_json, "w") as f:
        #     json5.dump(self.redundancy_table, f, indent=2)
        with open(pruning_json, "w") as f:
            json5.dump(self.pruning_table, f, indent=2)
        with open(pruned_assertions_json, "w") as f:
            json5.dump(self.pruned_assertions, f, indent=2)
            
    def trim(self):
        pass

    def compare(self):
        original_accuracy = 0
        pruned_accuracy = 0
        data = []
        for x, y in tqdm(zip(self.classifier.x_test, self.classifier.y_test), total=len(self.classifier.y_test), desc="Computing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
            outcome_original, draw_original = self.classifier.predict(x, False)
            if not draw_original and np.argmax(outcome_original) == y:
                original_accuracy += 1
            outcome_pruned, draw_pruned = self.classifier.predict(x, True)
            if not draw_pruned and np.argmax(outcome_pruned) == y:
                pruned_accuracy += 1
            if outcome_original != outcome_pruned or draw_original != draw_pruned:
                data.append((x, y, outcome_original, draw_original, outcome_pruned, draw_pruned))
        if data:
            print(tabulate(data, headers = ["Sample", "Class", "O.out", "O.draw", "P.out", "P.draw"], showindex="always"))

        