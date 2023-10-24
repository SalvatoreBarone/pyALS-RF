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
import os, numpy as np
from distutils.dir_util import mkpath
from pyalslib import YosysHelper, double_to_bin
from jinja2 import Environment, FileSystemLoader
from .HDLGenerator import HDLGenerator
from ..Model.Classifier import Classifier
from ..Model.DecisionTree import DecisionTree

class PruningHdlGenerator(HDLGenerator):
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str):
        super().__init__(classifier, yshelper, destination)
        
    def generate_axhdl(self, **kwargs):
        dest = f"{self.destination}/pruned/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        mkpath(f"{dest}/tb")
        self.copyfiles(dest)
        
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        trees_inputs = {}
        self.classifier.set_pruning(kwargs['pruned_assertions'])
        self.generate_ax_tb(f"{dest}/tb", features, env)
        for tree in self.classifier.trees:
            boxes = self.get_pruned_dbs(tree)
            inputs = self.implement_pruned_decision_boxes(tree, boxes, f"{dest}/src")
            self.implement_pruned_assertions(tree, boxes, f"{dest}/src")
            trees_inputs[tree.name] = inputs
            
        self.generate_rejection_module(f"{dest}/src", env)
        self.generate_majority_voter(f"{dest}/src", env)
        self.generate_classifier(f"{dest}/src", features, trees_inputs, env)
        self.generate_tcl(dest, trees_name, env)
        self.generate_cmakelists(dest, trees_name, env)
        
            
    def get_pruned_dbs(self, tree: DecisionTree):
        used_db_names = set()
        for a in tree.pruned_assertions:
            used_db_names.update(set(a['minimized'].replace('not ', '').replace('func_and(', ''). replace('func_or(', '').replace(')', '').replace(',', '').split(" ")))
        used_db = [ b for b in tree.decision_boxes if b["name"] in used_db_names ]
        print(f"Tree {tree.name} is using {len(used_db)} out of {len(tree.decision_boxes)} DBs due to pruning, saving {(1 - len(used_db) / len(tree.decision_boxes))*100}% of resources")
        return used_db
            
    def implement_pruned_decision_boxes(self, tree : DecisionTree, boxes : list, destination):
        feature_names = set(b["box"].feature_name for b in boxes )
        features = [ f for f in self.classifier.model_features if f['name'] in feature_names ]
        print(f"Tree {tree.name} is using {len(features)} out of {len(tree.model_features)} DBs due to pruning, saving {(1 - len(features) / len(tree.model_features))*100}% of resources")
        file_name = f"{destination}/decision_tree_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_decision_tree_source_template)
        output = template.render(
            tree_name = tree.name,
            features  = features,
            classes = self.classifier.classes_name,
            boxes = [b["box"].get_struct() for b in boxes])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return features
            
    def implement_pruned_assertions(self, tree : DecisionTree, boxes : list, destination : str):
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{destination}/assertions_block_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_assertions_source_template)
        output = template.render(
            tree_name = tree.name,
            boxes = [b["name"] for b in boxes],
            assertions = [{"class" : n, "expression" : a["minimized"]} for n, a in zip(self.classifier.classes_name, tree.pruned_assertions)])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name
    
    def generate_ax_test_vectors(self, **kwargs):    
        test_vectors = { f["name"] : [] for f in self.classifier.model_features }
        expected_outputs = { **{ c : [] for c in self.classifier.classes_name},  **{ "draw" : []} }
        for x in self.classifier.x_test:
            for k, v in zip(self.classifier.model_features, x):
                test_vectors[k["name"]].append(double_to_bin(v))
            output, draw = self.classifier.predict(x, True)
            expected_outputs["draw"].append(int(draw))
            for c, v in zip(self.classifier.classes_name, output):
                expected_outputs[c].append(v)
        return len(self.classifier.y_test), test_vectors, expected_outputs
    
    def generate_ax_tb(self, dest, features, env, **kwargs):    
        n_vectors, test_vectors, expected_outputs = self.generate_ax_test_vectors()
       
        tb_classifier_template = env.get_template(self.vhdl_tb_classifier_template_file)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.classifier.classes_name,
            n_vectors = n_vectors,
            latency = min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.trees)), 2)) + min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.model_classes)), 2)) + 3,
            test_vectors = test_vectors,
            expected_outputs = expected_outputs)
        with open(f"{dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)