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
from distutils.dir_util import mkpath
from pyalslib import YosysHelper
from pyalslib import YosysHelper
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
        
        self.generate_classifier(f"{dest}/src", features, trees_name, env)
        self.generate_tb(f"{dest}/tb", features, env)
        self.generate_tcl(dest, trees_name, env)
        self.generate_cmakelists(dest, trees_name, env)
        
        self.classifier.set_pruning(kwargs['pruned_assertions'])
        for tree in self.classifier.trees:
            self.implement_decision_boxes(tree, dest)
            self.implement_pruned_assertions(tree, dest)
            
    def implement_pruned_assertions(self, tree : DecisionTree, destination : str):
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{destination}/assertions_block_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_assertions_source_template)
        output = template.render(
            tree_name = tree.name,
            boxes = [b["name"] for b in tree.decision_boxes],
            classes = self.classifier.model_classes,
            assertions = tree.pruned_assertions)
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name