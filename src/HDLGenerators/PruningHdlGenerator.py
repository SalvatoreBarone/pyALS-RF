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
import os, numpy as np, logging
from distutils.dir_util import mkpath

from pyalslib import YosysHelper, double_to_bin
from jinja2 import Environment, FileSystemLoader
from .HDLGenerator import HDLGenerator
from .LutMapper import LutMapper
from ..Model.Classifier import Classifier
from ..Model.DecisionTree import DecisionTree
from ..AxCT.HedgeTrimming import HedgeTrimming

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
        HedgeTrimming.set_pruning_conf(self.classifier, kwargs['pruning_configuration'])
        self.generate_exact_tb(f"{dest}/tb", features, env) # once the pruning configuration is set, you can use exact generator functions!
        for tree in self.classifier.trees:
            boxes = self.get_dbs(tree)
            inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
            self.implement_assertions(tree, boxes, f"{dest}/src", kwargs['lut_tech'])
            trees_inputs[tree.name] = inputs
            
        self.generate_rejection_module(f"{dest}/src", env)
        self.generate_majority_voter(f"{dest}/src", env)
        self.generate_classifier(f"{dest}/src", features, trees_inputs, env)
        self.generate_tcl(dest, trees_name, env)
        self.generate_cmakelists(dest, trees_name, env)

