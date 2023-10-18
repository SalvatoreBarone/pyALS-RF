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
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper, double_to_bin
from .HDLGenerator import HDLGenerator
from ..Model.Classifier import Classifier

class PsHdlGenerator(HDLGenerator):
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str):
        super().__init__(classifier, yshelper, destination)
        
    def generate_axhdl(self, **kwargs):
        mkpath(f"{self.destination}/ax")
        mkpath(self.destination)
        
        copy_file(self.source_dir + self.run_all_file, self.destination)
        copy_file(self.source_dir + self.extract_luts_file, self.destination)
        copy_file(self.source_dir + self.extract_pwr_file, self.destination)
        
        trees_name = [t.name for t in self.classifier.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        
        for i, conf in enumerate(kwargs['configurations']):
            features = [{"name": f["name"], "nab": n} for f, n in zip(self.classifier.model_features, conf)]
            
            dest = f"{self.destination}/ax/variant_{i:05d}"
            mkpath(dest)
            mkpath(f"{dest}/src")
            mkpath(f"{dest}/tb")
            self.copyfiles(dest)
            
            nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, conf)}
            self.classifier.set_nabs(nabs)
            
            self.generate_tcl(dest, trees_name, env)
            self.generate_cmakelists(dest, trees_name, env)
            self.generate_rejection_module(f"{dest}/src", env)
            self.generate_majority_voter(f"{dest}/src", env)
            self.generate_classifier(f"{dest}/src", features, trees_name, env)
            self.generate_ax_tb(f"{dest}/tb/", features, env)
            for tree in self.classifier.trees:
                self.implement_decision_boxes(tree, f"{dest}/src")
                self.implement_assertions(tree, f"{dest}/src")
                
    def generate_ax_tb(self, dest, features, env, **kwargs):    
        n_vectors, test_vectors, expected_outputs = self.generate_exact_test_vectors()
       
        tb_classifier_template = env.get_template(self.vhdl_tb_classifier_template_file)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.classifier.model_classes,
            n_vectors = n_vectors,
            pipe_stages = min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.trees)), 2)),
            test_vectors = test_vectors,
            expected_outputs = expected_outputs)
        with open(f"{dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)