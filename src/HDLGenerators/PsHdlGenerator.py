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
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper
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
        
        tcl_template = env.get_template(f"{self.source_dir}{self.tcl_project_file}")
        classifier_template = env.get_template(f"{self.source_dir}{self.vhdl_classifier_template_file}")
        tb_classifier_template = env.get_template(f"{self.source_dir}{self.vhdl_tb_classifier_template_file}")
        tcl_file = tcl_template.render(
            assertions_blocks=[{ "file_name": f"assertions_block_{n}.vhd", "language": "VHDL" } for n in trees_name], 
            decision_trees=[{ "file_name": f"decision_tree_{n}.vhd", "language": "VHDL" } for n in trees_name])
        cmakelists_template = env.get_template(self.cmakelists_template_file)
        cmakelists = cmakelists_template.render(tree_names = trees_name)
        
        for i, conf in enumerate(kwargs['configurations']):
            features = [{"name": f["name"], "nab": n} for f, n in zip(self.classifier.model_features, conf)]
            
            dest = f"{self.destination}/ax/variant_{i:05d}"
            mkpath(dest)
            mkpath(f"{dest}/src")
            mkpath(f"{dest}/tb")
            self.copyfiles(dest)
            
            with open(f"{dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
                
            nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, conf)}
            self.classifier.set_nabs(nabs)
            classifier = classifier_template.render(trees = trees_name, features=features, classes=self.classifier.model_classes)
            with open(f"{dest}/src/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
                
            n_vectors, test_vectors, expected_outputs = self.generate_test_vectors()    
            tb_classifier = tb_classifier_template.render(
                features=features,
                classes=self.classifier.model_classes,
                n_vectors = n_vectors,
                test_vectors = test_vectors,
                expected_outputs = expected_outputs)
            with open(f"{dest}/tb/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
                
            with open(f"{dest}/CMakeLists.txt", "w") as out_file:
                out_file.write(cmakelists)
          
            for tree in self.classifier.trees:
                self.implement_decision_boxes(tree, f"{dest}/src")
                self.implement_assertions(tree, f"{dest}/src")