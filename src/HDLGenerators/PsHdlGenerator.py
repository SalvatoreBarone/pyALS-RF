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
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper, double_to_bin
from .HDLGenerator import HDLGenerator
from ..Model.Classifier import Classifier
from .LutMapper import LutMapper

class PsHdlGenerator(HDLGenerator):
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str, pareto_path: str = None):
        super().__init__(classifier, yshelper, destination)
        self.pareto_path = pareto_path # TODO: In future fix generate_axhdl with custom pareto generation

    def generate_axhdl(self, **kwargs):
        mkpath(f"{self.destination}/ax")
        mkpath(self.destination)
        
        copy_file(self.source_dir + self.run_all_file, self.destination)
        copy_file(self.source_dir + self.extract_luts_file, self.destination)
        copy_file(self.source_dir + self.extract_pwr_file, self.destination)
        
        trees_name = [t.name for t in self.classifier.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        
        for i, conf in enumerate(kwargs['pareto_set']):
            features = [{"name": f["name"], "nab": n} for f, n in zip(self.classifier.model_features, conf)]
            
            dest = f"{self.destination}/ax/variant_{i:05d}"
            mkpath(dest)
            mkpath(f"{dest}/src")
            mkpath(f"{dest}/tb")
            self.copyfiles(dest)
            
            nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, conf)}
            self.classifier.set_nabs(nabs)
            trees_inputs = {}
            for tree in self.classifier.trees:
                boxes = self.get_dbs(tree)
                inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
                self.implement_assertions(tree, boxes, f"{dest}/src", kwargs['lut_tech'])
                trees_inputs[tree.name] = inputs
                
            self.generate_rejection_module(f"{dest}/src", env)
            self.generate_majority_voter(f"{dest}/src", env)
            self.generate_classifier(f"{dest}/src", features, trees_inputs, env)
            self.generate_ax_tb(f"{dest}/tb/", features, env)
            self.generate_tcl(dest, trees_name, env)
            self.generate_cmakelists(dest, trees_name, env)
            
    def generate_ax_tb(self, dest, features, env, **kwargs):    
        n_vectors, test_vectors, expected_outputs = self.generate_exact_test_vectors()
        tb_classifier_template = env.get_template(self.vhdl_tb_classifier_template_file)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.classifier.model_classes,
            n_vectors = n_vectors,
            latency = min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.trees)), 2)) + min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.model_classes)), 2)) + 3,
            test_vectors = test_vectors,
            expected_outputs = expected_outputs)
        with open(f"{dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)

    def get_resource_usage_custom(self):
        logger = logging.getLogger("pyALS-RF")
        mapper = LutMapper()
        dbs = [self.get_dbs(tree) for tree in self.classifier.trees]
        nDBs = sum(len(dbs) for dbs_per_tree in dbs)
        # 32 bit representation
        nLUTs_dbs_exact   = 45 * nDBs 
        nFFs_dbs_exact    = 64 * nDBs
        # For each DBS, take the minimum value
        nLUTs_dbs = 0
        nFFs_dbs = 0
        for tree_dbs in dbs:
            for box in tree_dbs: 
                # Rescale in 32 bits and then evaluate. 
                real_nab = box['box'].nab - 32
                nbits = 32 - real_nab 
                nFFs_dbs += (nbits * 2)
                nLUTs_dbs += (nbits + 13)
        nLUTs_bns = 0
        for tree in self.classifier.trees:
            logger.debug(f"Mapping tree {tree.name}")
            for c, bn in zip(self.classifier.classes_name, tree.boolean_networks):
                if bn["minterms"]:
                    logger.debug(f"\tProcessing {bn['minterms']} for class {c}")
                    nLUTs_bns += len(mapper.map(bn["minterms"], c))
                else:
                    logger.debug(f"\tClass {c} is trivially implemented as using {bn['hdl_expression']}")
        return nLUTs_dbs, nLUTs_bns, nFFs_dbs, nLUTs_dbs_exact, nFFs_dbs_exact