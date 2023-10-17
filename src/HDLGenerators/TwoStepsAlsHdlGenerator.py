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

import os
from distutils.dir_util import mkpath
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper
from .HDLGenerator import HDLGenerator
from ..Model.Classifier import Classifier
class TwoStepsAlsHdlGenerator(HDLGenerator):
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str):
        super().__init__(classifier, yshelper, destination)
    
    def generate_axhdl(self, **kwargs):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.model_features]
        mkpath(self.destination)
        mkpath(f"{self.destination}/ax")
        copy_file(self.source_dir + self.run_all_file, self.destination)
        copy_file(self.source_dir + self.extract_luts_file, self.destination)
        copy_file(self.source_dir + self.extract_pwr_file, self.destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.tcl_project_file)
        classifier_template = env.get_template(
            self.vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.vhdl_tb_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees_name,
            features=features,
            classes=self.model_classes)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.model_classes)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for i, conf in enumerate(kwargs['configurations']):
            ax_dest = f"{self.destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.vhdl_simple_voter_source, ax_dest)
            copy_file(self.source_dir + self.vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.constraint_file, ax_dest)
            copy_file(self.source_dir + self.run_synth_file, ax_dest)
            copy_file(self.source_dir + self.run_sim_file, ax_dest)
            for t, i, c in zip(self.trees, range(len(self.trees)), conf):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(kwargs['inner_configuration'][i][c])
                t.generate_hdl_als_ax_assertions(ax_dest)