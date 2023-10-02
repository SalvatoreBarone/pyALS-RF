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

class HDLGenerator:
    resource_dir = "../resources/"
    # VHDL sources
    vhdl_bnf_source = "vhd/bnf.vhd"
    vhdl_reg_source = "vhd/pipe_reg.vhd"
    vhdl_decision_box_source = "vhd/decision_box.vhd"
    vhdl_voter_source = "vhd/voter.vhd"
    vhdl_debugfunc_source = "vhd/debug_func.vhd"
    vhdl_classifier_template_file = "vhd/classifier.vhd.template"
    vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
    bnf_vhd = "bnf.vhd"
    vhdl_assertions_source_template = "assertions_block.vhd.template"
    vhdl_decision_tree_source_template = "decision_tree.vhd.template"
    # sh files
    run_synth_file = "sh/run_synth.sh"
    run_sim_file = "sh/run_sim.sh"
    run_all_file = "sh/run_all.sh"
    extract_luts_file = "sh/extract_utilization.sh"
    extract_pwr_file = "sh/extract_power.sh"
    # tcl files
    tcl_project_file = "tcl/create_project.tcl.template"
    tcl_sim_file = "tcl/run_sim.tcl"
    # constraints
    constraint_file = "constraints.xdc"
    
    def __init__(self, classifier, yshelper, destination):
        self.classifier = classifier
        self.yshelper = yshelper
        self.destination = destination
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.source_dir =  f"{self.dir_path}/{self.resource_dir}"
    
    def generate_exact_implementation(self):
        features = [{"name": f["name"], "nab": 0} for f in self.model_features]
        mkpath(self.self.destination)
        ax_dest = f"{self.self.destination}/exact/"
        mkpath(ax_dest)
        copy_file(self.source_dir + self.extract_luts_file, ax_dest)
        copy_file(self.source_dir + self.extract_pwr_file, ax_dest)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.vhdl_tb_classifier_template_file)
        tcl_template = env.get_template(self.tcl_project_file)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
            out_file.write(tcl_file)
        classifier = template.render(
            trees=trees_name,
            features=features,
            classes=self.classifier.model_classes)
        with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.model_classes)
        with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)
        copy_file(self.source_dir + self.vhdl_bnf_source, ax_dest)
        copy_file(self.source_dir + self.vhdl_reg_source, ax_dest)
        copy_file(self.source_dir + self.vhdl_decision_box_source, ax_dest)
        copy_file(self.source_dir + self.vhdl_voter_source, ax_dest)
        copy_file(self.source_dir + self.vhdl_debugfunc_source, ax_dest)
        copy_file(self.source_dir + self.tcl_sim_file, ax_dest)
        copy_file(self.source_dir + self.constraint_file, ax_dest)
        copy_file(self.source_dir + self.run_synth_file, ax_dest)
        copy_file(self.source_dir + self.run_sim_file, ax_dest)
        for tree in self.classifier.trees:
            tree.generate_hdl_tree(ax_dest)
            tree.generate_hdl_exact_assertions(ax_dest)
  
    def generate_axhdl(self, **kwargs):    
        pass

    def generate_tree_hdl(self, tree):
        file_name = f"{self.destination}/decision_tree_{tree.name}.vhd"
        file_loader = FileSystemLoader(tree.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(tree.vhdl_decision_tree_source_template)
        output = template.render(
            tree_name = tree.name,
            features  = tree.model_features,
            classes = tree.model_classes,
            boxes = [ b["box"].get_struct() for b in tree.decision_boxes ])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name

    def generate_tree_hdl_exact_assertions(self, tree):
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{self.destination}/assertions_block_{tree.name}.vhd"
        file_loader = FileSystemLoader(tree.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(tree.__vhdl_assertions_source_template)
        output = template.render(
            tree_name = tree.name,
            boxes = [b["name"] for b in tree.decision_boxes],
            classes = tree.model_classes,
            assertions = tree.assertions)
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name

    def generate_hdl_als_ax_assertions(self, tree, design_name = None):
        tree.yosys_helper.load_design(tree.name if design_name is None else design_name)
        tree.yosys_helper.to_aig(tree.current_als_configuration)
        tree.yosys_helper.clean()
        tree.yosys_helper.opt()
        tree.yosys_helper.write_verilog(f"{self.destination}/assertions_block_{tree.name}")
    
    def generate_design_for_als(self, tree, luts_tech):
        self.destination = "/tmp/pyals-rf/"
        mkpath(self.destination)
        mkpath(f"{self.destination}/vhd")
        file_name, module_name = self.generate_tree_hdl_exact_assertions(tree, self.destination)
        tree.yosys_helper.load_ghdl()
        tree.yosys_helper.reset()
        tree.yosys_helper.ghdl_read_and_elaborate([tree.bnf_vhd, file_name], module_name)
        tree.yosys_helper.prep_design(luts_tech)
        