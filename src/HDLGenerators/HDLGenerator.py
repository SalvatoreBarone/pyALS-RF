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
from distutils.dir_util import mkpath, copy_tree
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper, double_to_bin
from pathlib import Path
from ..Model.Classifier import Classifier
from ..Model.DecisionTree import DecisionTree

class HDLGenerator:
    resource_dir = "/resources/"
    # VHDL sources
    vhdl_bnf_source = "vhd/bnf.vhd"
    vhdl_reg_source = "vhd/pipe_reg.vhd"
    vhdl_decision_box_source = "vhd/decision_box.vhd"
    vhdl_voter_source = "vhd/voter.vhd"
    vhdl_debugfunc_source = "vhd/debug_func.vhd"
    vhdl_classifier_template_file = "vhd/classifier.vhd.template"
    vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
    bnf_vhd = "bnf.vhd"
    vhdl_assertions_source_template = "vhd/assertions_block.vhd.template"
    vhdl_decision_tree_source_template = "vhd/decision_tree.vhd.template"
    # sh files
    run_synth_file = "sh/run_synth.sh"
    run_sim_file = "sh/run_sim.sh"
    run_all_file = "sh/run_all.sh"
    extract_luts_file = "sh/extract_utilization.sh"
    extract_pwr_file = "sh/extract_power.sh"
    ghdl_build = "sh/build.sh"
    # tcl files
    tcl_project_file = "tcl/create_project.tcl.template"
    tcl_sim_file = "tcl/run_sim.tcl"
    # constraints
    constraint_file = "constraints.xdc"
    # CMakeLists.txt
    cmake_files_dir = "/cmake"
    cmakelists_template_file = "CMakeLists.txt.template"
    
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str):
        self.classifier = classifier
        self.yshelper = yshelper
        self.destination = destination
        self.source_dir = f"{Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]}{self.resource_dir}"
    
    def generate_exact_implementation(self):
        dest = f"{self.destination}/exact/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        mkpath(f"{dest}/tb")
        self.copyfiles(dest)
        
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        
        self.generate_classifier(f"{dest}/src", features, trees_name, env)
        self.generate_exact_tb(f"{dest}/tb", features, env)
        self.generate_tcl(dest, trees_name, env)
        self.generate_cmakelists(dest, trees_name, env)
            
        for tree in self.classifier.trees:
            self.implement_decision_boxes(tree, f"{dest}/src")
            self.implement_assertions(tree, f"{dest}/src")

    def generate_classifier(self, dest, features, trees_name, env):
        classifier_template = env.get_template(self.vhdl_classifier_template_file)
        classifier = classifier_template.render( trees=trees_name, features=features, classes=self.classifier.model_classes)
        with open(f"{dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)

    def generate_tcl(self, dest, trees_name, env):
        tcl_template = env.get_template(self.tcl_project_file)
        tcl_file = tcl_template.render(
            assertions_blocks=[{ "file_name": f"assertions_block_{n}.vhd", "language": "VHDL" } for n in trees_name],
            decision_trees=[{ "file_name": f"decision_tree_{n}.vhd", "language": "VHDL" } for n in trees_name])
        with open(f"{dest}/create_project.tcl", "w") as out_file:
            out_file.write(tcl_file)

    def generate_exact_tb(self, dest, features, env):
        n_vectors, test_vectors, expected_outputs = self.generate_exact_test_vectors()
       
        tb_classifier_template = env.get_template(self.vhdl_tb_classifier_template_file)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.classifier.model_classes,
            n_vectors = n_vectors,
            test_vectors = test_vectors,
            expected_outputs = expected_outputs)
        with open(f"{dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)

    def generate_exact_test_vectors(self):
        test_vectors = { f["name"] : [] for f in self.classifier.model_features }
        expected_outputs = { c : [] for c in self.classifier.model_classes}
        n_vectors = 0
        for i, (x, y) in enumerate(zip(self.classifier.x_test, self.classifier.y_test)):
            for k, v in zip(self.classifier.model_features, x):
                test_vectors[k["name"]].append(double_to_bin(v))
            o = np.argmax(self.classifier.predict(x))
            output = [ 1 if i == o else 0 for i in range(len(self.classifier.model_classes)) ]
            for c, v in zip(self.classifier.model_classes, output):
                expected_outputs[c].append(v)
            n_vectors += 1
            if i == 10:
                break
        #return len(self.classifier.y_test), test_vectors, expected_outputs
        return n_vectors, test_vectors, expected_outputs

    def generate_cmakelists(self, dest, trees_name, env):
        cmakelists_template = env.get_template(self.cmakelists_template_file)
        cmakelists = cmakelists_template.render(tree_names = trees_name)
        with open(f"{dest}/CMakeLists.txt", "w") as out_file:
            out_file.write(cmakelists)

    def copyfiles(self, ax_dest : str):
        copy_file(self.source_dir + self.extract_luts_file, ax_dest)
        copy_file(self.source_dir + self.extract_pwr_file, ax_dest)
        copy_file(self.source_dir + self.vhdl_bnf_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_reg_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_decision_box_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_voter_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_debugfunc_source, f"{ax_dest}/tb")
        copy_file(self.source_dir + self.tcl_sim_file, ax_dest)
        copy_file(self.source_dir + self.constraint_file, ax_dest)
        copy_file(self.source_dir + self.run_synth_file, ax_dest)
        copy_file(self.source_dir + self.run_sim_file, ax_dest)
        copy_file(self.source_dir + self.ghdl_build, ax_dest)
        copy_tree(self.source_dir + self.cmake_files_dir, ax_dest)
  
    def generate_axhdl(self, **kwargs):    
        pass
    
    def generate_ax_test_vectors(self, **kwargs):    
        pass
    
    def generate_ax_tb(self, **kwargs):    
        pass

    def implement_decision_boxes(self, tree : DecisionTree, destination):
        file_name = f"{destination}/decision_tree_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_decision_tree_source_template)
        output = template.render(
            tree_name = tree.name,
            features  = self.classifier.model_features,
            classes = self.classifier.model_classes,
            boxes = [ b["box"].get_struct() for b in tree.decision_boxes ])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name

    def implement_assertions(self, tree : DecisionTree, destination):
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{destination}/assertions_block_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_assertions_source_template)
        output = template.render(
            tree_name = tree.name,
            boxes = [b["name"] for b in tree.decision_boxes],
            classes = self.classifier.model_classes,
            assertions = tree.assertions)
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name
    
    def brace_4_als(self, tree : DecisionTree, luts_tech : str):
        self.destination = "/tmp/pyals-rf/"
        mkpath(self.destination)
        mkpath(f"{self.destination}/vhd")
        file_name, module_name = self.implement_assertions(tree, self.destination)
        self.yshelper.load_ghdl()
        self.yshelper.reset()
        self.yshelper.ghdl_read_and_elaborate([tree.bnf_vhd, file_name], module_name)
        self.yshelper.prep_design(luts_tech)
        
    def generate_als_assertions(self, tree : DecisionTree, design_name : str = None):
        self.yshelper.load_design(tree.name if design_name is None else design_name)
        self.yshelper.to_aig(tree.current_als_configuration)
        self.yshelper.clean()
        self.yshelper.opt()
        self.yshelper.write_verilog(f"{self.destination}/assertions_block_{tree.name}")