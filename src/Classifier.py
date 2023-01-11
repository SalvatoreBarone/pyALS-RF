"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

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
import sys
import csv
from xml.etree import ElementTree
from anytree import Node
from jinja2 import Environment, FileSystemLoader
from distutils.file_util import copy_file
from .DecisionTree import *


class Classifier:
    __namespaces = {'pmml': 'http://www.dmg.org/PMML-4_4'}
    __source_dir = "./resources/"
    # VHDL sources
    __vhdl_bnf_source = "vhd/bnf.vhd"
    __vhdl_reg_source = "vhd/pipe_reg.vhd"
    __vhdl_decision_box_source = "vhd/decision_box.vhd"
    __vhdl_voter_source = "vhd/voter.vhd"
    __vhdl_debugfunc_source = "vhd/debug_func.vhd"
    __vhdl_classifier_template_file = "vhd/classifier.vhd.template"
    __vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
    # sh files
    __run_synth_file = "sh/run_synth.sh"
    __run_sim_file = "sh/run_sim.sh"
    __run_all_file = "sh/run_all.sh"
    __extract_luts_file = "sh/extract_utilization.sh"
    __extract_pwr_file = "sh/extract_power.sh"
    # tcl files
    __tcl_project_file = "tcl/create_project.tcl.template"
    __tcl_sim_file = "tcl/run_sim.tcl"
    # constraints
    __constraint_file = "constraints.xdc"

    def __init__(self, als_conf):
        self.__trees_list_obj = []
        self.__model_features_list_dict = []
        self.__model_classes_list_str = []
        self.__als_conf = als_conf

    def __deepcopy__(self, memo=None):
        classifier = Classifier(self.__als_conf)
        classifier.__trees_list_obj = copy.deepcopy(self.__trees_list_obj)
        classifier.__model_features_list_dict = copy.deepcopy(
            self.__model_features_list_dict)
        classifier.__model_classes_list_str = copy.deepcopy(
            self.__model_classes_list_str)
        return classifier

    def parse(self, pmml_file_name):
        self.__trees_list_obj = []
        self.__model_features_list_dict = []
        self.__model_classes_list_str = []
        tree = ElementTree.parse(pmml_file_name)
        root = tree.getroot()
        self.__namespaces["pmml"] = get_xmlns_uri(root)
        self.__get_features_and_classes(root)
        segmentation = root.find(
            "pmml:MiningModel/pmml:Segmentation", self.__namespaces)
        if segmentation is not None:
            for tree_id, segment in enumerate(segmentation.findall("pmml:Segment", self.__namespaces)):
                print(f"Parsing tree {tree_id}... ")
                tree_model_root = segment.find("pmml:TreeModel", self.__namespaces).find(
                    "pmml:Node", self.__namespaces)
                tree = self.__get_tree_model(str(tree_id), tree_model_root)
                self.__trees_list_obj.append(tree)
                print("\rDone")
        else:
            tree_model_root = root.find("pmml:TreeModel", self.__namespaces).find(
                "pmml:Node", self.__namespaces)
            tree = self.__get_tree_model("0", tree_model_root)
            self.__trees_list_obj.append(tree)

    def dump(self):
        print("Features:")
        for f in self.__model_features_list_dict:
            print("\tName: ", f["name"], ", Type: ", f["type"])
        print("\n\nClasses:")
        for c in self.__model_classes_list_str:
            print("\tName: ", c)
        print("\n\nTrees:")
        for t in self.__trees_list_obj:
            t.dump()

    def reset_nabs_configuration(self):
        self.set_nabs({f["name"]: 0 for f in self.__model_features_list_dict})

    def reset_assertion_configuration(self):
        for t in self.__trees_list_obj:
            t.reset_assertion_configuration()

    def set_nabs(self, nabs):
        for tree in self.__trees_list_obj:
            tree.set_nabs(nabs)

    def set_assertions_configuration(self, configurations):
        for t, c in zip(self.__trees_list_obj, configurations):
            t.set_assertions_configuration(c)

    def set_first_stage_approximate_implementations(self, configuration):
        for t, c in zip(self.__trees_list_obj, configuration):
            t.set_first_stage_approximate_implementations(c)

    def get_classes(self):
        return self.__model_classes_list_str

    def get_features(self):
        return self.__model_features_list_dict

    def get_trees(self):
        return self.__trees_list_obj

    def get_num_of_trees(self):
        return len(self.__trees_list_obj)

    def get_total_bits(self):
        return sum(t.get_total_bits() for t in self.__trees_list_obj)

    def get_total_retained(self):
        return sum(t.get_total_retained() for t in self.__trees_list_obj)

    def get_als_cells_per_tree(self):
        return [len(t.get_graph().get_cells()) for t in self.__trees_list_obj]

    def get_als_dv_upper_bound(self):
        ub = []
        for t in self.__trees_list_obj:
            ub.extend(iter(t.get_als_dv_upper_bound()))
        return ub

    def get_assertions_configuration(self):
        return [t.get_assertions_configuration() for t in self.__trees_list_obj]

    def get_assertions_distance(self):
        return [t.get_assertions_distance() for t in self.__trees_list_obj]

    def get_current_required_aig_nodes(self):
        return [t.get_current_required_aig_nodes() for t in self.__trees_list_obj]

    def get_num_of_first_stage_approximate_implementations(self):
        return [len(t.get_first_stage_approximate_implementations()) - 1 for t in self.__trees_list_obj]

    def get_struct(self):
        return [tree.get_struct() for tree in self.__trees_list_obj]

    def preload_dataset(self, csv_file):
        samples = []
        with open(csv_file, 'r') as data:
            for line in csv.DictReader(data, delimiter=';'):
                input_features = {}
                expected_result = {}
                for f in self.__model_features_list_dict:
                    try:
                        input_features[f["name"]] = float(line[f["name"]])
                    except:
                        print(self.__model_features_list_dict)
                        print(line)
                        print(f["name"], "feature not found in line")
                        exit()
                for c in self.__model_classes_list_str:
                    try:
                        expected_result[c] = int(line[c])
                    except:
                        print(self.__model_classes_list_str)
                        print(line)
                        print(c, "class not found in line")
                        exit()
                samples.append(
                    {"input": input_features, "outcome": expected_result})
        return samples

    def evaluate_preloaded_dataset(self, samples):
        return sum(1 if sample["outcome"] == self.__evaluate(sample["input"]) else 0 for sample in samples)

    def generate_hdl_exact_implementations(self, destination):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.__model_features_list_dict]
        mkpath(destination)
        ax_dest = f"{destination}/exact/"
        mkpath(ax_dest)
        copy_file(self.__source_dir + self.__extract_luts_file, ax_dest)
        copy_file(self.__source_dir + self.__extract_pwr_file, ax_dest)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        tcl_template = env.get_template(self.__tcl_project_file)
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
            classes=self.__model_classes_list_str)
        with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.__model_classes_list_str)
        with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)
        copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
        copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
        copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
        copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
        copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
        copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
        copy_file(self.__source_dir + self.__constraint_file, ax_dest)
        copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
        copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
        for tree in self.__trees_list_obj:
            tree.generate_hdl_tree(ax_dest)
            tree.generate_hdl_exact_assertions(ax_dest)

    def generate_hdl_ps_ax_implementations(self, destination, configurations):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(configurations, range(len(configurations))):
            features = [{"name": f["name"], "nab": n}
                        for f, n in zip(self.__model_features_list_dict, conf)]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            for tree in self.__trees_list_obj:
                tree.generate_hdl_tree(ax_dest)
                tree.generate_hdl_exact_assertions(ax_dest)

    def generate_hdl_onestep_asl_ax_implementations(self, destination, configurations):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.__model_features_list_dict]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees_name,
            features=features,
            classes=self.__model_classes_list_str)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.__model_classes_list_str)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(configurations, range(len(configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            chunks = []
            count = 0
            for size in [len(t.get_graph().get_cells()) for t in self.__trees_list_obj]:
                chunks.append([conf[i+count] for i in range(size)])
                count += size
            for t, c in zip(self.__trees_list_obj, chunks):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(c)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_twostep_asl_ax_implementations(self, destination, outer_configurations, inner_configuration):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.__model_features_list_dict]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees_name,
            features=features,
            classes=self.__model_classes_list_str)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.__model_classes_list_str)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(outer_configurations, range(len(outer_configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            for t, i, c in zip(self.__trees_list_obj, range(len(self.__trees_list_obj)), conf):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(inner_configuration[i][c])
                t.generate_hdl_als_ax_assertions(ax_dest)

    
    def generate_hdl_onestep_full_ax_implementations(self, destination, outer_configurations):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        tcl_template = env.get_template(self.__tcl_project_file)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(outer_configurations, range(len(outer_configurations))):
            features = [{"name": f["name"], "nab": n} for f, n in zip(
                self.__model_features_list_dict, conf[:len(self.__model_features_list_dict)])]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            chunks = []
            count = 0
            for size in [len(t.get_graph().get_cells()) for t in self.__trees_list_obj]:
                chunks.append(
                    [conf[i+count+len(self.__model_features_list_dict)] for i in range(size)])
                count += size
            for t, c in zip(self.__trees_list_obj, chunks):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(c)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_twostep_full_ax_implementations(self, destination, outer_configurations, inner_configuration):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        tcl_template = env.get_template(self.__tcl_project_file)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(outer_configurations, range(len(outer_configurations))):
            features = [{"name": f["name"], "nab": n} for f, n in zip(
                self.__model_features_list_dict, conf[:len(self.__model_features_list_dict)])]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.__model_classes_list_str)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            for t, n, c in zip(self.__trees_list_obj, range(len(self.__trees_list_obj)), conf[len(self.__model_features_list_dict):]):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(inner_configuration[n][c])
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_onestep_asl_wc_ax_implementations(self, destination, configurations):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.__model_features_list_dict]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees_name,
            features=features,
            classes=self.__model_classes_list_str)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.__model_classes_list_str)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for conf, i in zip(configurations, range(len(configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)
            
            assertions_conf = [conf for _ in range(len(self.__trees_list_obj))]
            self.set_assertions_configuration(assertions_conf)
            for t in self.__trees_list_obj:
                t.generate_hdl_tree(ax_dest)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_twostep_asl_wc_ax_implementations(self, destination, outer_configurations, inner_configuration):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.__model_features_list_dict]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.__source_dir + self.__run_all_file, destination)
        copy_file(self.__source_dir + self.__extract_luts_file, destination)
        copy_file(self.__source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.__trees_list_obj]
        file_loader = FileSystemLoader(self.__source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees_name,
            features=features,
            classes=self.__model_classes_list_str)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.__model_classes_list_str)
        tcl_file = tcl_template.render(
            assertions_blocks=[{
                "file_name": f"assertions_block_{n}.v",
                "language": "Verilog"
            } for n in trees_name],
            decision_trees=[{
                "file_name": f"decision_tree_{n}.vhd",
                "language": "VHDL"
            } for n in trees_name])
        for i, outer_conf in enumerate(outer_configurations):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.__source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.__source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.__source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.__source_dir + self.__constraint_file, ax_dest)
            copy_file(self.__source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.__source_dir + self.__run_sim_file, ax_dest)

            assertions_conf = [ inner_configuration[c] for c in outer_conf ]
            self.set_assertions_configuration(assertions_conf)
            for t in self.__trees_list_obj:
                t.generate_hdl_tree(ax_dest)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def __evaluate(self, features_value):
        classes_score = {c: 0 for c in self.__model_classes_list_str}
        for tree in self.__trees_list_obj:
            tree.evaluate(features_value, classes_score)
        for c in classes_score:
            classes_score[c] = 0 if classes_score[c] < (
                len(self.__trees_list_obj) / 2) else 1
        return classes_score

    def __get_features_and_classes(self, root):
        for child in root.find("pmml:DataDictionary", self.__namespaces).findall('pmml:DataField', self.__namespaces):
            if child.attrib["optype"] == "continuous":
                # the child is PROBABLY a feature
                self.__model_features_list_dict.append({
                    "name": child.attrib['name'].replace('-', '_'),
                    "type": "double" if child.attrib['dataType'] == "double" else "int"})
            elif child.attrib["optype"] == "categorical":
                # the child PROBABLY specifies model-classes
                for element in child.findall("pmml:Value", self.__namespaces):
                    self.__model_classes_list_str.append(element.attrib['value'].replace('-', '_'))

    def __get_tree_model(self, tree_name, tree_model_root, id=0):
        tree = Node(f"Node_{tree_model_root.attrib['id']}" if "id" in tree_model_root.attrib else f"Node_{id}",
                    feature="", operator="", threshold_value="", boolean_expression="")
        self.__get_tree_nodes_recursively(tree_model_root, tree, id)
        return DecisionTree(tree_name, tree, self.__model_features_list_dict, self.__model_classes_list_str, self.__als_conf)

    def __get_tree_nodes_recursively(self, element_tree_node, parent_tree_node, id=0):
        children = element_tree_node.findall("pmml:Node", self.__namespaces)
        assert len(
            children) == 2, f"Only binary trees are supported. Aborting. {children}"
        for child in children:
            boolean_expression = parent_tree_node.boolean_expression
            if boolean_expression:
                boolean_expression += " & "
            predicate = None
            if compound_predicate := child.find("pmml:CompoundPredicate", self.__namespaces):
                predicate = next(item for item in compound_predicate.findall(
                    "pmml:SimplePredicate", self.__namespaces) if item.attrib["operator"] != "isMissing")
            else:
                predicate = child.find("pmml:SimplePredicate", self.__namespaces)
            if predicate is not None:
                feature = predicate.attrib['field'].replace('-', '_')
                operator = predicate.attrib['operator']
                threshold_value = predicate.attrib['value']
                if operator in ('equal', 'lessThan', 'greaterThan'):
                    parent_tree_node.feature = feature
                    parent_tree_node.operator = operator
                    parent_tree_node.threshold_value = threshold_value
                    boolean_expression += parent_tree_node.name
                else:
                    boolean_expression += f"~{parent_tree_node.name}"
            if child.find("pmml:Node", self.__namespaces) is None:
                Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}", parent=parent_tree_node,
                    score=child.attrib['score'].replace('-', '_'), boolean_expression=boolean_expression)
            else:
                new_tree_node = Node(f"Node_{child.attrib['id']}" if "id" in child.attrib else f"Node_{id}",
                                    parent=parent_tree_node, feature="", operator="", threshold_value="", boolean_expression=boolean_expression)
                self.__get_tree_nodes_recursively(child, new_tree_node, id + 1)


def get_xmlns_uri(elem):
  if elem.tag[0] == "{":
    uri, ignore, tag = elem.tag[1:].partition("}")
  else:
    uri = None
  return uri
