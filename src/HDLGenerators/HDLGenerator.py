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
from distutils.dir_util import mkpath, copy_tree
from distutils.file_util import copy_file
from jinja2 import Environment, FileSystemLoader
from pyalslib import YosysHelper, double_to_bin, double_to_hex
from pathlib import Path
from .LutMapper import LutMapper
from ..Model.Classifier import Classifier
from ..Model.DecisionTree import DecisionTree
import math

class HDLGenerator:
    lut_x_db = 77
    ffs_x_db = 128
    resource_dir = "/resources/"
    # VHDL sources
    vhdl_bnf_source = "vhd/bnf.vhd"
    vhdl_luts_source = "vhd/luts.vhd"
    vhdl_reg_source = "vhd/pipe_reg.vhd"
    vhdl_decision_box_source = "vhd/decision_box.vhd"
    vhdl_swapper_block_source = "vhd/swapper_block.vhd"
    vhdl_simple_voter_source = "vhd/simple_voter.vhd"
    vhdl_sorting_network_source = "vhd/sorting_network.vhd"

    vhdl_debugfunc_source = "vhd/debug_func.vhd"
    vhdl_majority_voter_template = "vhd/majority_voter.vhd.template"
    vhdl_rejection_module_template = "vhd/rejection_module.vhd.template"
    vhdl_classifier_template_file = "vhd/classifier.vhd.template"
    vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
    bnf_vhd = "bnf.vhd"
    vhdl_assertions_source_template = "vhd/assertions_block.vhd.template" 
    vhdl_decision_tree_source_template = "vhd/decision_tree.vhd.template"
    
    vhdl_assertions_regressor_source_template = "vhd/assertions_block_regressor.vhd.template"
    vhdl_memory_block_regressor_source_template = "vhd/memory_block.vhd.template"

    vhdl_regressor_tree_source_template = "vhd/regressor_tree.vhd.template"
    vhdl_template_tree_adder = "vhd/tree_adder.vhd.template"

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
    
    def generate_exact_implementation(self, **kwargs):
        dest = f"{self.destination}/exact/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        mkpath(f"{dest}/tb")
        self.copyfiles(dest)
        
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        
        trees_inputs = {}
        for tree in self.classifier.trees:
            boxes = self.get_dbs(tree)
            inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
            self.implement_assertions(tree, boxes, f"{dest}/src", kwargs['lut_tech'])
            trees_inputs[tree.name] = inputs
        
        self.generate_classifier(f"{dest}/src", features, trees_inputs, env)
        self.generate_rejection_module(f"{dest}/src", env)
        self.generate_majority_voter(f"{dest}/src", env)
        self.generate_exact_tb(f"{dest}/tb", features, env)
        self.generate_cmakelists(dest, trees_name, env)
        self.generate_tcl(dest, trees_name, env)

    def generate_classifier(self, dest, features, trees, env):
        classifier_template = env.get_template(self.vhdl_classifier_template_file)
        classifier = classifier_template.render(
            trees=trees, 
            features=features, 
            classes=self.classifier.classes_name,
            candidates=self.classifier.classes_name,
            sorting_pipe_stages  = min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.trees)), 2)), 
            reject_pipe_stages = min(1, HDLGenerator.roundUp(np.log2(len(self.classifier.model_classes)), 2)))
        with open(f"{dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)

    def generate_majority_voter(self, dest, env):
        majority_voter_template = env.get_template(self.vhdl_majority_voter_template)
        majority_voter = majority_voter_template.render(candidates=self.classifier.classes_name)
        with open(f"{dest}/majority_voter.vhd", "w") as out_file:
            out_file.write(majority_voter)

    def generate_rejection_module(self, dest, env):
        rejection_module_template = env.get_template(self.vhdl_rejection_module_template)
        t_min = 2 if len(self.classifier.trees) < len(self.classifier.classes_name) else len(self.classifier.trees) // len(self.classifier.classes_name) + 1
        t_max = len(self.classifier.trees) // 2 + 1
        n_thresholds = t_max - t_min + 1
        thresholds = {}
        for i in range(n_thresholds):
            mask = ['0'] * (n_thresholds - i - 1) + ['1'] * (i + 1)
            thresholds[i] = ''.join(mask)

        rejection_module = rejection_module_template.render(candidates = self.classifier.classes_name, thresholds = thresholds)
        with open(f"{dest}/rejection_module.vhd", "w") as out_file:
            out_file.write(rejection_module)

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
            classes=self.classifier.classes_name,
            n_vectors = n_vectors,
            latency = min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.trees)), 2)) + min(2, HDLGenerator.roundUp(np.log2(len(self.classifier.model_classes)), 2)) + 3,
            test_vectors = test_vectors,
            expected_outputs = expected_outputs)
        with open(f"{dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)

    def generate_exact_test_vectors(self):
        test_vectors = { f["name"] : [] for f in self.classifier.model_features }
        expected_outputs = { **{ c : [] for c in self.classifier.classes_name},  **{ "draw" : []} }
        predictions = self.classifier.predict(self.classifier.x_test)
        for x, output in zip(self.classifier.x_test, predictions):
            for k, v in zip(self.classifier.model_features, x):
                test_vectors[k["name"]].append(double_to_bin(v))
            draw, max = Classifier.check_draw(output)
            if max != 0:
                output = output // max
            expected_outputs["draw"].append(int(draw))
            for c, v in zip(self.classifier.classes_name, output):
                expected_outputs[c].append(v)
        return len(self.classifier.x_test), test_vectors, expected_outputs
        
    def generate_cmakelists(self, dest, trees_name, env):
        cmakelists_template = env.get_template(self.cmakelists_template_file)
        cmakelists = cmakelists_template.render(tree_names = trees_name)
        with open(f"{dest}/CMakeLists.txt", "w") as out_file:
            out_file.write(cmakelists)

    def copyfiles(self, ax_dest : str):
        copy_file(self.source_dir + self.extract_luts_file, ax_dest)
        copy_file(self.source_dir + self.extract_pwr_file, ax_dest)
        copy_file(self.source_dir + self.vhdl_bnf_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_luts_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_reg_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_decision_box_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_swapper_block_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_simple_voter_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_sorting_network_source, f"{ax_dest}/src")
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
    
    def get_dbs(self, tree: DecisionTree):
        logger = logging.getLogger("pyALS-RF")
        used_db_names = set()
        for a in tree.boolean_networks:
            used_db_names.update(set(a['hdl_expression'].replace('not ', '').replace(' and ', ' '). replace('or', '').replace(')', '').replace('(', '').split(" ")))
        used_db = [ b for b in tree.decision_boxes if b["name"] in used_db_names ]
        if len(used_db) != len(tree.decision_boxes):
            logger.debug(f"Tree {tree.name} is using {len(used_db)} out of {len(tree.decision_boxes)} DBs due to optimization, saving {(1 - len(used_db) / len(tree.decision_boxes))*100}% of resources.")
            logger.debug(f"Hereafter the DBs: {[ b['name'] for b in used_db]}")
        return used_db
    
    def get_resource_usage(self):
        logger = logging.getLogger("pyALS-RF")
        mapper = LutMapper()
        nDBs = sum(len(self.get_dbs(tree)) for tree in self.classifier.trees)
        nLuts_dbs = self.lut_x_db * nDBs
        nFFs_dbs = self.lut_x_db * nDBs
        nLUTs_bns = 0
        
        for tree in self.classifier.trees:
            logger.debug(f"Mapping tree {tree.name}")
            for c, bn in zip(self.classifier.classes_name, tree.boolean_networks):
                if bn["minterms"]:
                    logger.debug(f"\tProcessing {bn['minterms']} for class {c}")
                    nLUTs_bns += len(mapper.map(bn["minterms"], c))
                else:
                    logger.debug(f"\tClass {c} is trivially implemented as using {bn['hdl_expression']}")
        return nLuts_dbs, nLUTs_bns, nFFs_dbs
       
    def implement_decision_boxes(self, tree : DecisionTree, boxes : list,  destination : str, regr: bool = False):
        logger = logging.getLogger("pyALS-RF")
        feature_names = set(b["box"].feature_name for b in boxes )
        features = [ f for f in self.classifier.model_features if f['name'] in feature_names ]
        if len(features) != len(tree.model_features):
            logger.info(f"Tree {tree.name} is using {len(features)} out of {len(tree.model_features)} features.")
            logger.debug(f"Hereafter, their names: {[f['name'] for f in features]}")
        file_name = f"{destination}/decision_tree_{tree.name}.vhd"
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        if regr:
            template = env.get_template(self.vhdl_regressor_tree_source_template)
            output = template.render(
                tree_name = tree.name,
                features  = features,
                classes = self.classifier.classes_name,
                boxes = [ b["box"].get_struct() for b in boxes ],
                leaves_number = (len(tree.leaves)))
            file_name = f"{destination}/regression_tree_{tree.name}.vhd"
        else:
            template = env.get_template(self.vhdl_decision_tree_source_template)
            output = template.render(
                tree_name = tree.name,
                features  = features,
                classes = self.classifier.classes_name,
                boxes = [ b["box"].get_struct() for b in boxes ])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return features
        

    def implement_assertions(self, tree : DecisionTree, boxes: list, destination : str, lut_tech : int = 6):  
        logger = logging.getLogger("pyALS-RF")      
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{destination}/assertions_block_{tree.name}.vhd"
        trivial_classes = []
        nontrivial_classes = []
        for c, bn in zip(self.classifier.classes_name, tree.boolean_networks):
            if bn["minterms"] and lut_tech != None:
                luts = LutMapper(lut_tech).map(bn["minterms"], c)
                nontrivial_classes.append({"class" : c, "luts": luts})
                logger.info(f"Tree {tree.name}, class {c} is using {len(luts)} LUTs")
            else:
                trivial_classes.append({"class" : c, "expression" : bn["hdl_expression"]})
                logger.info(f"Tree {tree.name}, class {c} is implemented as using {bn['hdl_expression']}")
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_assertions_source_template)
        box_list = [b["name"] for b in boxes]
        output = template.render(
            tree_name = tree.name,
            classes = self.classifier.classes_name,
            boxes = box_list,
            trivial_classes = trivial_classes,
            nontrivial_classes = nontrivial_classes)
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name
    
    def brace_4_als(self, tree : DecisionTree, luts_tech : str):
        self.destination = "/tmp/pyals-rf/"
        mkpath(self.destination)
        mkpath(f"{self.destination}/vhd")
        file_name, module_name = self.implement_assertions(tree, self.destination, luts_tech = int(luts_tech))
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
        
    @staticmethod
    def roundUp(numToRound : int, multiple : int):
        if multiple == 0:
            return numToRound
        remainder = numToRound % multiple
        if remainder == 0:
            return numToRound
        return numToRound + multiple - remainder

    # METHODS used for testing regressors, to delete.   
    # Used only for testing, to delete
    @staticmethod
    def get_resource_usage(classifier: Classifier):
        logger = logging.getLogger("pyALS-RF")
        mapper = LutMapper()
        nDBs = sum(len(HDLGenerator.get_dbs(tree)) for tree in classifier.trees)
        nLuts_dbs = HDLGenerator.lut_x_db * nDBs
        nFFs_dbs = HDLGenerator.ffs_x_db * nDBs
        nLUTs_bns = 0
        
        # for tree in classifier.trees:
        #     logger.debug(f"Mapping tree {tree.name}")
        #     for c, bn in zip(classifier.classes_name, tree.boolean_networks):
        #         if bn["minterms"]:
        #             logger.debug(f"\tProcessing {bn['minterms']} for class {c}")
        #             print(f"\tProcessing {bn['minterms']} for class {c}")
        #             nLUTs_bns += len(mapper.map(bn["minterms"], c))
        #         else:
        #             print(f"\tClass {c} is trivially implemented as using {bn['hdl_expression']}")
        #             logger.debug(f"\tClass {c} is trivially implemented as using {bn['hdl_expression']}")
        for tree in classifier.trees:
            for bn in tree.boolean_networks:
                # if bn["minterms"]:
                logger.debug(f"\tProcessing {bn['sop']} for class {bn['class']}")
                nLUTs_bns += len(mapper.map(bn["sop"], bn['class']))
                # else:
                #     print(f"\tClass {bn['class']} is trivially implemented as using {bn['hdl_expression']}")
                #     logger.debug(f"\tClass {bn['class']} is trivially implemented as using {bn['hdl_expression']}")
        return nLuts_dbs, nLUTs_bns, nFFs_dbs
    
    @staticmethod
    def get_dbs( tree: DecisionTree):
        logger = logging.getLogger("pyALS-RF")
        used_db_names = set()
        for a in tree.boolean_networks:
            used_db_names.update(set(a['hdl_expression'].replace('not ', '').replace(' and ', ' '). replace('or', '').replace(')', '').replace('(', '').split(" ")))
        used_db = [ b for b in tree.decision_boxes if b["name"] in used_db_names ]
        if len(used_db) != len(tree.decision_boxes):
            logger.debug(f"Tree {tree.name} is using {len(used_db)} out of {len(tree.decision_boxes)} DBs due to optimization, saving {(1 - len(used_db) / len(tree.decision_boxes))*100}% of resources.")
            logger.debug(f"Hereafter the DBs: {[ b['name'] for b in used_db]}")
        return used_db
    
    def regressor_generate_exact_implementation(self):

        dest = f"{self.destination}/exact/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        mkpath(f"{dest}/tb")
        self.copyfiles(dest)
        self.implement_tree_adder(5,4, f"{dest}/src")
        exit(1)
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        
        trees_inputs = {}
        #for tree in self.classifier.trees:
            # boxes = self.get_dbs(tree)
            # inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
            # self.implement_assertions(tree, boxes, f"{dest}/src", 6)
            # trees_inputs[tree.name] = inputs
        boxes = self.get_dbs(self.classifier.trees[0])
        inputs = self.implement_decision_boxes(self.classifier.trees[0], boxes, f"{dest}/src", True)
        self.implement_assertions_regressor(self.classifier.trees[0], boxes, f"{dest}/src", 6)
        self.regressor_implement_memory_block(self.classifier.trees[0], f"{dest}/src")
        #self.implement_assertions(self.classifier.trees[0], boxes, f"{dest}/src", 6)
        # trees_inputs[self.classifier.trees[0].name] = inputs
    
    def implement_assertions_regressor(self, tree : DecisionTree, boxes: list, destination : str, lut_tech : int = 6):  
        logger = logging.getLogger("pyALS-RF")      
        module_name = f"assertions_block_regressor_{tree.name}"
        file_name = f"{destination}/assertions_block_regressor_{tree.name}.vhd"
        trivial_classes = []
        nontrivial_classes = []
        # print(type(tree.boolean_networks))
        # for bn in tree.boolean_networks:
        #     print(bn["class"])
        #     print(type(bn["class"]))
        # exit(1)
        dict_leaves = []
        counter = len(tree.boolean_networks) - 1
        for bn in tree.boolean_networks:
            dict_leaves.append({'idx' : counter, 'expression': bn["hdl_expression"]})
            counter -= 1
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_assertions_regressor_source_template)
        box_list = [b["name"] for b in boxes]
        output = template.render(
            tree_name = tree.name,
            classes = self.classifier.classes_name,
            boxes = box_list,
            leaves_number = len(tree.leaves),
            leaves_enumerate = dict_leaves,
            trivial_classes = [],
            nontrivial_classes = [])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name
 
        # for bn in tree.boolean_networks:
        #     if bn["minterms"] and lut_tech != None:
        #         luts = LutMapper(lut_tech).map(bn["sop"], bn['class'])
        #         nontrivial_classes.append({"class" : bn['class'] , "luts": luts})
        #     else:
        #         trivial_classes.append({"class" : bn['class'], "expression" : bn["hdl_expression"]})
        #         logger.info(f"Tree {tree.name}, class {bn['class']} is implemented as using {bn['hdl_expression']}")
        # file_loader = FileSystemLoader(self.source_dir)
        # env = Environment(loader=file_loader)
        # template = env.get_template(self.vhdl_assertions_regressor_source_template)
        # box_list = [b["name"] for b in boxes]
        # output = template.render(
        #     tree_name = tree.name,
        #     classes = self.classifier.classes_name,
        #     boxes = box_list,
        #     leaves = len(tree.leaves),
        #     trivial_classes = trivial_classes,
        #     nontrivial_classes = nontrivial_classes)
        # with open(file_name, "w") as out_file:
        #     out_file.write(output)
        # return file_name, module_name
    
    def regressor_implement_memory_block(self, tree : DecisionTree, destination : str):  
        logger = logging.getLogger("pyALS-RF")      
        module_name = f"memory_block_regressor_{tree.name}"
        file_name = f"{destination}/memory_block_regressor_{tree.name}.vhd"
        def generate_leaves_one_hot(N):
            results = []
            for i in range(N):
                one_hot_string = ['0'] * N
                one_hot_string[i] = '1'
                results.append(''.join(one_hot_string))
            return results
        leaves_idx_one_hot = generate_leaves_one_hot(len(tree.boolean_networks))
        leaves_conditions = []
        for l,bn in zip(leaves_idx_one_hot, tree.boolean_networks):
            leaves_conditions.append({'leaf_active' : l, 'leaf_output': double_to_hex(bn["class"])[2:]})
                
        file_loader = FileSystemLoader(self.source_dir)    
        env = Environment(loader=file_loader)
        template = env.get_template(self.vhdl_memory_block_regressor_source_template)

        output = template.render(
            tree_name = tree.name,
            leaves_number = len(tree.leaves),
            memory_block_condition = leaves_conditions)
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name, module_name
    
    # Implement a tree structure for an adder.
    def implement_tree_adder(self, num_inputs, data_width, destination):
        n_levels = math.ceil(math.log2(num_inputs))
        level_descriptions = []
        inputs_per_level = num_inputs
        outputs_per_level = math.ceil(inputs_per_level / 2)
        #width_per_level = data_width
        use_additive = inputs_per_level % 2 != 0
        last_level_width = 0
        for level in range(0,n_levels):
            # Setup the number of adders per each level and 
            # subsequently update the variables for the next level.
            last_level_width = level + 1
            level_description = {
                "level_id"      : level,                # Level of tree adder
                "num_inputs"    : inputs_per_level,     # Number of inputs to the level
                "num_outputs"   : outputs_per_level,    # Number of outputs
                "in_data_width" : level,                # Data width offset of inputs      
                "out_data_width": last_level_width,     # Data width offset of outputs
                "use_additive"  : use_additive,         # Flag used to add a signal
                "is_last"       : level == n_levels - 1 # Check if it is the last level.
            }              
            use_additive = outputs_per_level % 2 != 0   # If not even then add a 0 pad signal.
            inputs_per_level = outputs_per_level        # The number of inputs is equal to the number of outs of the previous level
            outputs_per_level = math.ceil(inputs_per_level / 2)        
            #data_width += 1
            level_descriptions.append(level_description)
            print(level_description)

        file_loader = FileSystemLoader(self.source_dir)    
        env = Environment(loader=file_loader)
        tree_adder_template = env.get_template(self.vhdl_template_tree_adder)
        out = tree_adder_template.render(level_descriptions = level_descriptions, out_width = last_level_width - 1)
        file_name = f"{destination}/tree_adder.vhd"

        with open(file_name, "w") as out_file:
            out_file.write(out)
