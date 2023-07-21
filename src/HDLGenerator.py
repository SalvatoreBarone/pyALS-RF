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

class HDLGenerator:
    __source_dir = "../resources/"
    # VHDL sources
    __vhdl_bnf_source = "vhd/bnf.vhd"
    __vhdl_reg_source = "vhd/pipe_reg.vhd"
    __vhdl_decision_box_source = "vhd/decision_box.vhd"
    __vhdl_voter_source = "vhd/voter.vhd"
    __vhdl_debugfunc_source = "vhd/debug_func.vhd"
    __vhdl_classifier_template_file = "vhd/classifier.vhd.template"
    __vhdl_tb_classifier_template_file = "vhd/tb_classifier.vhd.template"
    __bnf_vhd = "bnf.vhd"
    __vhdl_assertions_source_template = "assertions_block.vhd.template"
    __vhdl_decision_tree_source_template = "decision_tree.vhd.template"
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
    
    def __init__(self, classifier, yshelper):
        self.classifier = classifier
        self.yshelper = yshelper
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.source_dir =  f"{dir_path}/{self.__source_dir}"
    
    def generate_hdl_exact_implementations(self, destination):
        features = [{"name": f["name"], "nab": 0} for f in self.model_features]
        mkpath(destination)
        ax_dest = f"{destination}/exact/"
        mkpath(ax_dest)
        copy_file(self.source_dir + self.__extract_luts_file, ax_dest)
        copy_file(self.source_dir + self.__extract_pwr_file, ax_dest)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
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
            classes=self.classifier.model_classes)
        with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)
        tb_classifier = tb_classifier_template.render(
            features=features,
            classes=self.model_classes)
        with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
            out_file.write(tb_classifier)
        copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
        copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
        copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
        copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
        copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
        copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
        copy_file(self.source_dir + self.__constraint_file, ax_dest)
        copy_file(self.source_dir + self.__run_synth_file, ax_dest)
        copy_file(self.source_dir + self.__run_sim_file, ax_dest)
        for tree in self.classifier.trees:
            tree.generate_hdl_tree(ax_dest)
            tree.generate_hdl_exact_assertions(ax_dest)

    def generate_hdl_ps_ax_implementations(self, destination, configurations):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
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
                        for f, n in zip(self.model_features, conf)]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            for tree in self.trees:
                tree.generate_hdl_tree(ax_dest)
                tree.generate_hdl_exact_assertions(ax_dest)

    def generate_hdl_onestep_asl_ax_implementations(self, destination, configurations):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.model_features]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
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
        for conf, i in zip(configurations, range(len(configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            chunks = []
            count = 0
            for size in [len(t.get_graph().get_cells()) for t in self.trees]:
                chunks.append([conf[i+count] for i in range(size)])
                count += size
            for t, c in zip(self.trees, chunks):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(c)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_twostep_asl_ax_implementations(self, destination, outer_configurations, inner_configuration):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.model_features]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
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
        for conf, i in zip(outer_configurations, range(len(outer_configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            for t, i, c in zip(self.trees, range(len(self.trees)), conf):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(inner_configuration[i][c])
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_onestep_full_ax_implementations(self, destination, outer_configurations):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
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
                self.model_features, conf[:len(self.model_features)])]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            chunks = []
            count = 0
            for size in [len(t.get_graph().get_cells()) for t in self.trees]:
                chunks.append(
                    [conf[i+count+len(self.model_features)] for i in range(size)])
                count += size
            for t, c in zip(self.trees, chunks):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(c)
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_twostep_full_ax_implementations(self, destination, outer_configurations, inner_configuration):
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
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
                self.model_features, conf[:len(self.model_features)])]
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            classifier = classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            tb_classifier = tb_classifier_template.render(
                trees=trees_name,
                features=features,
                classes=self.model_classes)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            for t, n, c in zip(self.trees, range(len(self.trees)), conf[len(self.model_features):]):
                t.generate_hdl_tree(ax_dest)
                t.set_assertions_configuration(inner_configuration[n][c])
                t.generate_hdl_als_ax_assertions(ax_dest)

    def generate_hdl_onestep_asl_wc_ax_implementations(self, destination, configurations):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.model_features]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
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
        for conf, i in zip(configurations, range(len(configurations))):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)
            
            assertions_conf = [conf for _ in range(len(self.trees))]
            self.set_assertions_configuration(assertions_conf)
            for t in self.trees:
                t.generate_hdl_tree(ax_dest)
                t.generate_hdl_als_ax_assertions(ax_dest, "tree_0")

    def generate_hdl_twostep_asl_wc_ax_implementations(self, destination, outer_configurations, inner_configuration):
        features = [{"name": f["name"], "nab": 0}
                    for f in self.model_features]
        mkpath(destination)
        mkpath(f"{destination}/ax")
        copy_file(self.source_dir + self.__run_all_file, destination)
        copy_file(self.source_dir + self.__extract_luts_file, destination)
        copy_file(self.source_dir + self.__extract_pwr_file, destination)
        trees_name = [t.get_name() for t in self.trees]
        file_loader = FileSystemLoader(self.source_dir)
        env = Environment(loader=file_loader)
        tcl_template = env.get_template(self.__tcl_project_file)
        classifier_template = env.get_template(
            self.__vhdl_classifier_template_file)
        tb_classifier_template = env.get_template(
            self.__vhdl_tb_classifier_template_file)
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
        for i, outer_conf in enumerate(outer_configurations):
            ax_dest = f"{destination}/ax/configuration_{str(i)}"
            mkpath(ax_dest)
            with open(f"{ax_dest}/classifier.vhd", "w") as out_file:
                out_file.write(classifier)
            with open(f"{ax_dest}/tb_classifier.vhd", "w") as out_file:
                out_file.write(tb_classifier)
            with open(f"{ax_dest}/create_project.tcl", "w") as out_file:
                out_file.write(tcl_file)
            copy_file(self.source_dir + self.__vhdl_bnf_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_reg_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_decision_box_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_voter_source, ax_dest)
            copy_file(self.source_dir + self.__vhdl_debugfunc_source, ax_dest)
            copy_file(self.source_dir + self.__tcl_sim_file, ax_dest)
            copy_file(self.source_dir + self.__constraint_file, ax_dest)
            copy_file(self.source_dir + self.__run_synth_file, ax_dest)
            copy_file(self.source_dir + self.__run_sim_file, ax_dest)

            assertions_conf = [ inner_configuration[c] for c in outer_conf ]
            self.set_assertions_configuration(assertions_conf)
            for t in self.trees:
                self.generate_hdl_tree(t, ax_dest)
                self.generate_hdl_als_ax_assertions(t, ax_dest, "tree_0")


    def generate_tree_hdl(self, tree, destination):
        file_name = f"{destination}/decision_tree_{tree.name}.vhd"
        file_loader = FileSystemLoader(tree.source_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(tree.__vhdl_decision_tree_source_template)
        output = template.render(
            tree_name = tree.name,
            features  = tree.model_features,
            classes = tree.model_classes,
            boxes = [ b["box"].get_struct() for b in tree.decision_boxes ])
        with open(file_name, "w") as out_file:
            out_file.write(output)
        return file_name

    def generate_tree_hdl_exact_assertions(self, tree, destination):
        module_name = f"assertions_block_{tree.name}"
        file_name = f"{destination}/assertions_block_{tree.name}.vhd"
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

    def generate_hdl_als_ax_assertions(self, tree, destination, design_name = None):
        tree.yosys_helper.load_design(tree.name if design_name is None else design_name)
        tree.yosys_helper.to_aig(tree.current_als_configuration)
        tree.yosys_helper.clean()
        tree.yosys_helper.opt()
        tree.yosys_helper.write_verilog(f"{destination}/assertions_block_{tree.name}")
        
    
    def generate_design_for_als(self, tree, luts_tech):
        destination = "/tmp/pyals-rf/"
        mkpath(destination)
        mkpath(f"{destination}/vhd")
        file_name, module_name = self.generate_tree_hdl_exact_assertions(tree, destination)
        tree.yosys_helper.load_ghdl()
        tree.yosys_helper.reset()
        tree.yosys_helper.ghdl_read_and_elaborate([tree.bnf_vhd, file_name], module_name)
        tree.yosys_helper.prep_design(luts_tech)
        