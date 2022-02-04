"""
Copyright 2021 Salvatore Barone <salvatore.barone@unina.it>

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
import argparse, configparser
from pyosys import libyosys as ys
from .Classifier import *
from .MOP import *


class Worker:
    __report_file = "/pareto_front.csv"
    __pareto_view = "/pareto_front.pdf"

    def __init__(self):
        self.__print_tree = False
        self.__pmml_file = "model.pmml"
        self.__test_dataset = "dataset.txt"
        self.__output_dir = "output"
        self.ax_conf = None
        self.als_conf = None
        self.hw_conf = None
        self.singlestep_opt_conf = None
        self.twostep_opt_conf = None
        self.__cli_parser()
        self.__config_parser()
        self.__classifier = Classifier(self.als_conf)
        design = ys.Design()
        ys.run_pass("plugin -i ghdl", design)

    def work(self):
        if self.__output_dir != ".":
            mkpath(self.__output_dir)
        self.__classifier.parse(self.__pmml_file)
        if self.__print_tree:
            self.__classifier.dump()
            exit()

        self.__classifier.generate_hdl_exact_implementations(self.__output_dir)
        problem = None
        if self.ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
            if self.ax_conf.technique == AxConfig.Technique.PS:
                problem = SingleStepPsOnly(self.__classifier, self.__test_dataset, self.singlestep_opt_conf)
            elif self.ax_conf.technique == AxConfig.Technique.ALS:
                problem = SingleStepAlsOnly(self.__classifier, self.__test_dataset, self.singlestep_opt_conf)
            elif self.ax_conf.technique == AxConfig.Technique.FULL:
                problem = SingleStepCombined(self.__classifier, self.__test_dataset, self.singlestep_opt_conf)
            optimizer = AMOSA(self.singlestep_opt_conf.amosa_conf)
            optimizer.minimize(problem)
            optimizer.save_results(problem, self.__output_dir + self.__report_file)
            optimizer.plot_pareto(problem, self.__output_dir + self.__pareto_view)
            if self.ax_conf.technique == AxConfig.Technique.PS:
                self.__classifier.generate_hdl_ps_ax_implementations(self.__output_dir, optimizer.pareto_set())
            elif self.ax_conf.technique == AxConfig.Technique.ALS:
                self.__classifier.generate_hdl_onestep_asl_ax_implementations(self.__output_dir, optimizer.pareto_set())
            elif self.ax_conf.technique == AxConfig.Technique.FULL:
                self.__classifier.generate_hdl_onestep_full_ax_implementations(self.__output_dir, optimizer.pareto_set())
        elif self.ax_conf.strategy == AxConfig.Strategy.TWO_STEPS:
            if self.ax_conf.technique == AxConfig.Technique.ALS:
                problem = SecondStepOptimizerAlsOnly(self.__classifier, self.__test_dataset, self.twostep_opt_conf)
            elif self.ax_conf.technique == AxConfig.Technique.FULL:
                problem = SecondStepOptimizerCombined(self.__classifier, self.__test_dataset, self.twostep_opt_conf)
            optimizer = AMOSA(self.twostep_opt_conf.snd_amosa_conf)
            optimizer.minimize(problem)
            optimizer.save_results(problem, self.__output_dir + self.__report_file)
            optimizer.plot_pareto(problem, self.__output_dir + self.__pareto_view)
            if self.ax_conf.technique == AxConfig.Technique.ALS:
                self.__classifier.generate_hdl_twostep_asl_ax_implementations(self.__output_dir, optimizer.pareto_set())
            elif self.ax_conf.technique == AxConfig.Technique.FULL:
                self.__classifier.generate_hdl_twostep_full_ax_implementations(self.__output_dir, optimizer.pareto_set())
        print("All done! Take a look at the ", self.__output_dir, " directory.")

    def __cli_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dump", help = "Dump the model", action = "store_true")
        parser.add_argument("--config", type = str, help = "path of the configuration file", default = "config.ini")
        parser.add_argument("--pmml", type = str, help = "specify the input PMML file", default = "model.pmml")
        parser.add_argument("--dataset", type = str, help = "specify the file name for the input dataset", default = "dataset.txt")
        parser.add_argument("--output", type = str, help = "Output directory. Everything will be placed there.", default = "output/")
        args, left = parser.parse_known_args()
        sys.argv = sys.argv[:1] + left
        self.__print_tree = args.dump
        self.__config_file = args.config
        self.__pmml_file = args.pmml
        self.__test_dataset = args.dataset
        self.__output_dir = args.output

    def __config_parser(self):
        config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]}, empty_lines_in_values=False)
        config.read(self.__config_file)
        self.ax_conf = AxConfig(
            config["approximation"]["technique"],
            config["approximation"]["strategy"])
        self.als_conf = ALSConfig(
            config["als"]["cut_size"] if "cut_size" in config["als"] else "4",
            config["als"]["catalog"] if "catalog" in config["als"] else "lut_catalog.db",
            config["als"]["solver"] if "solver" in config["als"] else "boolector",
            int(config["als"]["timeout"]) if "timeout" in config["als"] else 60000)
        if self.ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
            error_conf = ErrorConfig("eprob", float(config["singlestage"]["error_threshold"]) if "error_threshold" in config["singlestage"] else .5, 0)
            amosa_conf = AMOSAConfig(
                int(config["singlestage"]["archive_hard_limit"]) if "archive_hard_limit" in config["singlestage"] else 50,
                int(config["singlestage"]["archive_soft_limit"]) if "archive_soft_limit" in config["singlestage"] else 100,
                int(config["singlestage"]["archive_gamma"]) if "archive_gamma" in config["singlestage"] else 3,
                int(config["singlestage"]["hill_climbing_iterations"]) if "hill_climbing_iterations" in config["singlestage"] else 100,
                float(config["singlestage"]["initial_temperature"]) if "initial_temperature" in config["singlestage"] else 500,
                float(config["singlestage"]["final_temperature"]) if "final_temperature" in config["singlestage"] else 0.0000001,
                float(config["singlestage"]["cooling_factor"]) if "cooling_factor" in config["singlestage"] else 0.8,
                int(config["singlestage"]["annealing_iterations"]) if "annealing_iterations" in config["singlestage"] else 100)
            self.singlestep_opt_conf = SingleStepOptimizerConf(error_conf, amosa_conf)
        else:
            fst_error_conf = ErrorConfig(
                config["twostages"]["fst_error_metric"] if  "fst_error_metric" in config["twostages"] else "eprob",
                float(config["twostages"]["fst_error_threshold"]) if "error_threshold" in config["twostages"] else .5,
                int(config["twostages"]["fst_num_vectors"] if  "fst_num_vectors" in config["twostages"] else 0))
            fst_amosa_conf = AMOSAConfig(
                int(config["twostages"]["fst_archive_hard_limit"]) if "fst_archive_hard_limit" in config["twostages"] else 50,
                int(config["twostages"]["fst_archive_soft_limit"]) if "fst_archive_soft_limit" in config["twostages"] else 100,
                int(config["twostages"]["fst_archive_gamma"]) if "fst_archive_gamma" in config["twostages"] else 3,
                int(config["twostages"]["fst_hill_climbing_iterations"]) if "fst_hill_climbing_iterations" in config["twostages"] else 100,
                float(config["twostages"]["fst_initial_temperature"]) if "fst_initial_temperature" in config["twostages"] else 500,
                float(config["twostages"]["fst_final_temperature"]) if "fst_final_temperature" in config["twostages"] else 0.0000001,
                float(config["twostages"]["fst_cooling_factor"]) if "fst_cooling_factor" in config["twostages"] else 0.8,
                int(config["twostages"]["fst_annealing_iterations"]) if "fst_annealing_iterations" in config["twostages"] else 100)
            snd_error_conf = ErrorConfig(
                config["twostages"]["snd_error_metric"] if "snd_error_metric" in config["twostages"] else "eprob",
                float(config["twostages"]["snd_error_threshold"]) if "snd_error_threshold" in config["twostages"] else .5,
                int(config["twostages"]["snd_num_vectors"] if "snd_num_vectors" in config["twostages"] else 0))
            snd_amosa_conf = AMOSAConfig(
                int(config["twostages"]["snd_archive_hard_limit"]) if "snd_archive_hard_limit" in config["twostages"] else 50,
                int(config["twostages"]["snd_archive_soft_limit"]) if "snd_archive_soft_limit" in config["twostages"] else 100,
                int(config["twostages"]["snd_archive_gamma"]) if "snd_archive_gamma" in config["twostages"] else 3,
                int(config["twostages"]["snd_hill_climbing_iterations"]) if "snd_hill_climbing_iterations" in config["twostages"] else 100,
                float(config["twostages"]["snd_initial_temperature"]) if "snd_initial_temperature" in config["twostages"] else 500,
                float(config["twostages"]["snd_final_temperature"]) if "snd_final_temperature" in config["twostages"] else 0.0000001,
                float(config["twostages"]["snd_cooling_factor"]) if "snd_cooling_factor" in config["twostages"] else 0.8,
                int(config["twostages"]["snd_annealing_iterations"]) if "snd_annealing_iterations" in config["twostages"] else 100)
            self.twostep_opt_conf = TwoStepsOptimizerConf(fst_error_conf, fst_amosa_conf, snd_error_conf, snd_amosa_conf)