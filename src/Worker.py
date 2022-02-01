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
import os, argparse, configparser
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
        self.onestep_opt_conf = None
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
        if (self.ax_conf.technique == AxConfig.Technique.PS):
            pass
            # optimizer = OneStepOptimizer(self.ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
            # optimizer.optimize()
            # optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
            # optimizer.get_report(self.__output_dir + self.__report_file)
            # self.__classifier.generate_hdl_ps_ax_implementations(self.__output_dir, optimizer.get_individuals())
        else:
            pass
            # if self.ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
            #   optimizer = OneStepOptimizer(self.ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
            #   optimizer.optimize()
            #   optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
            #   optimizer.get_report(self.__output_dir + self.__report_file)
            #   if (self.ax_conf.technique == AxConfig.Technique.ALS):
            #     self.__classifier.generate_hdl_onestep_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
            #   elif(self.ax_conf.technique == AxConfig.Technique.FULL):
            #     self.__classifier.generate_hdl_onestep_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
            # else:
            #   self.__classifier.generate_first_step_ax_assertions(self.__test_dataset, self.__output_dir, self.__als_opt_conf)
            #   optimizer = SecondStepOptimizer(self.ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
            #   optimizer.optimize()
            #   optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
            #   optimizer.get_report(self.__output_dir + self.__report_file)
            #   if (self.ax_conf.technique == AxConfig.Technique.ALS):
            #     self.__classifier.generate_hdl_twostep_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
            #   elif (self.ax_conf.technique == AxConfig.Technique.FULL):
            #     self.__classifier.generate_hdl_twostep_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
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
        print(self.ax_conf.technique, self.ax_conf.strategy)

        self.als_conf = ALSConfig(
            config["als"]["cut_size"] if "cut_size" in config["als"] else "4",
            config["als"]["catalog"] if "catalog" in config["als"] else "lut_catalog.db",
            config["als"]["solver"] if "solver" in config["als"] else "boolector",
            int(config["als"]["timeout"]) if "timeout" in config["als"] else 60000)
        print(self.als_conf.cut_size, self.als_conf.catalog, self.als_conf.solver, self.als_conf.timeout)

        if self.ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
            error_conf = ErrorConfig("eprob", float(config["singlestage"]["error_threshold"]) if "error_threshold" in config["singlestage"] else .5, 0)
            print(error_conf.metric, error_conf.threshold, error_conf.threshold)
            hw_conf = HwConfig(list(map(str, config.getlist('singlestage', 'hw_metrics'))) if "hw_metrics" in config["singlestage"] else ["gates"])
            print(hw_conf.metrics)
            amosa_conf = AMOSAConfig(
                int(config["singlestage"]["archive_hard_limit"]) if "archive_hard_limit" in config["singlestage"] else 50,
                int(config["singlestage"]["archive_soft_limit"]) if "archive_soft_limit" in config["singlestage"] else 100,
                int(config["singlestage"]["archive_gamma"]) if "archive_gamma" in config["singlestage"] else 3,
                int(config["singlestage"]["hill_climbing_iterations"]) if "hill_climbing_iterations" in config["singlestage"] else 100,
                float(config["singlestage"]["initial_temperature"]) if "initial_temperature" in config["singlestage"] else 500,
                float(config["singlestage"]["final_temperature"]) if "final_temperature" in config["singlestage"] else 0.0000001,
                float(config["singlestage"]["cooling_factor"]) if "cooling_factor" in config["singlestage"] else 0.8,
                int(config["singlestage"]["annealing_iterations"]) if "annealing_iterations" in config["singlestage"] else 100)
            print(amosa_conf.archive_hard_limit,
                  amosa_conf.archive_soft_limit,
                  amosa_conf.archive_gamma,
                  amosa_conf.hill_climbing_iterations,
                  amosa_conf.initial_temperature,
                  amosa_conf.final_temperature,
                  amosa_conf.cooling_factor,
                  amosa_conf.final_temperature)
            self.onestep_opt_conf = SingleStepOptimizerConf(error_conf, hw_conf, amosa_conf)
        else:
            fst_error_conf = ErrorConfig(
                config["twostages"]["fst_error_metric"] if  "fst_error_metric" in config["twostages"] else "eprob",
                float(config["twostages"]["fst_error_threshold"]) if "error_threshold" in config["twostages"] else .5,
                int(config["twostages"]["fst_num_vectors"] if  "fst_num_vectors" in config["twostages"] else 0))
            print(fst_error_conf.metric, fst_error_conf.threshold, fst_error_conf.threshold)
            fst_hw_conf = HwConfig(list(map(str, config.getlist('twostages', 'fst_hw_metrics'))) if "fst_hw_metrics" in config["twostages"] else ["gates"])
            print(fst_hw_conf.metrics)
            fst_amosa_conf = AMOSAConfig(
                int(config["twostages"]["fst_archive_hard_limit"]) if "fst_archive_hard_limit" in config["twostages"] else 50,
                int(config["twostages"]["fst_archive_soft_limit"]) if "fst_archive_soft_limit" in config["twostages"] else 100,
                int(config["twostages"]["fst_archive_gamma"]) if "fst_archive_gamma" in config["twostages"] else 3,
                int(config["twostages"]["fst_hill_climbing_iterations"]) if "fst_hill_climbing_iterations" in config["twostages"] else 100,
                float(config["twostages"]["fst_initial_temperature"]) if "fst_initial_temperature" in config["twostages"] else 500,
                float(config["twostages"]["fst_final_temperature"]) if "fst_final_temperature" in config["twostages"] else 0.0000001,
                float(config["twostages"]["fst_cooling_factor"]) if "fst_cooling_factor" in config["twostages"] else 0.8,
                int(config["twostages"]["fst_annealing_iterations"]) if "fst_annealing_iterations" in config["twostages"] else 100)
            print(fst_amosa_conf.archive_hard_limit,
                  fst_amosa_conf.archive_soft_limit,
                  fst_amosa_conf.archive_gamma,
                  fst_amosa_conf.hill_climbing_iterations,
                  fst_amosa_conf.initial_temperature,
                  fst_amosa_conf.final_temperature,
                  fst_amosa_conf.cooling_factor,
                  fst_amosa_conf.final_temperature)

            snd_error_conf = ErrorConfig(
                config["twostages"]["snd_error_metric"] if "snd_error_metric" in config["twostages"] else "eprob",
                float(config["twostages"]["snd_error_threshold"]) if "snd_error_threshold" in config["twostages"] else .5,
                int(config["twostages"]["snd_num_vectors"] if "snd_num_vectors" in config["twostages"] else 0))
            print(snd_error_conf.metric, snd_error_conf.threshold, snd_error_conf.threshold)
            snd_hw_conf = HwConfig(list(map(str, config.getlist('twostages', 'snd_hw_metrics'))) if "snd_hw_metrics" in config["twostages"] else ["gates"])
            print(snd_hw_conf.metrics)
            snd_amosa_conf = AMOSAConfig(
                int(config["twostages"]["snd_archive_hard_limit"]) if "snd_archive_hard_limit" in config["twostages"] else 50,
                int(config["twostages"]["snd_archive_soft_limit"]) if "snd_archive_soft_limit" in config["twostages"] else 100,
                int(config["twostages"]["snd_archive_gamma"]) if "snd_archive_gamma" in config["twostages"] else 3,
                int(config["twostages"]["snd_hill_climbing_iterations"]) if "snd_hill_climbing_iterations" in config["twostages"] else 100,
                float(config["twostages"]["snd_initial_temperature"]) if "snd_initial_temperature" in config["twostages"] else 500,
                float(config["twostages"]["snd_final_temperature"]) if "snd_final_temperature" in config["twostages"] else 0.0000001,
                float(config["twostages"]["snd_cooling_factor"]) if "snd_cooling_factor" in config["twostages"] else 0.8,
                int(config["twostages"]["snd_annealing_iterations"]) if "snd_annealing_iterations" in config["twostages"] else 100)
            print(snd_amosa_conf.archive_hard_limit,
                  snd_amosa_conf.archive_soft_limit,
                  snd_amosa_conf.archive_gamma,
                  snd_amosa_conf.hill_climbing_iterations,
                  snd_amosa_conf.initial_temperature,
                  snd_amosa_conf.final_temperature,
                  snd_amosa_conf.cooling_factor,
                  snd_amosa_conf.final_temperature)
            self.twostep_opt_conf = TwoStepsOptimizerConf(fst_error_conf, fst_hw_conf, fst_amosa_conf, snd_error_conf, snd_hw_conf, snd_amosa_conf)