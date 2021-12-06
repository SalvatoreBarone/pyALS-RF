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
from .Configs import *
from .OneStepOptimizer import *
from .SecondStepOptimizer import *


class Worker:
  __report_file = "/pareto_front.csv"
  __pareto_view = "/pareto_front.pdf"

  def __init__(self):
    self.__print_tree = False
    self.__pmml_file = "model.pmml"
    self.__test_dataset = "dataset.txt"
    self.__output_dir = "output"
    self.__ax_conf = AxConfig("full", "one")
    self.__als_conf = ALSConfig("6", "lut_catalog.db", 120000)
    self.__final_opt_conf = NSGAConfig(500, 123, 0.9, 1, 0.1, 20, 5)
    self.__als_opt_conf = NSGAConfig(500, 123, 0.9, 1, 0.1, 20, 5)
    self.__cli_parser()
    self.__config_parser()
    self.__classifier = Classifier(self.__als_conf)
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
    if (self.__ax_conf.technique == AxConfig.Technique.PS):
      optimizer = OneStepOptimizer(self.__ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
      optimizer.optimize()
      optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
      optimizer.get_report(self.__output_dir + self.__report_file)
      self.__classifier.generate_hdl_ps_ax_implementations(self.__output_dir, optimizer.get_individuals())
    else:
      if self.__ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
        optimizer = OneStepOptimizer(self.__ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
        optimizer.optimize()
        optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
        optimizer.get_report(self.__output_dir + self.__report_file)
        if (self.__ax_conf.technique == AxConfig.Technique.ALS):
          self.__classifier.generate_hdl_onestep_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
        elif(self.__ax_conf.technique == AxConfig.Technique.FULL):
          self.__classifier.generate_hdl_onestep_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
      else:
        self.__classifier.generate_first_step_ax_assertions(self.__test_dataset, self.__output_dir, self.__als_opt_conf)
        optimizer = SecondStepOptimizer(self.__ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
        optimizer.optimize()
        optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
        optimizer.get_report(self.__output_dir + self.__report_file)
        if (self.__ax_conf.technique == AxConfig.Technique.ALS):
          self.__classifier.generate_hdl_twostep_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
        elif (self.__ax_conf.technique == AxConfig.Technique.FULL):
          self.__classifier.generate_hdl_twostep_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
    print("All done! Take a look at the ", self.__output_dir, " directory.")

  def __cli_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", help="Dump the model", action="store_true")
    parser.add_argument("--config", type = str, help = "path of the configuration file", default = "config.ini")
    parser.add_argument("--pmml", type = str, help="specify the input PMML file", default = "model.pmml")
    parser.add_argument("--dataset", type = str, help="specify the file name for the input dataset", default = "dataset.txt")
    parser.add_argument("--output", type = str, help="Output directory. Everything will be placed there.", default = "output/")
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left
    self.__print_tree = args.dump
    self.__config_file = args.config
    self.__pmml_file = args.pmml
    self.__test_dataset = args.dataset
    self.__output_dir = args.output

  def __config_parser(self):
    config = configparser.ConfigParser()
    config.read(self.__config_file)
    self.__ax_conf = AxConfig(
      config["approximation"]["technique"],
      config["approximation"]["strategy"])

    self.__als_conf = ALSConfig(
      config["als"]["luttech"] if "luttech" in config["als"] else "6",
      config["als"]["catalog"] if "catalog" in config["als"] else "lut_catalog.db",
      int(config["als"]["timeout"]) if "timeout" in config["als"] else 60000,)

    self.__final_opt_conf = NSGAConfig(
      int(config["final-optimization"]["population_size"]) if "population_size" in config["final-optimization"] else 0,
      int(config["final-optimization"]["iterations"]) if "iterations" in config["final-optimization"] else 123,
      float(config["final-optimization"]["crossover_probability"]) if "crossover_probability" in config["final-optimization"] else 0.9,
      float(config["final-optimization"]["crossover_eta"]) if "crossover_eta" in config["final-optimization"] else 1,
      float(config["final-optimization"]["mutation_probability"]) if "mutation_probability" in config["final-optimization"] else 0,
      float(config["final-optimization"]["mutation_eta"]) if "mutation_eta" in config["final-optimization"] else 20,
      float(config["final-optimization"]["max_accuracy_loss"]) if "max_accuracy_loss" in config["final-optimization"] else 5)

    self.__als_opt_conf = NSGAConfig(
      int(config["als-optimization"]["population_size"]) if "population_size" in config["als-optimization"] else 0,
      int(config["als-optimization"]["iterations"]) if "iterations" in config["als-optimization"] else 123,
      float(config["als-optimization"]["crossover_probability"]) if "crossover_probability" in config["als-optimization"] else 0.9,
      float(config["als-optimization"]["crossover_eta"]) if "crossover_eta" in config["als-optimization"] else 1,
      float(config["als-optimization"]["mutation_probability"]) if "mutation_probability" in config["als-optimization"] else 0,
      float(config["als-optimization"]["mutation_eta"]) if "mutation_eta" in config["als-optimization"] else 20,
      float(config["als-optimization"]["max_error_frequency"]) if "max_error_frequency" in config["als-optimization"] else 10)

