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
from .Optimizer import *


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

    if self.__ax_conf.strategy == AxConfig.Strategy.ONE_STEP:
      optimizer = Optimizer(self.__ax_conf.technique, self.__classifier, self.__test_dataset, self.__final_opt_conf)
      optimizer.optimize()
      optimizer.plot_pareto(self.__output_dir + self.__pareto_view)
      optimizer.get_report(self.__output_dir + self.__report_file)
      self.__classifier.generate_implementations(self.__output_dir)
      if (self.__ax_conf.technique == AxConfig.Technique.ALS):
        self.__classifier.generate_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
      elif(self.__ax_conf.technique == AxConfig.Technique.PS):
        self.__classifier.generate_ps_ax_implementations(self.__output_dir, optimizer.get_individuals())
      elif(self.__ax_conf.technique == AxConfig.Technique.FULL):
        self.__classifier.generate_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
    else:
      pass

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
    print("Dump:               ", self.__print_tree)
    print("Config file         ", self.__config_file)
    print("PMML file:          ", self.__pmml_file)
    print("Test dataset:       ", self.__test_dataset)
    print("Outoput directory:  ", self.__output_dir)

  def __config_parser(self):
    config = configparser.ConfigParser()
    config.read(self.__config_file)
    self.__ax_conf = AxConfig(config["approximation"]["technique"], config["approximation"]["strategy"])
    self.__als_conf = ALSConfig(config["als"]["luttech"], config["als"]["catalog"], config["als"]["timeout"])
    self.__final_opt_conf = NSGAConfig(int(config["final-optimization"]["population_size"]),
                                       int(config["final-optimization"]["iterations"]),
                                       float(config["final-optimization"]["crossover_probability"]),
                                       float(config["final-optimization"]["crossover_eta"]),
                                       float(config["final-optimization"]["mutation_probability"]),
                                       float(config["final-optimization"]["mutation_eta"]),
                                       float(config["final-optimization"]["max_accuracy_loss"]))

    self.__als_opt_conf = NSGAConfig(int(config["als-optimization"]["population_size"]),
                                     int(config["als-optimization"]["iterations"]),
                                     float(config["als-optimization"]["crossover_probability"]),
                                     float(config["als-optimization"]["crossover_eta"]),
                                     float(config["als-optimization"]["mutation_probability"]),
                                     float(config["als-optimization"]["mutation_eta"]),
                                     float(config["als-optimization"]["max_error_frequency"]))

    print("Technique:          ", self.__ax_conf.technique)
    print("Strategy:           ", self.__ax_conf.strategy)

    print("ALS LUTs:           ", self.__als_conf.luttech)
    print("ALS catalog:        ", self.__als_conf.catalog)
    print("ALS timeout:        ", self.__als_conf.timeout)

    print("NSGA-II pop.size:   ", self.__final_opt_conf.pop_size)
    print("NSGA-II iterations: ", self.__final_opt_conf.iterations)
    print("NSGA-II Pcross:     ", self.__final_opt_conf.cross_p)
    print("NSGA-II Ncross:     ", self.__final_opt_conf.cross_eta)
    print("NSGA-II Pmut:       ", self.__final_opt_conf.mut_p)
    print("NSGA-II Nmut:       ", self.__final_opt_conf.mut_eta)
    print("NSGA-II max. error: ", self.__final_opt_conf.max_error)

    print("NSGA-II pop.size:   ", self.__als_opt_conf.pop_size)
    print("NSGA-II iterations: ", self.__als_opt_conf.iterations)
    print("NSGA-II Pcross:     ", self.__als_opt_conf.cross_p)
    print("NSGA-II Ncross:     ", self.__als_opt_conf.cross_eta)
    print("NSGA-II Pmut:       ", self.__als_opt_conf.mut_p)
    print("NSGA-II Nmut:       ", self.__als_opt_conf.mut_eta)
    print("NSGA-II max. error: ", self.__als_opt_conf.max_error)
