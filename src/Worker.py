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
import sys, argparse
from distutils.dir_util import mkpath
from multiprocessing import cpu_count
from pyosys import libyosys as ys
from .Classifier import *
from .Optimizer import *


class Worker:
  def __init__(self):
    self.__n_threads = 1
    self.__print_tree = False
    self.__pmml_file = "model.pmml"
    self.__test_dataset = "dataset.txt"
    self.__output_dir = "output"
    self.__report_file = "/Pareto.csv"
    self.__nsgaii_pop_size = 500
    self.__nsgaii_iter = 11
    self.__nsgaii_cross_prob = 0.9
    self.__nsgaii_cross_eta = 50
    self.__nsgaii_mut_prob = 0.9
    self.__nsgaii_mut_eta = 50
    self.__als_lut = "4"
    self.__als_catalog = "catalog"
    self.__als_timeout = 60000
    self.__cli_parser()
    self.__classifier = Classifier(self.__als_lut, self.__als_catalog, self.__als_timeout)
    design = ys.Design()
    ys.run_pass("plugin -i ghdl", design)

    

  def work(self):
    if self.__output_dir != ".":
      mkpath(self.__output_dir)
    self.__classifier.parse(self.__pmml_file)
    if self.__print_tree:
      self.__classifier.dump()
    print("Performing design-space exploration using NSGA-II. Please wait patiently, this may take quite a long time...")
    optimizer = Optimizer(self.__axtechnique, self.__classifier, self.__test_dataset, self.__n_threads, self.__nsgaii_pop_size, self.__nsgaii_iter, self.__nsgaii_cross_prob, self.__nsgaii_cross_eta, self.__nsgaii_mut_prob, self.__nsgaii_mut_eta)
    optimizer.optimize()
    optimizer.print_pareto()
    optimizer.get_report(self.__report_file)
    print("Performing HDL code generation using the embedded coder.")
    self.__classifier.generate_implementations(self.__output_dir)
    if (self.__axtechnique == Optimizer.AxTechnique.ALS):
      self.__classifier.generate_asl_ax_implementations(self.__output_dir, optimizer.get_individuals())
    elif(self.__axtechnique == Optimizer.AxTechnique.PS):
      self.__classifier.generate_ps_ax_implementations(self.__output_dir, optimizer.get_individuals())
    elif(self.__axtechnique == Optimizer.AxTechnique.FULL):
      self.__classifier.generate_full_ax_implementations(self.__output_dir, optimizer.get_individuals())
    print("All done! Take a look at the ", self.__output_dir, " directory.")

  def __cli_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax", type = str, help="specify the AxC technique to be adopted [ps, als, full]", default = "")
    parser.add_argument("--pmml", type = str, help="specify the input PMML file", default = "model.pmml")
    parser.add_argument("--dump", help="Dump the model", action="store_true")
    parser.add_argument("--dataset", type = str, help="specify the file name for the input dataset", default = "dataset.txt")
    parser.add_argument("--output", type = str, help="Output directory. Everything will be placed there.", default = "output/")
    parser.add_argument("--threads", type = int, help="specify the amount of parallel worker threads.", default = cpu_count())
    parser.add_argument("--popsize", type = int, help="NSGA-II population size.", default = 500)
    parser.add_argument("--iter", type = int, help="NSGA-II termination criteria, in terms of iterations.", default = 11)
    parser.add_argument("--pcross", type = float, help="NSGA-II crossover probability.", default = .9)
    parser.add_argument("--etac", type = float, help="NSGA-II crossover distribution index.", default = 50)
    parser.add_argument("--pmut", type = float, help="NSGA-II mutation probability.", default = .9)
    parser.add_argument("--etam", type = float, help="NSGA-II mutation distribution index.", default = 50)
    parser.add_argument("--lut", type = str, help = "Select the LUT technology to be adopted (4-LUT, 6-LUT...) during ALS", default = "4")
    parser.add_argument("--catalog", type = str, help = "Path to the ALS LUT cache", default = "lut_catalog.db")
    parser.add_argument("--timeout", type = int, help = "Set the time budget for the SMT synthesis of LUTs, in ms, during ALS", default = 60000)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if (args.ax == "als"):
      self.__axtechnique = Optimizer.AxTechnique.ALS
    elif(args.ax == "ps"):
      self.__axtechnique = Optimizer.AxTechnique.PS
    elif(args.ax == "full"):
      self.__axtechnique = Optimizer.AxTechnique.FULL
    else:
      raise ValueError('Approximation technique not recognized')
      
    self.__pmml_file = args.pmml
    self.__test_dataset = args.dataset
    self.__print_tree = args.dump
    self.__n_threads = int(args.threads)
    self.__nsgaii_pop_size = int(args.popsize)
    self.__nsgaii_iter = int(args.iter)
    self.__nsgaii_cross_prob = float(args.pcross)
    self.__nsgaii_cross_eta = float(args.etac)
    self.__nsgaii_mut_prob = float(args.pmut)
    self.__nsgaii_mut_eta = float(args.etam)
    self.__output_dir = args.output
    self.__report_file = self.__output_dir + self.__report_file
    self.__als_lut = args.lut
    self.__als_catalog = args.catalog
    self.__als_timeout = int(args.timeout)
    print("PMML file:          ", self.__pmml_file)
    print("Test dataset:       ", self.__test_dataset)
    print("Dump:               ", self.__print_tree)
    print("Parallel workers:   ", self.__n_threads)
    print("NSGA-II pop.size:   ", self.__nsgaii_pop_size)
    print("NSGA-II iterations: ", self.__nsgaii_iter)
    print("NSGA-II Pcross:     ", self.__nsgaii_cross_prob)
    print("NSGA-II Ncross:     ", self.__nsgaii_cross_eta)
    print("NSGA-II Pmut:       ", self.__nsgaii_mut_prob)
    print("NSGA-II Nmut:       ", self.__nsgaii_mut_eta)
    print("Output dir.:        ", self.__output_dir)
    print("Report file:        ", self.__report_file)
    print("ALS LUTs:           ", self.__als_lut)
    print("ALS catalog:        ", self.__als_catalog)
    print("ALS timeout:        ", self.__als_timeout)


