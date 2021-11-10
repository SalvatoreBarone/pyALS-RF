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
import sys, csv
import numpy as np
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from .Classifier import *

"""
@brief Implements the multi-objective optimization problem through which the design-space exploration phase is driven.

@details
As stated in Kalyanmoy Deb "Multi-Objective Optimization Using Evolutionary Algorithms" John Wiley & Sons, Inc., New 
York, NY, USA, 2001. ISBN 047187339X, a multi-objective optimization problem (MOP) has several objective functions with
subject to inequality and equality constraints to optimize. The goal is to find a set of solutions that do not have any
constraint violation and are as good as possible regarding all its objectives values.
The problem definition in its general form is given by the following equations, which define MOP with N variables, M 
objectives, J inequality and K equality constraints. Moreover, for each variable, both the lower and upper variable 
boundaries are defined.

\f{eqnarray*}[ 
 min/max  f_m(x) \; m=1 \cdots M \\
 s.t.   g_j(x) \leq 0 \;j=1 \cdots J \\
        h_k(x) = 0 \; k = 1 \cdots K \\
        x_i^L \leq x_i \leq x_i^U \; i = 1 \cdots N
\f}

In order to perform multi-objective optimization, the Optimizer class exploits the NSGA-II implementation provided by
the pymoo module, which requires the following
  1. Implementation of a Problem (element-wise class, in our case)
  2. Initialization of an Algorithm (in our case, NSGA-II)
  3. Definition of a Termination Criterion (the number of NSGA-II generations)
  4. Optimize (minimize error and hardware requirements, in our case)
"""
class Optimizer:

#  class PrecisionScalingOnly(ElementwiseProblem):

  class ALSOnly(ElementwiseProblem):
    def __init__(self, classifier, dataset_csv, threads):
      self.classifier = classifier
      self.test_dataset = self.classifier.preload_dataset(dataset_csv)
      self.threads = threads
      #self.total_bits = self.classifier.get_total_bits()
      self.total_gates = self.classifier.get_current_required_aig_nodes()
      self.baseline_accuracy = self.classifier.evaluate_preloaded_dataset(self.test_dataset)
      self.als_genes_per_tree = self.classifier.get_als_genes_per_tree()
      self.ngenes = sum( self.als_genes_per_tree )
      lower_bound = np.zeros(self.ngenes, dtype = np.uint32)
      als_ub = self.classifier.get_als_genes_upper_bound()
      upper_bound = np.array(als_ub, dtype = np.uint32)
      super().__init__(n_var= self.ngenes, n_obj = 2, n_constr = 0, xl = lower_bound, xu = upper_bound)

    def genotype_to_phenotype(self, X):
      configurations = []
      count = 0
      for size in self.als_genes_per_tree:
        configurations.append([X[i+count] for i in range(size)])
        count += size
      self.classifier.set_assertions_configuration(configurations)

    def _evaluate(self, X, out, *args, **kwargs):
      self.genotype_to_phenotype(X)
      out["F"] = [
        self.baseline_accuracy - self.classifier.evaluate_preloaded_dataset(self.test_dataset), 
        #self.classifier.get_total_retained(), 
        self.classifier.get_current_required_aig_nodes()]

  #class CombinedTechniques(ElementwiseProblem):

  def __init__(self, classifier, test_dataset, n_threads, nsgaii_pop_size, nsgaii_iter, nsgaii_cross_prob, nsgaii_cross_eta, nsgaii_mut_prob, nsgaii_mut_eta):
    self.problem = Optimizer.ALSOnly(classifier, test_dataset, n_threads)
    self.algorithm = NSGA2(
      pop_size = nsgaii_pop_size,
      n_offsprings = None,
      sampling = get_sampling("int_random"),
      crossover = get_crossover("int_sbx", prob = nsgaii_cross_prob, eta = nsgaii_cross_eta),
      mutation = get_mutation("int_pm", prob = nsgaii_mut_prob, eta = nsgaii_mut_eta),
      eliminate_duplicates = True)
    self.termination = get_termination('n_gen', nsgaii_iter)
    self.result = None

  def optimize(self):
    self.result = minimize(self.problem, self.algorithm, self.termination, verbose = True)

  def print_pareto(self):
    row_format = "{:<10}" * (len(self.result.pop.get("F")[0])) + "{:>3}" * (len(self.result.pop.get("X")[0]))
    print("Final population:\nError     Cost        Chromosome")
    for fitness, chromosome in zip(self.result.pop.get("F"), self.result.pop.get("X")):
      print(row_format.format(*fitness, *chromosome))

  def get_report(self, report_file):
    original_stdout = sys.stdout
    row_format = "{:<10};" * (len(self.result.pop.get("F")[0])) + "{:>3};" * (len(self.result.pop.get("X")[0]))
    with open(report_file, "w") as file:
      sys.stdout = file
      print("Final population:\nError     Cost        Chromosome")
      for fitness, chromosome in zip(self.result.pop.get("F"), self.result.pop.get("X")):
        print(row_format.format(*fitness, *chromosome))
    sys.stdout = original_stdout
  
  def get_individuals(self):
    return self.result.pop.get("X")
  
  def get_elapsed_time(self):
    return self.result.exec_time
