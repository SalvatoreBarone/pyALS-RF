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
import time
from enum import Enum
from multiprocessing import Pool
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from .Classifier import *
from .Utility import *

class Optimizer:
    class AxTechnique(Enum):
        ALS = 1,
        PS = 2,
        FULL = 3

    class MOP:
        def __init__(self, classifier, dataset_csv, threads, emax):
            self._threads = threads
            self.emax = emax
            dataset = classifier.preload_dataset(dataset_csv)
            self._n_samples = len(dataset)
            dataset_partioned = list_partitioning(dataset, threads)
            classifier.reset_assertion_configuration()
            classifier.reset_nabs_configuration()
            classifiers = [ copy.deepcopy(classifier) ] * threads
            self._partitions = [ [c, d] for c, d in zip(classifiers, dataset_partioned) ]
            start_time = time.time()
            self.baseline_accuracy = self.__evaluate_dataset()
            duration = time.time() - start_time
            print(f"Baseline accuracy: {self.baseline_accuracy}. Took {duration} sec.")

        def __evaluate_dataset(self):
            with Pool(self._threads) as pool:
                res = pool.starmap(evaluate_preloaded_dataset, self._partitions)
            return sum(res) * 100 / self._n_samples

        def _get_accuracy_loss(self):
            return self.baseline_accuracy - self.__evaluate_dataset()

    class PSOnly(MOP, ElementwiseProblem):
        def __init__(self, classifier, dataset_csv, threads, emax):
            self.total_bits = classifier.get_total_bits()
            print(f"Baseline requirements: {self.total_bits} bits")
            self.als_genes_per_tree = classifier.get_als_genes_per_tree()
            self.ngenes = len( classifier.get_features())
            print(f"# genes: {self.ngenes}")
            lower_bound = np.zeros(self.ngenes, dtype = np.uint32)
            upper_bound = np.array([53] * self.ngenes, dtype = np.uint32)
            Optimizer.MOP.__init__(self, classifier,  dataset_csv, threads, emax)
            ElementwiseProblem.__init__(self, n_var = self.ngenes, n_obj = 2, n_constr = 1, xl = lower_bound, xu = upper_bound)

        def __genotype_to_phenotype(self, X):
            features = self._partitions[0][0].get_features()
            for item in self._partitions:
                item[0].set_nabs([ {"name": f["name"], "nab" : x} for f, x in zip(features, X) ])

        def _evaluate(self, X, out, *args, **kwargs):
            self.__genotype_to_phenotype(X)
            err = self._get_accuracy_loss()
            bits = self._partitions[0][0].get_total_retained()
            out["F"] = [err, bits]
            out["G"] = err - self.emax

    class ALSOnly(MOP, ElementwiseProblem):
        def __init__(self, classifier, dataset_csv, threads, emax):
            self.als_genes_per_tree = classifier.get_als_genes_per_tree()
            self.ngenes = sum(self.als_genes_per_tree)
            print(f"Genes: {self.als_genes_per_tree} Tot. #genes: {self.ngenes}")
            gates_list = classifier.get_current_required_aig_nodes()
            self.total_gates = sum(gates_list)
            print(f"Baseline requirements: [{gates_list}] {self.total_gates} gates")
            lower_bound = np.zeros(self.ngenes, dtype = np.uint32)
            als_ub = classifier.get_als_genes_upper_bound()
            upper_bound = np.array(als_ub, dtype = np.uint32)
            Optimizer.MOP.__init__(self, classifier, dataset_csv, threads, emax)
            ElementwiseProblem.__init__(self, n_var=self.ngenes, n_obj=2, n_constr=1, xl=lower_bound, xu=upper_bound)

        def __genotype_to_phenotype(self, X):
            configurations = []
            count = 0
            for size in self.als_genes_per_tree:
                configurations.append([X[i+count] for i in range(size)])
                count += size
            for item in self._partitions:
                item[0].set_assertions_configuration(configurations)

        def _evaluate(self, X, out, *args, **kwargs):
            self.__genotype_to_phenotype(X)
            err = self._get_accuracy_loss()
            gates = sum(self._partitions[0][0].get_current_required_aig_nodes())
            out["F"] = [err, gates]
            out["G"] = err - self.emax

    class Full(MOP, ElementwiseProblem):
        def __init__(self, classifier, dataset_csv, threads, emax):
            super().__init__(classifier, dataset_csv, threads)
            self.total_bits = classifier.get_total_bits()
            self.als_genes_per_tree = classifier.get_als_genes_per_tree()
            n_features = len(classifier.get_features())
            self.ngenes =  n_features + sum(self.als_genes_per_tree)
            print(f"Features: {n_features}, genes per tree: {self.als_genes_per_tree}, Tot. #genes: {self.ngenes}")
            gates_list = classifier.get_current_required_aig_nodes()
            self.total_gates = sum(gates_list)
            print(f"Baseline requirements: {self.total_bits} bits, {gates_list} {self.total_gates} gates")
            lower_bound = np.zeros(self.ngenes, dtype = np.uint32)
            als_ub = [53] * len(classifier.get_features()) + classifier.get_als_genes_upper_bound()
            upper_bound = np.array(als_ub, dtype = np.uint32)
            Optimizer.MOP.__init__(self, classifier, dataset_csv, threads, emax)
            ElementwiseProblem.__init__(n_var = self.ngenes, n_obj = 3, n_constr = 1, xl = lower_bound, xu = upper_bound)

        def __genotype_to_phenotype(self, X):
            features = self._partitions[0][0].get_features()
            configurations = []
            count = 0
            for size in self.als_genes_per_tree:
                configurations.append([X[i + count + len(features)] for i in range(size)])
                count += size
            for item in self._partitions:
                item[0].set_nabs([ {"name": f["name"], "nab" : x} for f, x in zip(features, X[:len(features)]) ])
                item[0].set_assertions_configuration(configurations)

        def _evaluate(self, X, out, *args, **kwargs):
            self.__genotype_to_phenotype(X)
            err = self._get_accuracy_loss()
            bits = self._partitions[0][0].get_total_retained()
            gates = sum(self._partitions[0][0].get_current_required_aig_nodes())
            out["F"] = [err, bits, gates]
            out["G"] = err - self.emax

    def __init__(self, axtechnique, classifier, test_dataset, n_threads, nsgaii_pop_size, nsgaii_iter, nsgaii_emax, nsgaii_cross_prob, nsgaii_cross_eta, nsgaii_mut_prob, nsgaii_mut_eta):
        self.__axtechnique = axtechnique
        self.__nsgaii_emax  = nsgaii_emax
        if axtechnique == Optimizer.AxTechnique.ALS:
            self.problem = Optimizer.ALSOnly(classifier, test_dataset, n_threads, nsgaii_emax)
        elif axtechnique == Optimizer.AxTechnique.PS:
            self.problem = Optimizer.PSOnly(classifier, test_dataset, n_threads, nsgaii_emax)
        elif axtechnique == Optimizer.AxTechnique.FULL:
            self.problem = Optimizer.Full(classifier, test_dataset, n_threads, nsgaii_emax)
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
        print("n_gen:         the current number of generations or iterations until this point.")
        print("n_eval:        the number of function evaluations so far.")
        print("n_nds:         the number of non-dominated solutions of the optima found.")
        print("cv (min/avg):  minimum/average constraint violation in the current population")
        print("eps/indicator: the change of the indicator (ideal, nadir, f) over the last few generations.")
        start_time = time.time()
        self.result = minimize(self.problem, self.algorithm, self.termination, verbose = True)
        duration = time.time() - start_time
        print(f"Took {duration} sec.")
        return duration

    def print_pareto(self):
        if self.__axtechnique == Optimizer.AxTechnique.ALS:
            print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #gates {self.problem.total_gates}")
        elif self.__axtechnique == Optimizer.AxTechnique.PS:
            print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #bits {self.problem.total_bits}")
        elif self.__axtechnique == Optimizer.AxTechnique.FULL:
            print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #gates {self.problem.total_gates}, #bits {self.problem.total_bits}")
        row_format = "{:<16}" * (len(self.result.pop.get("F")[0])) + "{:>4}" * (len(self.result.pop.get("X")[0]))
        print("Final population:\nError     Cost        Chromosome")
        for fitness, chromosome in zip(self.result.pop.get("F"), self.result.pop.get("X")):
            print(row_format.format(*fitness, *chromosome))

    def plot_pareto(self, pdf_file):
        if self.__axtechnique == Optimizer.AxTechnique.FULL:
            # TODO: implementa con subfigure
            pass
        else:
            F = self.result.pop.get("F")
            plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(F[:,0], F[:,1], 'k.')
            plt.axvline(x = self.__nsgaii_emax, c = 'r')
            plt.xlim([0, 100])
            plt.xticks(list(range(0, 100, 10)) + [self.__nsgaii_emax], list(range(0, 100, 10)) + [plt.Text(0, 0, "$e_{max}$")])
            plt.xlabel("Classification-accuracy loss (%)")
            plt.ylabel("# of AIG gates" if self.__axtechnique == Optimizer.AxTechnique.ALS else "# of retained bits" )
            plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0)

    def get_report(self, report_file):
        original_stdout = sys.stdout
        row_format = "{:};" * (len(self.result.pop.get("F")[0])) + "{:};" * (len(self.result.pop.get("X")[0]))
        with open(report_file, "w") as file:
            sys.stdout = file
            if self.__axtechnique == Optimizer.AxTechnique.ALS:
                print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #gates {self.problem.total_gates}")
            elif self.__axtechnique == Optimizer.AxTechnique.PS:
                print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #bits {self.problem.total_bits}")
            elif self.__axtechnique == Optimizer.AxTechnique.FULL:
                print(f"Baseline accuracy: {self.problem.baseline_accuracy}, #gates {self.problem.total_gates}, #bits {self.problem.total_bits}")
            print("Final population:\nError;Cost;Chromosome")
            for fitness, chromosome in zip(self.result.pop.get("F"), self.result.pop.get("X")):
                print(row_format.format(*fitness, *chromosome))
        sys.stdout = original_stdout

    def get_individuals(self):
        return self.result.pop.get("X")

    def get_elapsed_time(self):
        return self.result.exec_time

def evaluate_preloaded_dataset(classifier, samples):
    return classifier.evaluate_preloaded_dataset(samples)

