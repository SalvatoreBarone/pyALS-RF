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
import time, sys
from multiprocessing import cpu_count, Pool
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from .Configs import *
from .DecisionTree import DecisionTree
from .ALSGraph import *
from .ALSCatalog import *

class FirsStageOptimizer:
    class MOP(ElementwiseProblem):
        def __init__(self, tree, samples, max_error):
            self.tree = tree
            tree.reset_assertion_configuration()
            self.ngenes = len(self.tree.get_graph().get_cells())
            self.trees = [copy.deepcopy(self.tree)] * cpu_count()
            self.max_error = max_error
            self.total_samples = len(samples)

            self.duration = time.time()
            args = [[c, d] for c, d in zip(self.trees, list_partitioning(samples, cpu_count()))]
            with Pool(cpu_count()) as pool:
                exact_output = pool.starmap(evaluate_exact_output, args)
            self.duration = (time.time() - self.duration)

            self.args = [[c, d] for c, d in zip(self.trees, exact_output)]

            lower_bound = np.zeros(self.ngenes, dtype=np.uint32)
            cells = [{"name": c["name"], "spec": c["spec"]} for c in self.tree.get_graph().get_cells()]
            catalog = self.tree.get_catalog_for_assertions()
            upper_bound = [len(e) - 1 for c in cells for e in catalog if e[0]["spec"] == c["spec"]]
            ElementwiseProblem.__init__(self, n_var=self.ngenes, n_obj=2, n_constr=1, xl=lower_bound, xu=upper_bound)

        def __genotype_to_phenotype(self, x):
            self.tree.set_assertions_configuration(x)
            for t in self.trees:
                t.set_assertions_configuration(x)

        def _evaluate(self, X, out, *args, **kwargs):
            self.__genotype_to_phenotype(X)
            with Pool(cpu_count()) as pool:
                res = pool.starmap(evaluate_error_frequency, self.args)
            err = sum(res) * 100 / self.total_samples
            gates = self.tree.get_current_required_aig_nodes()
            out["F"] = [err, gates]
            out["G"] = err - self.max_error

    def __init__(self, tree, samples, nsgaii_conf):
        self.nsgaii_conf = nsgaii_conf
        self.problem = FirsStageOptimizer.MOP(tree, samples, nsgaii_conf.max_error)
        pop_size = nsgaii_conf.pop_size if nsgaii_conf.pop_size > 0 else 10 * self.problem.ngenes
        mut_p = nsgaii_conf.mut_p if nsgaii_conf.mut_p > 0 else 1 / self.problem.ngenes
        self.algorithm = NSGA2(
            pop_size = pop_size,
            n_offsprings = None,
            sampling = get_sampling("int_random"),
            crossover = get_crossover("int_sbx", prob = nsgaii_conf.cross_p, eta = nsgaii_conf.cross_eta),
            mutation = get_mutation("int_pm", prob = mut_p, eta = nsgaii_conf.mut_eta),
            eliminate_duplicates=True)
        self.termination = get_termination('n_gen', nsgaii_conf.iterations)
        self.result = None
        eta_secs = 2 * pop_size * nsgaii_conf.iterations * self.problem.duration
        self.__eta_hours = int(eta_secs / 3600)
        self.__eta_min = int((eta_secs - self.__eta_hours * 3600) / 60)

    def optimize(self):
        print(f"\n\nOptimizing tree {self.problem.tree.get_name()}.")
        print(f"Performing NSGA-II using {cpu_count()} threads.")
        print(f"#genes: {self.problem.ngenes}. Pop: {self.nsgaii_conf.pop_size if self.nsgaii_conf.pop_size > 0 else 10 * self.problem.ngenes}, Iter: {self.nsgaii_conf.iterations}, Pcross: {self.nsgaii_conf.cross_p}, Ecross: {self.nsgaii_conf.cross_eta}, Pmut: {self.nsgaii_conf.mut_p if self.nsgaii_conf.mut_p > 0 else 1 / self.problem.ngenes}, Emut: {self.nsgaii_conf.mut_eta}")
        print(f"Please wait patiently. This may take quite a long time (ETA: {self.__eta_hours} h, {self.__eta_min} min.)")
        print("\nReported infos:")
        print("n_gen:         the current number of generations or iterations until this point.")
        print("n_eval:        the number of function evaluations so far.")
        print("n_nds:         the number of non-dominated solutions of the optima found.")
        print("cv (min/avg):  minimum/average constraint violation in the current population")
        print("eps/indicator: the change of the convergence indicator (ideal, nadir, f) over the last few generations.\n")
        self.result = minimize(self.problem, self.algorithm, self.termination, verbose = True)

    def plot_pareto(self, pdf_file):
        F = self.result.F
        plt.figure(figsize=(10, 10), dpi=300)
        plt.plot(F[:,0], F[:,1], 'k.')
        plt.axvline(x = self.problem.max_error, c = 'r')
        x_min = int(min(F[:,0])) - 10
        x_min = x_min if x_min < 0 else 0
        x_max = int(max(F[:,0])) + 10
        x_max = x_max if x_max < 100 else 100
        plt.xlim([ x_min, x_max ])
        plt.xticks(list(range(x_min, x_max, 10)) + [self.problem.max_error], list(range(x_min, x_max, 10)) + [plt.Text(0, 0, "$e_{max}$")])
        plt.xlabel("Error-frequency (%)")
        plt.ylabel("# of AIG gates")
        plt.savefig(pdf_file, bbox_inches='tight', pad_inches=0)

    def get_report(self, report_file):
        original_stdout = sys.stdout
        row_format = "{:};" * (len(self.result.F[0])) + "{:};" * (len(self.result.X[0]))
        with open(report_file, "w") as file:
            sys.stdout = file
            print("Final population:\nError;Cost;Chromosome")
            for fitness, chromosome in zip(self.result.F, self.result.X):
                print(row_format.format(*fitness, *chromosome))
        sys.stdout = original_stdout

    def get_individuals(self):
        return self.result.X

    def get_elapsed_time(self):
        return self.result.exec_time

def evaluate_exact_output(tree, samples):
    exact_output = []
    for s in samples:
        outcomes = [{"name": c, "score": 0} for c in tree.get_model_classes()]
        tree.evaluate(s["input"], outcomes)
        exact_output.append({"input": s["input"], "output": outcomes})
    return exact_output

def evaluate_error_frequency(tree, samples):
    error = 0
    for sample in samples:
        outcomes = [{"name": c, "score": 0} for c in tree.get_model_classes()]
        tree.evaluate(sample["input"], outcomes)
        error += 0 if sample["output"] == outcomes else 1
    return error
