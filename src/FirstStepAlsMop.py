"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

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
from .BaseMop import *
from pyamosa import Optimizer


class FirstStepAlsMop(Optimizer.Problem):
    def __init__(self, decision_tree, preloaded_dataset, error_config):
        self.decision_tree = decision_tree
        graph = self.decision_tree.get_graph()
        n_vars = graph.get_num_cells()
        ub = self.decision_tree.get_als_dv_upper_bound()

        self.error_config = error_config
        self.decision_tree.reset_assertion_configuration()
        self.samples = self.generate_samples(graph, preloaded_dataset)
        self.total_samples = len(self.samples)
        self.args = [[g, s, [0] * n_vars] for g, s in zip([copy.deepcopy(graph)] * cpu_count(), list_partitioning(self.samples, cpu_count()))]

        print(f"Tree {self.decision_tree.get_name()}. d.v. #{len(ub)}: {ub}")
        Optimizer.Problem.__init__(self, n_vars, [Optimizer.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def generate_samples(self, graph, preloaded_dataset):
        samples = []
        for sample in preloaded_dataset:
            inputs = self.decision_tree.get_boxes_output(sample["input"])
            samples.append({"input": inputs, "output": graph.evaluate(inputs)})
        return samples

    def __set_matter_configuration(self, x):
        self.decision_tree.set_assertions_configuration(x)
        for a in self.args:
            a[2] = self.decision_tree.get_assertions_configuration()

    def __get_eprob(self):
        with Pool(cpu_count()) as pool:
            error = pool.starmap(evaluate_eprob, self.args)
        return sum(error) * 100 / len(self.samples)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.__get_eprob()
        f2 = self.decision_tree.get_current_required_aig_nodes()
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.error_config.threshold]