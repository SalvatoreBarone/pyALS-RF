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

class SingleStepAlsWcMop(BaseMop, Optimizer.Problem):
    def __init__(self, classifier, error_config):
        self.error_config = error_config
        BaseMop.__init__(self, classifier, self.error_config.test_dataset)
        self.cells_per_tree = classifier.get_als_cells_per_tree()
        # TODO check whether all the trees have the same amount of cells
        for i in range(1, len(self.cells_per_tree)):
            assert len(self.cells_per_tree[0]) == len(self.cells_per_tree[i]), "Tree {i} has a different amount of cells"
        n_vars = self.cells_per_tree[0]
        # TODO pick only the actual subset of upper bounds
        print(n_vars, ub)
        ub = classifier.get_als_dv_upper_bound()
        ub = ub[:n_vars-1]
        print(ub)
        exit()
        
        
        print(f"{len(ub)} d.v.: {ub}")
        Optimizer.Problem.__init__(self, n_vars, [Optimizer.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        configurations = []
        count = 0
        # TODO set matter configuration
        for size in self.cells_per_tree:
            configurations.append([x[i + count] for i in range(size)])
            count += size
        for item in self.args:
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.error_config.max_loss_perc]
