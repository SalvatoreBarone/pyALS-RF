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

class SingleStepCombined(BaseMop, Optimizer.Problem):
    def __init__(self, classifier, dataset_csv):
        BaseMop.__init__(self, classifier, dataset_csv)
        self.cells_per_tree = classifier.get_als_cells_per_tree()
        n_features = len(self.features)
        n_cells = sum(self.cells_per_tree)
        n_vars = n_features + n_cells
        Optimizer.Problem.__init__(self, n_vars, [Optimizer.Type.INTEGER] * n_vars, [0] * n_vars,  [53] *  n_features + classifier.get_als_dv_upper_bound(), 3, 1)

    def __set_matter_configuration(self, x):
        configurations = []
        count = 0
        for size in self.cells_per_tree:
            configurations.append([x[i + count + len(self.features)] for i in range(size)])
            count += size
        nabs = {f["name"]: n for f, n in zip(self.features, x[:len(self.features)])}
        for item in self.args:
            item[0].set_nabs(nabs)
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.error_conf.threshold]