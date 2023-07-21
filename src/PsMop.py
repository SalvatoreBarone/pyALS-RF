"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

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
import numpy as np, pyamosa

class PsMop(BaseMop, pyamosa.Problem):
    def __init__(self, classifier, error_conf, ncpus):
        self.error_conf = error_conf
        BaseMop.__init__(self, classifier, self.error_conf.test_dataset, ncpus)
        n_vars = len(classifier.get_features())
        ub = [53] * n_vars
        print(f"#vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        nabs = {f["name"]: n for f, n in zip(self.features, x[:len(self.features)])}
        for item in self.args:
            item[0].set_nabs(nabs)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss_noals()
        f2 = self.args[0][0].get_total_retained()
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.error_conf.max_loss_perc]