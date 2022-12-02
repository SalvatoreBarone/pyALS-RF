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
from .SecondStepBaseMop import *


class SecondStepCombinedMop(SecondStepBaseMop, Optimizer.Problem):
    def __init__(self, classifier, dataset_csv, config, improve, outdir):
        SecondStepBaseMop.__init__(self, classifier, dataset_csv, config, improve, outdir)
        n_vars = len(self.features) + self.classifier.get_num_of_trees()
        ub = [53] * len(self.features) + [ len(i)-1 for i in self.opt_solutions_for_trees ]
        print(f"Baseline accuracy: {self.baseline_accuracy}.")
        print(f"d.v. #{len(ub)}, {ub}")
        Optimizer.Problem.__init__(self, n_vars, [Optimizer.Type.INTEGER] * n_vars, [0] * n_vars, ub, 3, 1)

    def __set_matter_configuration(self, x):
        nabs = { f["name"] : n for f, n in zip(self.features, x[:len(self.features)]) }
        configurations = [ s[c] for s, c in zip(self.opt_solutions_for_trees, x[len(self.features):])  ]
        for item in self.args:
            item[0].set_nabs(nabs)
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.snd_error_conf.threshold]
