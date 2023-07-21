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
from .SecondStepWcBaseMop import *
import numpy as np
class SecondStepAlsWcMop(SecondStepWcBaseMop, pyamosa.Problem):

    def __init__(self, classifier, error_config, fst_opt_conf, fst_opt_term_criterion, outdir):
        SecondStepWcBaseMop.__init__(self, classifier, error_config, fst_opt_conf, fst_opt_term_criterion, outdir)
        n_vars = self.classifier.get_num_of_trees()
        ub = [len(self.first_step_optimizer.pareto_set())] * n_vars
        print(f"Second step optimization (WC) #vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        pareto_set = self.first_step_optimizer.pareto_set()
        configurations = [ pareto_set[c] for c in x ]
        for item in self.args:
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.error_conf.max_loss_perc]