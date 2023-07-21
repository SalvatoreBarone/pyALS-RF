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

class PsMop(pyamosa.Problem):
    def __init__(self, classifier, error_conf, ncpus):
        self.classifier = classifier
        self.error_conf = error_conf
        self.ncpus = ncpus
        self.classifier.reset_nabs_configuration()
        self.classifier.reset_assertion_configuration()
        self.classifier.read_dataset(self.error_conf.test_dataset, self.error_conf.dataset_description)
        self.baseline_accuracy = self.classifier.evaluate_test_dataset()
        print(f"Baseline accuracy: {self.baseline_accuracy} %")
        
        n_vars = len(self.classifier.model_features)
        ub = [53] * n_vars
        print(f"#vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, x[:len(self.classifier.model_features)])}
        self.classifier.set_nabs(nabs)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        acc_loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset()
        retained_bits = self.classifier.get_total_retained()
        out["f"] = [acc_loss, retained_bits]
        out["g"] = [acc_loss - self.error_conf.max_loss_perc]