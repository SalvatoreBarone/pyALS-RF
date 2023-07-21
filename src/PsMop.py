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
import numpy as np, pyamosa
from .Classifier import *
from .rank_based import datasetRanking, estimateLoss

class PsMop(pyamosa.Problem):
    def __init__(self, classifier, max_loss, ncpus):
        self.classifier = classifier
        self.max_loss = max_loss
        self.ncpus = ncpus
        self.classifier.reset_nabs_configuration()
        self.classifier.reset_assertion_configuration()
        
        
        self.baseline_accuracy = self.classifier.evaluate_test_dataset()
        print(f"Baseline accuracy: {self.baseline_accuracy} %")
        print(f"Baseline retained bits: {self.classifier.get_total_retained()}")
        n_vars = len(self.classifier.model_features)
        ub = [53] * n_vars
        print(f"#vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def set_matter_configuration(self, x):
        nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, x[:len(self.classifier.model_features)])}
        self.classifier.set_nabs(nabs)

    def evaluate(self, x, out):
        self.set_matter_configuration(x)
        acc_loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset()
        retained_bits = self.classifier.get_total_retained()
        out["f"] = [acc_loss, retained_bits]
        out["g"] = [acc_loss - self.max_loss]
        
class RankBasedPsMop(pyamosa.Problem):
    def __init__(self, classifier, max_loss, alpha, beta, gamma, ncpus):
        self.classifier = classifier
        self.max_loss = max_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ncpus = ncpus
        self.classifier.reset_nabs_configuration()
        self.classifier.reset_assertion_configuration()
        self.C, self.M = datasetRanking(self.classifier)
        self.sample_count = []
        self.baseline_accuracy = len(self.C) / (len(self.C) + len(self.M)) * 100
        print(f"Baseline accuracy: {self.baseline_accuracy} %")
        print(f"Baseline retained bits: {self.classifier.get_total_retained()}")
        n_vars = len(self.classifier.model_features)
        ub = [53] * n_vars
        print(f"#vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)
        
    def set_matter_configuration(self, x):
        nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, x[:len(self.classifier.model_features)])}
        self.classifier.set_nabs(nabs)

    def evaluate(self, x, out):
        self.set_matter_configuration(x)
        acc_loss, samples = estimateLoss(self.baseline_accuracy, self.max_loss, self.alpha, self.beta, self.gamma, self.classifier, self.C, self.M)
        self.sample_count.append(samples)
        retained_bits = self.classifier.get_total_retained()
        out["f"] = [acc_loss, retained_bits]
        out["g"] = [acc_loss - self.max_loss]
        
    def archived_actual_accuracy(self, archive):
        for solution in tqdm(archive, total=len(archive), desc="Evaluating actual accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False):
            self.set_matter_configuration(solution["x"])
            loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset()
            solution["f"][0] = loss
            solution["g"][0] = loss - self.max_loss
        return pyamosa.Optimizer.remove_dominated(archive)
        
    