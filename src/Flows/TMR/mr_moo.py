
"""
Copyright 2021-2025 Antonio Emmanuele <antonio.emmanuele@unina.it>
                    Salvatore Barone <salvatore.barone@unina.it>
                    
This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging, joblib, numpy as np
from multiprocessing import cpu_count, Pool
from itertools import combinations, product
from tqdm import tqdm
from ...Model.Classifier import Classifier
from ...Model.DecisionTree import *
from ..GREP.GREP import GREP
from .mr_axc import MrAxC
import time
import csv 
import os
import json5
import pyamosa
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re
import time

class MrMop(pyamosa.Problem):
    
    def __init__(self, mr_axc : MrAxC, max_loss : float, ncpus : int):
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info(f"[MR-MOOP] Initializing MR-MOOP problem")
        self.mr_axc = mr_axc
        self.max_loss = max_loss
        self.ncpus = ncpus
        # Initialize the problem.
    #             ub = [53] * n_vars
    #     logger.info(f"#vars: {n_vars}, ub:{ub}, #conf.s {np.prod([ float(x + 1) for x in ub ])}.")
    #     pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    # Define the matter configuration.
    # def set_matter_configuration(self, x):
    #     nabs = {f["name"]: n for f, n in zip(self.classifier.model_features, x[:len(self.classifier.model_features)])}
    #     self.classifier.set_nabs(nabs)

    # Evaluate the function.
    # def evaluate(self, x, out):
    #     self.set_matter_configuration(x)
    #     #acc_loss = self.baseline_accuracy - self.classifier.evaluate_test_dataset()
    #     acc_loss = self.baseline_accuracy_mop - self.classifier.evaluate_accuracy(self.x_mop, self.y_mop, disable_tqdm = True)
    #     retained_bits = self.classifier.get_total_retained()
    #     out["f"] = [acc_loss, retained_bits]
    #     out["g"] = [acc_loss - self.max_loss]