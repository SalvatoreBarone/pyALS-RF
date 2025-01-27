
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
import csv 
import os
import json5
import pyamosa
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re


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

    """ Initialize the MOO problem. Optionally, if mr_axc is not none, then the current mr_axc is overwritten. """
    def initialize_problem(self, mr_axc: MrAxC = None):
        if mr_axc != None:
            self.mr_axc = mr_axc
        """ In our approach variables are t_(i,j). Such variables indicate whether a  tree i is used for the classification of class j."""
        n_vars = len(self.mr_axc.classifier.trees) * len(self.mr_axc.classifier.model_classes)
        """ 
        Init sign: 
        (self, num_of_variables : int , types : list, lower_bounds : list, upper_bounds : list, num_of_objectives : int, num_of_constraints : int):
            num_of_variables :  the number of variables in our problem is equal to the number of tree * number of classes.
            types:              Different variable types for each class. For us they're 0 or 1.
            lower_bounds:       The lower bound values for each variable. For us, as they're binary variables, they are 0.
            upper_bounds:       The upper bound values for each variable. For us, as they're binary variables, they are 1. 
            num_of_objectives : The number of different objectives considered during optimization. For us they're two different objectives.
            constraints:        Different constraints, for us there is only one, i.e. the maximum accuracy loss.         
        """
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.INTEGER] * n_vars, [0] * n_vars, [2] * n_vars, 2, 1)
    
    """ Given a solution, represented in terms of 0 and 1, return the configuration variable.
        This function returns for each class the set of trees that classify that specific class.
    """
    @staticmethod
    def get_tree_cfg(mr_axc, x):
        n_trees = len(mr_axc.classifier.trees)
        per_class_cfg = []
        # Transform the x solution into a new configuration. 
        for i in range(0, len(mr_axc.classifier.model_classes)):
            # Consider the Tree.
            class_cfg = [ tree_idx  for tree_idx, t_bin in enumerate(x[i * n_trees : ( i + 1) * n_trees ]) if t_bin == 1 ]
            per_class_cfg.append(class_cfg)
        return per_class_cfg

    # """ Given a solution, where for each class the set of trees is listed ( set of trees Per Class), transform the solution into che 
    #     set of classes per tree.
    # """
    # @staticmethod
    # def cfg_per_class_in_cfg_per_tree(mr_axc, trees_per_class_cfg):
    #     n_trees = len(mr_axc.classifier.trees)
    #     per_tree_cfg = []
    #     for tree in range(0, n_trees):
    #         tree_classes = []
    #         for considered_class, class_cfg in enumerate(trees_per_class_cfg): # It is a list.
    #             if tree in class_cfg:
    #                 tree_classes.append(considered_class)
    #         per_tree_cfg.append(tree_classes)
    #     return per_tree_cfg

    def evaluate(self, x, out):
    
        cfg_under_eval = MrMop.get_tree_cfg(self.mr_axc, x)
        # Get the current accuracy.
        accuracies = self.mr_axc.evaluate_mr_cfg_xmop(cfg_under_eval)
        # print(f"Accuracies {accuracies}")
        # Get the current cost
        current_cost = self.mr_axc.evaluate_mr_cfg_cost(cfg_under_eval)
        # Evaluate the accuracy loss
        acc_loss = self.mr_axc.x_mop_baseline_accuracy - accuracies[0] # The evaluated accuracy depends on whether the draw conditions are considered or not.
        # F should refer to the two objectives while g to the constraints
        out["f"] = [float(acc_loss), int(current_cost)]
        out["g"] = [float(acc_loss - self.max_loss)]

