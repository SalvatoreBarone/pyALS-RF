
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
from scipy.stats import norm # For cut-offs.
from sklearn.model_selection import train_test_split
import re
import time

class MR_HEU:

    def __init__(self, mr_order: int = 3, ncpus : int = os.cpu_count()): 
        assert mr_order >= 3, "[MR-HEU] Provide a Modular Redundancy order >= 3"
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info("[MR-HEU] Inizializing Modular Redundancy Heuristic")
        self.mr_order = 3
        self.n_cpus = ncpus
        self.is_problem_initialized = False
        self.logger.info("[MR-HEU] Initialization of MR-HEU completed !")
    
    
    """ Initialize the MOO problem. Optionally, if mr_axc is not none, then the current mr_axc is overwritten. """
    def initialize_problem(self, mr_axc: MrAxC = None):
        self.logger.info("[MR-HEU] Initializing MR-HEU problem")
        if mr_axc != None:
            self.mr_axc = mr_axc
        assert self.mr_axc != None, "[MR-HEU] Provide a valid MrAxc approximation class"
        self.logger.info("[MR-HEU] Extracting infos from the dataset...")
        self.is_problem_initialized = True
        self.logger.info("[MR-HEU] Initialization of MR-HEU completed")
        
    def heu_tree_acc(self):
        assert self.is_problem_initialized, "[MR-HEU] You should first initialize the problem! "
        self.logger.info("[MR-HEU] Starting heuristic Accuracy based. This may take a while, but be patient !")
        time = os.times()
        # Get the classes per each tree
        pertree_classes = self.mr_axc.classifier.transform_leaves_into_classess(self.x_heu_leaves)
        # Now get the accuracy for each class
        perclass_accs = self.mr_axc.pertree_classess_into_perclass_pertree_acc(pertree_classes)
        # Sort the class indexes 
        # Each configuration consists in the first mr_order treees.
        mr_cfg = [list(np.argsort(c_accs)[::-1])[:self.mr_order] for c_accs in perclass_accs]  
        self.logger.info("[MR-HEU] Accuracy based heuristic completed !")
        
        # Get the leaves of the validation set.
        # Get the validation leaves
        self.logger.info("Initiating evaluation on Validation Set")
        validation_leaves = self.mr_axc.classifier.compute_leaves_idx(self.x_val, False)
        validation_classes = self.mr_axc.classifier.transform_leaves_into_classess(validation_leaves)
        validation_classes = MrAxC.per_tree_classess_into_classes_per_tree(validation_classes)
        mr_pred_vectors =  MrAxC.get_mr_vectors(validation_classes, mr_cfg)
        val_acc_draw, val_acc_no_draw = MrAxC.get_accuracy_from_vectors(mr_pred_vectors, self.y_heu)
        loss_draw = self.mr_axc.x_val_baseline_accuracy - val_acc_draw
        loss_no_draw = self.mr_axc.x_val_baseline_accuracy - val_acc_no_draw
        self.logger.info(f"Evaluation completed! : Baseline: {self.mr_axc.x_val_baseline_accuracy}")
        self.logger.info(f"Draw considered as missclassifications Acc. : {val_acc_draw}, Loss: {loss_draw}")
        self.logger.info(f"Draw NOT considered as missclassification Acc. : {val_acc_no_draw}, Loss: {loss_no_draw}")
        
        # Dump vectors and preds.

        # Save the pruning cfg.
        
        # Update stats.