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
import pandas as pd
class MrHeu:

    def __init__(self, mr_order: int = 3, ncpus : int = os.cpu_count()): 
        assert mr_order >= 3, "[MR-HEU] Provide a Modular Redundancy order >= 3"
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info("[MR-HEU] Inizializing Modular Redundancy Heuristic")
        self.mr_order = mr_order
        self.n_cpus = ncpus
        self.is_problem_initialized = False
        self.is_pruining_outdir_initialized = False
        self.is_csv_out_initialized = False
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
    
    """  
        Initialize all the output cfg paths related to the pruning configuration.
        PRUNING CONFIGURATION FILES ARE DIFFERENT FROM OUTDIR FOR STATUS (I.E. THE CSV)
    """
    def initialize_pruning_cfg_out(self, outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.approx_cfg_outdir = outdir
        self.pruning_outfiles_dict = {
            "outfile_per_class_cfg" : os.path.join(outdir, "per_class_cfg.json5"),
            "outfile_per_tree_cfg"  : os.path.join(outdir, "per_tree_cfg.json5"),
            "outfile_leaves_idx"    : os.path.join(outdir, "leaves_idx.json5"),
            "outfile_pruning_cfg"   : os.path.join(outdir, "pruning_conf.json5"),
            "outfile_directions"    : os.path.join(outdir, "leaf_pruning_directions.json5"),
            "considered_classes"    : os.path.join(outdir, "considered_classes.json5") # Each configuration refers to the ordered set of the considered classes
        }
        self.is_pruining_outdir_initialized = True

    def initialize_summary_files(self, outdir):
        if not os.path.exists(outdir):
             os.makedirs(outdir)
        self.csv_outfile = os.path.join(outdir, "mr_report.csv")
        self.is_csv_out_initialized = True

    def heu_tree_acc(self):
        assert self.is_problem_initialized, "[MR-HEU] You should first initialize the problem! "
        assert self.is_pruining_outdir_initialized, "[MR-HEU] You should first initialize the pruning out dir!"
        assert self.is_csv_out_initialized, "[MR-HEU] You should first initialize the CSV outfile!"
        self.logger.info("[MR-HEU] Starting heuristic Accuracy based. This may take a while, but be patient !")
        tm = time.time()
        # Get the classes per each tree
        pertree_classes = self.mr_axc.classifier.transform_leaves_into_classess(self.mr_axc.x_mop_leaves)
        # Now get the accuracy for each class
        perclass_accs = self.mr_axc.pertree_classess_into_perclass_pertree_acc(pertree_classes, self.mr_axc.y_mop)
        
        # for class_original, accuracies in zip(self.mr_axc.sampled_classes, perclass_accs):
        #     self.logger.info(f"Class {class_original} : {accuracies}")
        # print(f"Sampled classes: {self.mr_axc.sampled_classes}")
        # exit(1)
        
        # Sort the class indexes 
        # Each configuration consists in the first mr_order treees.
        mr_cfg = [list(np.argsort(c_accs)[::-1])[:self.mr_order] for c_accs in perclass_accs]  
        mr_cfg = [[int(m) for m in cfg] for cfg in mr_cfg] # Convert numpy.int64 in int
        tm = time.time() - tm
        self.logger.info("[MR-HEU] Accuracy based heuristic completed !")
        
        # Getting XMOP accuracy values.
        self.logger.info("[MR-HEU] Initiating evaluation on XAxC Set")
        validation_leaves = self.mr_axc.classifier.compute_leaves_idx(self.mr_axc.x_val, False)
        validation_classes = self.mr_axc.classifier.transform_leaves_into_classess(validation_leaves)
        validation_classes = MrAxC.per_tree_classess_into_classes_per_tree(validation_classes)
        xaxc_mr_pred_vectors =  self.mr_axc.get_mr_vectors(validation_classes, mr_cfg)
        heu_acc_draw, heu_acc_no_draw = self.mr_axc.get_accuracy_from_vectors(xaxc_mr_pred_vectors, self.mr_axc.y_val)
        heu_loss_draw = self.mr_axc.x_mop_baseline_accuracy - heu_acc_draw
        heu_loss_no_draw = self.mr_axc.x_mop_baseline_accuracy_nodraw - heu_acc_no_draw
        self.logger.info(f"[MR-HEU] XAxC-Set Evaluation completed! Baseline: {self.mr_axc.x_mop_baseline_accuracy}")
        self.logger.info(f"[MR-HEU] XAxC-Set Draw considered as missclassifications Acc. : {heu_acc_draw}, Loss: {heu_loss_draw}")
        self.logger.info(f"[MR-HEU] XAxC-Set Draw NOT considered as missclassification Acc. : {heu_acc_no_draw}, Loss: {heu_loss_no_draw}")
        
        

        # Get the leaves of the validation set.
        # Get the validation leaves
        self.logger.info("[MR-HEU] Initiating evaluation on Validation Set")
        validation_leaves = self.mr_axc.classifier.compute_leaves_idx(self.mr_axc.x_val, False)
        validation_classes = self.mr_axc.classifier.transform_leaves_into_classess(validation_leaves)
        validation_classes = MrAxC.per_tree_classess_into_classes_per_tree(validation_classes)
        mr_pred_vectors =  self.mr_axc.get_mr_vectors(validation_classes, mr_cfg)
        val_acc_draw, val_acc_no_draw = self.mr_axc.get_accuracy_from_vectors(mr_pred_vectors, self.mr_axc.y_val)
        loss_draw = self.mr_axc.x_val_baseline_accuracy - val_acc_draw
        loss_no_draw = self.mr_axc.x_val_baseline_accuracy_nodraw - val_acc_no_draw
        self.logger.info(f"[MR-HEU] Validation-Set Evaluation completed! : Baseline: {self.mr_axc.x_val_baseline_accuracy}")
        self.logger.info(f"[MR-HEU] Validation-Set Draw considered as missclassifications Acc. : {val_acc_draw}, Loss: {loss_draw}")
        self.logger.info(f"[MR-HEU] Validation-Set Draw NOT considered as missclassification  Acc. : {val_acc_no_draw}, Loss: {loss_no_draw}")
        
        # Dump vectors and preds.
        self.mr_axc.dump_mop_val_indexes(self.approx_cfg_outdir)
        np.savetxt(os.path.join(self.approx_cfg_outdir, "xaxc_pred_vectors.txt"), xaxc_mr_pred_vectors, fmt = "%d")
        np.savetxt(os.path.join(self.approx_cfg_outdir, "val_pred_vectors.txt"), mr_pred_vectors, fmt = "%d")

        # Save the pruning cfg.
        self.logger.info(f"[MR-HEU] Dumping configuration / pruning / direction files ....")
        _, _, pruning_cfg, _ = self.mr_axc.dump_cfg(self.pruning_outfiles_dict, mr_cfg)
        self.logger.info(f"[MR-HEU] Dump of  configuration / pruning / direction files COMPLETED")
        
        # Update stats.
        sol_summary = {
                "Algo"                   : "pertree_acc_heu",
                "MrOrder"                : self.mr_order,
                "Pruned-Leaves"         : len(pruning_cfg),

                "Baseline_XMOP_Acc"     : self.mr_axc.x_mop_baseline_accuracy,
                "Acc-XAxC_Draw"         : heu_acc_draw,     # First insert without considering the draw condition, then append.
                "Loss-XAxC_Draw"        : heu_loss_draw,
                "Acc-XAxC_NO_Draw"         : heu_acc_no_draw,
                "Loss-XAxC_NO_Draw"        : heu_loss_no_draw,

                "Baseline_XVal_Acc."    : self.mr_axc.x_val_baseline_accuracy,
                "Acc-XVal_Draw"         : val_acc_draw,
                "Loss-XVal_Draw"        : loss_draw,
                "Acc-XVal_NO_Draw"      : val_acc_no_draw,
                "Loss-XVal_NO_Draw"     : loss_no_draw,
                "Comp Time [s]"         : tm
            }
        add_header = not os.path.exists(self.csv_outfile)
        df = pd.DataFrame(sol_summary, index=[0]).to_csv(self.csv_outfile, index = False, header = add_header, mode = "a")
        self.logger.info(f"Summary CSV updated! Please check {self.csv_outfile}")
