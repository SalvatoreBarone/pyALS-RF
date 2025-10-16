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
# Used for margins.
from ..EnsemblePruning.EnsemblePruner import Pruner
from pyalslib import list_partitioning
class MrHeu:

    def __init__(self, mr_order: int = 3, ncpus : int = os.cpu_count(), method: str = "pertree_acc_heu", excluded_trees = []): 
        assert mr_order >= 3, "[MR-HEU] Provide a Modular Redundancy order >= 3"
        self.logger = logging.getLogger("pyALS-RF")
        self.logger.info("[MR-HEU] Inizializing Modular Redundancy Heuristic")
        self.mr_order = mr_order
        self.n_cpus = ncpus
        self.is_problem_initialized = False
        self.is_pruining_outdir_initialized = False
        self.is_csv_out_initialized = False
        self.pool = Pool(self.n_cpus)
        if method == "pertree_acc_heu":
            self.ranking_procedure = rank_trees_per_accuracy
        elif method == "pertree_margin_heu":
            self.ranking_procedure = rank_trees_per_margin
        else:
            self.logger.error("[MR-HEU] Ranking method not supported, supported : pertree_acc_heu and pertree_margin_heu")
            exit(1)
        self.ranking_procedure_str = method
        self.logger.info("[MR-HEU] Initialization of MR-HEU completed !")
        self.excluded_trees = excluded_trees

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
    
    def tree_ranking(self):
        return self.ranking_procedure(self)


    def heu_tree_acc(self):

        assert self.is_problem_initialized, "[MR-HEU] You should first initialize the problem! "
        assert self.is_pruining_outdir_initialized, "[MR-HEU] You should first initialize the pruning out dir!"
        assert self.is_csv_out_initialized, "[MR-HEU] You should first initialize the CSV outfile!"
        self.logger.info("[MR-HEU] Starting heuristic. This may take a while, but be patient !")
        tm = time.time()
        mr_cfg = self.tree_ranking()
        tm = time.time() - tm
        self.logger.info("[MR-HEU] Heuristic completed !")
        
        # Getting XMOP accuracy values.
        self.logger.info("[MR-HEU] Initiating evaluation on XAxC Set")
        validation_leaves = self.mr_axc.classifier.compute_leaves_idx(self.mr_axc.x_mop, False)


        validation_classes = self.mr_axc.classifier.transform_leaves_into_classess(validation_leaves)
        validation_classes = MrAxC.per_tree_classess_into_classes_per_tree(validation_classes)
        xaxc_mr_pred_vectors =  self.mr_axc.get_mr_vectors(validation_classes, mr_cfg)
        heu_acc_draw, heu_acc_no_draw = self.mr_axc.get_accuracy_from_vectors(xaxc_mr_pred_vectors, self.mr_axc.y_mop)
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
        np.savetxt(os.path.join(self.approx_cfg_outdir, "original_ensemble_labels.txt"), self.mr_axc.x_val_class_labels, fmt = "%d")
        np.savetxt(os.path.join(self.approx_cfg_outdir, "original_ensemble_labels_nodraw.txt"), self.mr_axc.x_val_class_labels_nodraw, fmt = "%d")

        # Save the pruning cfg.
        self.logger.info(f"[MR-HEU] Dumping configuration / pruning / direction files ....")
        _, _, pruning_cfg, _ = self.mr_axc.dump_cfg(self.pruning_outfiles_dict, mr_cfg)
        self.logger.info(f"[MR-HEU] Dump of  configuration / pruning / direction files COMPLETED")
        
        # Update stats.
        sol_summary = {
                "Algo"                   : self.ranking_procedure_str,
                "MrOrder"                : self.mr_order,
                "Pruned-Leaves"         : len(pruning_cfg),

                "Baseline_XMOP_Acc"     : self.mr_axc.x_mop_baseline_accuracy,
                "Acc-XAxC_Draw"         : heu_acc_draw,     # First insert without considering the draw condition, then append.
                "Loss-XAxC_Draw"        : heu_loss_draw,
                "Acc-XAxC_NO_Draw"      : heu_acc_no_draw,
                "Loss-XAxC_NO_Draw"     : heu_loss_no_draw,
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
    
    """ ********* METHODS RELATED TO FULL RANKING PROCEDURES ! """

    """ This code, contains the old heuristic. 
    """
    @staticmethod
    def compute_accuracy_per_tree(p_trees, class_samples, in_lab):
        # Initialize a dictionary mantaining for each tree the number of correctly labeled samples.
        correctly_labeled = {}
        """ NOTE THAT ACTIVITY IS USELESS !"""
        activity = {} 
        # For each tree, visit all the samples and estimate the number of correctly labeled samples.
        for tree_id, tree in p_trees:
            # Initialize the counter
            correctly_labeled[tree_id] = 0
            for sample in class_samples:
                label = int(np.argmax(tree.visit(sample)))
                correctly_labeled[tree_id] += (1 if label == in_lab else 0)
            correctly_labeled[tree_id] = (correctly_labeled[tree_id] / len(class_samples)) * 100
        return correctly_labeled, activity
    
    """ Compute leaves values per tree.
        Returns a dictionary containing for each tree the set of predictions
    """
    @staticmethod
    def compute_leaves_per_tree(p_trees, class_samples, in_lab):
        # Initialize a dictionary mantaining for each tree the number of labeled samples.
        tree_preds = {}
        # For each tree, visit all the samples and estimate the number of correctly labeled samples.
        for tree_id, tree in p_trees:
            # Initialize the counter
            tree_preds[tree_id] = []
            for sample in class_samples:
                label = int(np.argmax(tree.visit(sample)))
                tree_preds[tree_id].append(label)
        return tree_preds

    def do_full_ranking_accuracy(self):
        mr_cfg = []
        pruning_cfg = []
        total_leaves_pruned = 0
        for class_idx in tqdm(self.mr_axc.sampled_classes, desc = "[MR-HEU] Approximating trees per class"):
            self.logger.debug(f"[MR-HEU] Starting heuristic for class {class_idx}.!")
            # Identify the samples belonging to that class.
            class_samples = [self.mr_axc.x_mop[i] for i in range(0, len(self.mr_axc.y_mop)) if self.mr_axc.y_mop[i] == class_idx]
            # Rank the trees for that class, take in mind that this function considers also the excluded trees.
            trees_for_evaluation = [(i, self.mr_axc.classifier.trees[i]) for i in range(0, len(self.mr_axc.classifier.trees)) if i not in self.excluded_trees]
            # This simply performs a multicore, ranking operation !
            p_tree = list_partitioning(trees_for_evaluation, self.n_cpus)
            args = [(tree_sublist, class_samples, class_idx) for tree_sublist in p_tree ]
            sorted_trees = self.pool.starmap(MrHeu.compute_accuracy_per_tree, args)

            # sorted_trees is a list of dictionaries, we have to merge them.
            merged_dict = {}
            #merged_dict_activity = {}
            for t, activity in sorted_trees:
                merged_dict.update(t) 
                #merged_dict_activity.update(activity)
            
            # With this, we sort the list of trees for the parameters.
            sorted_trees = [k for k, v in sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)]
            class_cfg = sorted_trees[:self.mr_order]
            # For each tree not in the class configuration
            for tree in range(len(self.mr_axc.classifier.trees)):
                if tree not in class_cfg:
                    # Get the leaves related to that class 
                    to_prune = GREP.get_pruning_conf_by_class(self.mr_axc.classifier.trees[tree], tree, class_idx)   
                    # Extend the pruning configuration
                    pruning_cfg.extend(to_prune)
                    # Set the pruning configuration.
                    pruned_leaves = GREP.set_pruning(self.mr_axc.classifier.trees[tree], to_prune)
                    total_leaves_pruned += pruned_leaves
            mr_cfg.append(class_cfg)
        return mr_cfg, pruning_cfg, total_leaves_pruned
    
    @staticmethod
    def get_votes_vector_by_tree_leaves_dict(leaves_dict, num_labels, num_samples):
        votes_vectors = [[0 for _ in range(num_labels)] for _ in range(num_samples)]
        for treeLabel in leaves_dict.keys():
            for sampleId, sampleLabel in enumerate(leaves_dict[treeLabel]):
                votes_vectors[sampleId][sampleLabel] += 1
        return votes_vectors
    
    def do_full_ranking_margin(self):
        mr_cfg = []
        pruning_cfg = []
        total_leaves_pruned = 0
        for class_idx in tqdm(self.mr_axc.sampled_classes, desc = "[MR-HEU] Approximating trees per class"):
            self.logger.debug(f"[MR-HEU] Starting heuristic for class {class_idx}.!")
            # Identify the samples belonging to that class.
            class_samples = [self.mr_axc.x_mop[i] for i in range(0, len(self.mr_axc.y_mop)) if self.mr_axc.y_mop[i] == class_idx]
            # Rank the trees for that class, take in mind that this function considers also the excluded trees.
            trees_for_evaluation = [(i, self.mr_axc.classifier.trees[i]) for i in range(0, len(self.mr_axc.classifier.trees)) if i not in self.excluded_trees]
            remaining_trees = [i for i in range(0, len(self.mr_axc.classifier.trees)) if i not in self.excluded_trees]
            # Identify the set of leaves.
            p_tree = list_partitioning(trees_for_evaluation, self.n_cpus)
            args = [(tree_sublist, class_samples, class_idx) for tree_sublist in p_tree ]
            sorted_trees = self.pool.starmap(MrHeu.compute_leaves_per_tree, args)
            # Merge the dictionary, so that we have just one.
            merged_dict = {}
            for t in sorted_trees:
                merged_dict.update(t) 
            # Convert the leaves into a votes vector
            votesVectors = MrHeu.get_votes_vector_by_tree_leaves_dict(leaves_dict=merged_dict, num_labels=len(self.mr_axc.sampled_classes), num_samples=len(class_samples))            
            y_prun = [class_idx for _ in range(len(class_samples))]
            # Estimate the margins.
            per_sample_margins = Pruner.per_sample_margin(votesVectors, y_prun)
            per_sample_margin_gains, new_pv = Pruner.update_margins_dict(tree_preds=merged_dict, pred_vectors=votesVectors, remaining_trees=remaining_trees, yprun=y_prun)

            gains = Pruner.evaluate_mean_dm_dict(per_sample_margins, per_sample_margin_gains)
            # With this, we sort the list of trees for the parameters.
            sorted_trees = [k for k, v in sorted(gains.items(), key=lambda x: x[1], reverse=True)]
            class_cfg = sorted_trees[:self.mr_order]
            # For each tree not in the class configuration
            for tree in range(len(self.mr_axc.classifier.trees)):
                if tree not in class_cfg:
                    # Get the leaves related to that class 
                    to_prune = GREP.get_pruning_conf_by_class(self.mr_axc.classifier.trees[tree], tree, class_idx)   
                    # Extend the pruning configuration
                    pruning_cfg.extend(to_prune)
                    # Set the pruning configuration.
                    pruned_leaves = GREP.set_pruning(self.mr_axc.classifier.trees[tree], to_prune)
                    total_leaves_pruned += pruned_leaves
            mr_cfg.append(class_cfg)
        return mr_cfg, pruning_cfg, total_leaves_pruned

    def set_full_ranking_method(self):
        if self.ranking_procedure_str == "pertree_acc_heu":
            self.full_ranking_method = self.do_full_ranking_accuracy
        elif self.ranking_procedure_str == "pertree_margin_heu":
            self.full_ranking_method = self.do_full_ranking_margin
        else:
            self.logger.error("[MR-HEU] Ranking method not supported, supported : pertree_acc_heu and pertree_margin_heu")
            exit(1)
    
    def do_full_ranking(self):
        return self.full_ranking_method()
    
    def heu_tree_acc_2(self):

        assert self.is_problem_initialized, "[MR-HEU] You should first initialize the problem! "
        assert self.is_pruining_outdir_initialized, "[MR-HEU] You should first initialize the pruning out dir!"
        assert self.is_csv_out_initialized, "[MR-HEU] You should first initialize the CSV outfile!"
        self.set_full_ranking_method()
        self.logger.info("[MR-HEU] Starting heuristic Accuracy based. This may take a while, but be patient !")
        tm = time.time()
        mr_cfg, pruning_cfg, total_leaves_pruned = self.do_full_ranking()
        tm = time.time() - tm

        self.logger.info(f"[MR-HEU] Accuracy based heuristic completed in {tm} !")
        self.mr_axc.num_cores = self.n_cpus
        # thds = self.mr_axc.tune_thds(mr_cfg)

        mr_vectors = self.mr_axc.mr_predict(mr_cfg, self.mr_axc.x_val, None)
        correct_draw, correct_no_draw = MrAxC.get_correctly_predicted_from_vectors_static(mr_vectors, self.mr_axc.y_val)
        acc = (correct_draw/len(self.mr_axc.y_val)) * 100
        acc_no_draw = correct_no_draw/len(self.mr_axc.y_val) * 100
        self.logger.info(f"[MR-HEU] Acc. {acc} Acc. No Draw {acc_no_draw}")
        # Summary on the evaluation.
        sol_summary = {
                "Algo"                  : self.ranking_procedure_str,
                "MrOrder"               : self.mr_order,
                "Pruned-Leaves"         : len(pruning_cfg),
                "Baseline_XVal_Acc."    : self.mr_axc.x_val_baseline_accuracy,
                "Acc-XVal_Draw"         : acc,
                "Loss-XVal_Draw"        : self.mr_axc.x_val_baseline_accuracy - acc, 
                "Acc-XVal_NO_Draw"      : acc_no_draw,
                "Loss-XVal_NO_Draw"     : self.mr_axc.x_val_baseline_accuracy_nodraw - acc_no_draw,
                "Comp Time [s]"         : tm
            }
        add_header = not os.path.exists(self.csv_outfile)
        df = pd.DataFrame(sol_summary, index=[0]).to_csv(self.csv_outfile, index = False, header = add_header, mode = "a")
        self.logger.info(f"Summary CSV updated! Please check {self.csv_outfile}")
        
        # Dumping pruning and validation indexes
        
        np.savetxt(os.path.join(self.approx_cfg_outdir, "original_ensemble_labels.txt"), self.mr_axc.x_val_class_labels, fmt = "%d")
        np.savetxt(os.path.join(self.approx_cfg_outdir, "original_ensemble_labels_nodraw.txt"), self.mr_axc.x_val_class_labels_nodraw, fmt = "%d")
        np.savetxt(os.path.join(self.approx_cfg_outdir, "val_pred_vectors.txt"), mr_vectors, fmt = "%d")
        self.mr_axc.dump_mop_val_indexes(self.approx_cfg_outdir)
       
        # Dump the per class cfg
        with open(self.pruning_outfiles_dict["outfile_per_class_cfg"], "w") as f:
            json5.dump(mr_cfg, f, indent = 2)
        # Dump the pruning configuration
        with open(self.pruning_outfiles_dict["outfile_pruning_cfg"], "w") as f:
            json5.dump(pruning_cfg, f, indent = 2)



""" This functions are related to the previous implementation of the HEURISTIC !"""
def rank_trees_per_margin(heu_solver: MrHeu):
    remaining_trees = [i for i in range(0, len(heu_solver.mr_axc.classifier.trees))]
    # Get the prediction vectors for each single classifier
    prediction_vectors = heu_solver.mr_axc.classifier.get_votes_vectors_by_leaves_idx(heu_solver.mr_axc.x_mop_leaves, heu_solver.mr_axc.y_mop)
    #x_prun_classes = self.mr_axc.x_mop_classes_transposed
    # For each class. 
    cfgs = []
    for class_idx in heu_solver.mr_axc.sampled_classes:
        pv_per_class    = []    # Set of prediction vectors per each class
        tree_preds      = [[] for tree in heu_solver.mr_axc.classifier.trees]    # Set of tree predictions
        # Get the samples per class 
        for pred_vec, y_idx in zip(prediction_vectors, range(0, len(heu_solver.mr_axc.y_mop))):
            if heu_solver.mr_axc.y_mop[y_idx]== class_idx:
                pv_per_class.append(pred_vec)
                # For each tree, copy the predictions related to that specific sample
                for tree_id in range(0, len(heu_solver.mr_axc.classifier.trees)):
                    tree_preds[tree_id].append(heu_solver.mr_axc.x_mop_classes_transposed[tree_id][y_idx])

        y_prun = [class_idx for _ in pv_per_class]
        per_sample_margins = Pruner.per_sample_margin(pv_per_class, y_prun)
        per_sample_margin_gains, new_pv = Pruner.update_margins(tree_preds=tree_preds, pred_vectors=pv_per_class, remaining_trees=remaining_trees, yprun=y_prun)
        gains = Pruner.evaluate_mean_dm(per_sample_margins, per_sample_margin_gains)
        # Sort the trees by gains
        sorted_trees = np.argsort(gains)
        """ Old code without excluded trees skipping"""        
        # # Keep the trees more contributing to the gain
        # cfgs.append([int(g) for g in sorted_trees[-heu_solver.mr_order:]])
        """ New code with excluded trees skipping"""
        curr_cfg = []
        for tree in sorted_trees[::-1]: # Equivalent to reversed(sorted_trees)
            if tree not in heu_solver.excluded_trees:
                curr_cfg.append(int(tree))
                if len(curr_cfg) == heu_solver.mr_order:
                    break
        assert len(curr_cfg) == heu_solver.mr_order, "[MR-HEU] Something wrong during the configuration generation"
        cfgs.append(curr_cfg)
    return cfgs



def rank_trees_per_accuracy(heu_solver: MrHeu):
    # Get the classes per each tree
    pertree_classes = heu_solver.mr_axc.classifier.transform_leaves_into_classess(heu_solver.mr_axc.x_mop_leaves)
    # Now get the accuracy for each class
    perclass_accs = heu_solver.mr_axc.pertree_classess_into_perclass_pertree_acc(pertree_classes, heu_solver.mr_axc.y_mop)
    # Sort the class indexes 
    # Each configuration consists in the first mr_order treees.
    """ The successive line was old code. """
    #mr_cfg = [list(np.argsort(c_accs)[::-1])[:heu_solver.mr_order] for c_accs in perclass_accs] 
    #mr_cfg = [[int(m) for m in cfg] for cfg in mr_cfg] # Convert numpy.int64 in int
    """ This is the new one. 
        It simply takes into consideration the excluded trees, i.e. those trees which 
        we don't want to insert into the ensemble ( maybe coming from a previous pruning phase).
    """
    mr_cfg = []
    for c_accs in perclass_accs:
        sorted_trees = np.argsort(c_accs)[::-1]
        to_add = heu_solver.mr_order
        added_trees =  0
        cfg = []

        for tree in sorted_trees:
            if tree not in heu_solver.excluded_trees:
                cfg.append(int(tree))
                added_trees += 1
                if added_trees == to_add:
                    break
        assert len(cfg) == heu_solver.mr_order, "[MR-HEU] Something wrong during the configuration generation"
        mr_cfg.append(cfg)
    return mr_cfg


    