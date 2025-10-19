"""
Copyright 2021-2024 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it>
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
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow, create_problem, create_optimizer, can_improve

from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .TMR.tmr import TMR
from .TMR.mr_axc import MrAxC
from .TMR.mr_moo import MrMop
from .TMR.mr_heu import MrHeu
import os 
import time
import pandas as pd
from .GREP.GREP import GREP
import re
# import json5
# Given a pareto front (a list of dictionaries), generate a set of unique solutions 
def __unique_pareto(pareto):
    seen = set()
    unique_data = []
    for entry in pareto:
        x_tuple = tuple(entry["x"])  # Convert `x` to a tuple (hashable)
        if x_tuple not in seen:
            seen.add(x_tuple)
            unique_data.append(entry)
    return unique_data

def tmr_flow(ctx, output, fraction,  ncpus, report, it, test_samples, mr_order, report_name):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    tmr = TMR (ctx.obj["classifier"], fraction,  ncpus,ctx.obj['configuration'].outdir,ctx.obj["flow"], it, mr_order, report_name)
    tmr.approx(test_samples = test_samples)

""" This is a substitute for the TMR flow.  
    The code in TMR flow was bloated and full of initial experiments.
"""
def mr_heu_flow(ctx, quantization_type, in_pruning, method, fraction, mr_order, ncpus, pruning_dir, csv_dir):
    logger = logging.getLogger("pyALS-RF")
    logger.info("[MR-HEU-FLOW] Running the MR Heuristics flow")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    """ Useless code used to import a pruning configuration.
        Now the heuristic themself manages a previous ensemble pruning conf by simply taking in input the list of pruned trees."""
    if in_pruning is not None:
        pruned_trees = np.loadtxt(in_pruning, dtype = int)
        # print(pruned_trees)
    else:
        pruned_trees = []
    # Initialize the MRAxC object.
    logger.info("[MR-HEU-FLOW] Initializing the MrAxC object..")
    classifier = ctx.obj["classifier"]
    if quantization_type != None:
        classifier.set_thds_type(quantization_type)
    mr_axc = MrAxC(classifier, 1, fraction) # Fix the num_cores value to 1.
    logger.info("[MR-HEU-FLOW] MrAxC object initialized!")
    logger.info("[MR-HEU-FLOW] Initializing the MrHeu object")
    mr_heu = MrHeu(mr_order, ncpus, method=method, excluded_trees=pruned_trees)
    mr_heu.initialize_problem(mr_axc)
    mr_heu.initialize_pruning_cfg_out(pruning_dir)
    mr_heu.initialize_summary_files(csv_dir)
    logger.info("[MR-HEU-FLOW] MrHeu initialized !")
    logger.info("[MR-HEU-FLOW] Running problem!")
    mr_heu.heu_tree_acc_2()
    logger.info(f"[MR-HEU-FLOW] Problem completed, take a look at {pruning_dir} and {csv_dir}")
    

def mr_mop_flow(ctx, alpha : float, beta : float, gamma : float, output : str, n_jobs: int = 1, fraction: float = None):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR-MOO flow.")
    load_configuration_ps(ctx)
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    mr_axc = MrAxC(ctx.obj["classifier"], 1, fraction) # Fix this value to 1.
    create_problem(ctx, mode = None, alpha = alpha, beta = beta, gamma = gamma)
    # Ad hoc function.
    ctx.obj["problem"].initialize_problem(mr_axc)
    create_optimizer(ctx)
    can_improve(ctx)
    # RIMUOVERE APPENA SI E' FIXATA LA GENERAZIONE DELLE DIREZIONI.
    # Now create the problem
    ctx.obj["optimizer"].run(ctx.obj["problem"], termination_criterion = ctx.obj['configuration'].termination_criterion, improve = ctx.obj["improve"])
    logger.info(f"AMOSA heuristic completed!")
    hours = int(ctx.obj["optimizer"].duration / 3600)
    minutes = int((ctx.obj["optimizer"].duration - hours * 3600) / 60)
    logger.info(f"Took {hours} hours, {minutes} minutes")
    logger.info(f"Cache hits: {ctx.obj['problem'].cache_hits} over {ctx.obj['problem'].total_calls} evaluations.")
    logger.info(f"{len(ctx.obj['problem'].cache)} cache entries collected")
    logger.info(f"Saving the validation and mop set indexes")
    ctx.obj["optimizer"].archive.write_json(f"{ctx.obj['configuration'].outdir}/final_archive.json")
    ctx.obj["pareto_front"] = ctx.obj["optimizer"].archive
    logger.info(f"Pareto front saved! Take a look at the {ctx.obj['configuration'].outdir} directory.")
    logger.info("Dumping the MOP and Validation indexes...")
    # Dump MOP and validation indexes.
    mr_axc.dump_mop_val_indexes(ctx.obj['configuration'].outdir)
    logger.info(f"Dump of Validation and MOP completed ! Check {ctx.obj['configuration'].outdir} directory.") 
    
    # # Used for log infos generation tests.
    # with open(os.path.join(ctx.obj['configuration'].outdir, "final_archive.json"), "r") as f:
    #     pareto_no_rep = json5.load(f)

    # Get the leaves for each sample in the validation set.
    logger.info(f"Initializing classes for evaluating validation across solutions....")
    validation_leaves = mr_axc.classifier.compute_leaves_idx(mr_axc.x_val, False)
    validation_classes = mr_axc.classifier.transform_leaves_into_classess(validation_leaves)
    validation_classes = MrAxC.per_tree_classess_into_classes_per_tree(validation_classes)
    logger.info(f"Classes per tree initialized ! Now evaluating the different CFG.")
    # For each configuration in the pareto front, save the pruning indexed of MOP and validation.
    pareto_no_rep = __unique_pareto(ctx.obj["pareto_front"].candidate_solutions) 
    out_path = ctx.obj['configuration'].outdir
    
    # Save the solution in the set of unique pareto fronts. 
    for solution in pareto_no_rep:
        logger.info(f"Evaluating solution {solution}")
        # Get solution X
        configuration = solution["x"]
        x_mop_acc_draw = solution["f"][0]

        # Transform the solution into a feasible configuration.
        configuration = MrMop.get_tree_cfg(mr_axc, configuration)
        # Evaluate
        mr_pred_vectors =  MrAxC.get_mr_vectors(validation_classes, configuration)
        val_acc_draw, val_acc_no_draw = MrAxC.get_accuracy_from_vectors(mr_pred_vectors, mr_axc.y_val)
        # Need to dump pred_vector for future cross validation procedures. 
        # So the accuracy evaluation function is not directly called.
        #val_acc_draw, val_acc_no_draw = MrAxC.evaluate_mr_cfg_accuracy(validation_classes, mr_axc.y_val, configuration)
        loss_draw = mr_axc.x_val_baseline_accuracy - val_acc_draw
        loss_no_draw = mr_axc.x_val_baseline_accuracy - val_acc_no_draw
        logger.info(f"Evaluation completed! : Baseline: {mr_axc.x_val_baseline_accuracy}")
        logger.info(f"Draw considered as missclassifications Acc. : {val_acc_draw}, Loss: {loss_draw}")
        logger.info(f"Draw NOT considered as missclassification Acc. : {val_acc_no_draw}, Loss: {loss_no_draw}")
        # Get indexes for the configuration 
        out_dir_cfg = os.path.join(out_path, f"cfg_{loss_no_draw:.2f}")
        if not os.path.exists(out_dir_cfg):
            os.makedirs(out_dir_cfg)
        logger.info(f"Dumping MR prediction vectors on validation set")
        pred_vec_dump_path = os.path.join(out_dir_cfg, "mr_pred_vectors.json5")
        with open(pred_vec_dump_path, "w") as f:
            json5.dump(mr_pred_vectors.tolist(), f, indent = 2)
 
        logger.info(f"Starting the dump of CFG infos.")
        # The configuration consists in the set of trees per each class, so this function returns the set of classes
        # per each different tree.
        per_tree_cfg = MrAxC.cfg_per_class_in_cfg_per_tree(mr_axc, configuration)
        pruned_leaves = mr_axc.classifier.get_leaf_indexes_not_in_class_list(per_tree_cfg)
        # Dump the configuration per class object. 
        with open(os.path.join(out_dir_cfg, "per_class_cfg.json5"), "w") as f:
            json5.dump(configuration, f, indent = 2)
        # Dump the configuration itself.
        with open(os.path.join(out_dir_cfg, "per_tree_cfg.json5"), "w") as f:
            json5.dump(per_tree_cfg, f, indent = 2)
        # Dump the leaf indexes.
        with open(os.path.join(out_dir_cfg, "leaves_idx.json5"), "w") as f:
            json5.dump(pruned_leaves, f, indent = 2)
        logger.info("Generating and dumping pruning configuration for the accelerator...")
        pruning_cfg = GREP.get_pruning_cfg_from_leaves_idx(mr_axc.classifier, pruned_leaves)
        pruning_cfg_path = os.path.join(out_dir_cfg, "pruning_conf.json5")
        with open(pruning_cfg_path, "w") as f:
            json5.dump(pruning_cfg, f, indent = 2)
        logger.info(f"Pruning CFG Dump Completed! Check {pruning_cfg_path}")
        # Generate the pruning configuration used by the GREP-like tools.
        logger.info(f"Updating the summary CSV file.")
        pruned_leaves_ctr = 0
        for tree, classes_per_tree_pruned_leaves in pruned_leaves.items():
            for _, pruned_leaves in classes_per_tree_pruned_leaves.items():
                pruned_leaves_ctr += len(pruned_leaves)
        logger.info(f"CFG infos dumped. Check {out_dir_cfg}")
        logger.info(f"Generating the Direction Files for exporting pruning configuration.")
        # Generating direction files for dumping.
        direction_file_json = mr_axc.classifier.transform_assertion_into_directions(pruning_cfg)
        out_path_directions = os.path.join(out_dir_cfg, "leaf_pruning_directions.json5")
        # Dump the direction file.
        with open(out_path_directions, "w") as f:
            json5.dump(direction_file_json, f, indent = 2)
        logger.info(f"Direction File dumped at {out_path_directions}")
        sol_summary = {
                "Pruned-Leaves"         : pruned_leaves_ctr,
                "Baseline_XMOP_Acc"     : mr_axc.x_val_baseline_accuracy,
                "Acc-XMOP_Draw"         : x_mop_acc_draw,  
                "Loss-XMOP_Draw"        : mr_axc.x_val_baseline_accuracy - x_mop_acc_draw,
                "Baseline_XVal_Acc."    : mr_axc.x_val_baseline_accuracy,
                "Acc-XVal_Draw"         : val_acc_draw,
                "Loss-XVal_Draw"        : loss_draw,
                "Acc-XVal_NO_Draw"      : val_acc_no_draw,
                "Loss-XVal_NO_Draw"     : loss_no_draw,
            }
        out_summary_csv = os.path.join(out_path, "summary.csv")
        add_header = not os.path.exists(out_summary_csv)
        df = pd.DataFrame(sol_summary, index=[0]).to_csv(out_summary_csv, index = False, header = add_header, mode = "a")
        logger.info(f"Summary CSV updated! Please check {out_summary_csv}")

        


def mr_additional_eval(ctx, quantization_type, ncpus, exp_path, subpath_k, subpath_rep, k_lb, k_ub, k_step, nreps):
    
    # ************ START UTILITY FUNCTIONS
    # Estimate the node number 
    def estimate_node_number(tree_leaves, classifier):
        """  
            Given the set of tree leaves for a classifier, this function estimates the number of 
                nodes required for an inference.
            Parameters
                    ----------
                    tree_leaves : np.ndarray
                        #Rows == # Trees
                        # Colums = Leaves for that tree reached from the sample.
                    classifier : Classifier
                        classifier object for which the leaves were computed
                    
                    Returns
                    -------
                        nodeCountsPerLeaf : np.ndarray
                        # Rows = # Samples
                        # Columns = # Node counts for a specific tree.
        """
        nodeCountsPerLeaf = [[ 0 for treeIdx in range(len(tree_leaves)) ] for sampleIdx in range(len(tree_leaves[0]))]
        nodesPerSample = [0 for sample in range(len(tree_leaves[0]))]

        for treeIdx in range(len(tree_leaves)):
            for sampleIdx in range(len(tree_leaves[0])):
                # Get the leaf SOP
                minterm = classifier.trees[treeIdx].leaves[tree_leaves[treeIdx][sampleIdx]]["sop"] 
                nodeCountsPerLeaf[sampleIdx][treeIdx] = len(re.findall(r'Node_\d+', minterm))
                nodesPerSample[sampleIdx] += nodeCountsPerLeaf[sampleIdx][treeIdx]

        return np.asarray(nodeCountsPerLeaf), np.asarray(nodesPerSample)
    

    def analyze_vectors(vectors):
        """
        Given an array of binary vectors, returns for each vector:
        - index of the first 1 (or -1 if none)
        - flag indicating whether there are two or more 1s
        - flag indicating whether there are no 1s
        """
        results = []

        for vec in vectors:
            ones_indices = np.where(vec == 1)[0]

            if len(ones_indices) == 0:
                first_idx = -1
                has_two = False
                has_none = True
            else:
                first_idx = ones_indices[0]
                has_two = len(ones_indices) > 1
                has_none = False

            results.append([first_idx, has_two, has_none])

        return np.array(results, dtype=object)

    
    def estimate_node_number_MR(nodesPerSamplePerLeaf, mr_cfg, tmr_vectors):
        """  
            Estimate the number of nodes required for MR on BRAM/Microcontroller based 
                accelerators.

            Parameters
                    ----------
                    nodesPerSamplePerLeaves : np.ndarray
                        np.ndarray 
                        #Rows == # Samples
                        # Colums = # Leaves
                        Each element i,j represents the number of nodes required to reach a leaf ( stimulated by sample i).
                    mr_cfg: 
                        np.ndarray
                        For each class, mantains the Modular Redundancy CFG of that class.
                    tmr_vectors_results: 
                        np.ndarray
                        Results
                    Returns
                    -------
                        nodeCountsPerLeafMR : np.ndarray
                        # Rows = # Samples
                        Mantains for each sample the number of nodes visited by the modular redundancy structure.
        """
        nodesPerSampleMR = [0 for sample in range(len(nodesPerSamplePerLeaf))]
        for sampleIdx in range(len(nodesPerSamplePerLeaf)): # For each sample
            nodes4Tree = 0
            alreadyExplored = []
            for cfgIdx, trees_cfg in enumerate(mr_cfg): # For each tree in the MR CFG
                for tree in trees_cfg: 
                    if tree not in alreadyExplored: # If the tree has not been yet explored
                        alreadyExplored.append(tree) # Append the tree and increase the node count
                        nodes4Tree += nodesPerSamplePerLeaf[sampleIdx][tree]
                if tmr_vectors[sampleIdx][cfgIdx] == 1: # If the sample was labeled accordingly, stop the inference. 
                    break
            nodesPerSampleMR[sampleIdx] = nodes4Tree
        return nodesPerSampleMR

    
    def tmr_accuracy_per_class(vec_results, y_true):
        """
        Computes accuracy per class given prediction results and true labels.
        
        Parameters
        ----------
        vec_results : np.ndarray
            Each row = [predicted_class, has_two, has_none]
        y_true : np.ndarray
            Ground truth labels (same length as vec_results)
        
        Returns
        -------
        dict
            {class_label: accuracy_percentage}
        """
        y_pred = vec_results[:, 0].astype(int)
        has_two = vec_results[:, 1].astype(bool)
        has_none = vec_results[:, 2].astype(bool)

        # Mask mislabelled samples: either multiple ones or no ones
        valid = ~(has_two | has_none)
        correct = (y_pred == y_true) & valid

        classes, total_counts = np.unique(y_true, return_counts=True)
        accuracies = {}
        correctly_labeled_per_class = {}
        for cls, total in zip(classes, total_counts):
            correct_cls = np.sum(correct[y_true == cls])
            accuracies[cls] = (correct_cls / total) * 100
            correctly_labeled_per_class[cls] = correct_cls
        return accuracies, correctly_labeled_per_class
    
    def argmax_with_tie(vectors):
        """
        Returns the argmax index for each vector.
        If there is a tie for the maximum value, returns -1.
        
        Parameters
        ----------
        vectors : np.ndarray
            2D array of shape (n_samples, n_classes)
        
        Returns
        -------
        np.ndarray
            Array of labels (argmax or -1 in case of tie)
        """
        # Get argmax indices
        argmax_indices = np.argmax(vectors, axis=1)
        # Get max values
        max_values = np.max(vectors, axis=1, keepdims=True)
        # Count how many times the max appears in each row
        ties = np.sum(vectors == max_values, axis=1) > 1
        # Assign -1 where there is a tie
        result = np.where(ties, -1, argmax_indices)
        return result


    def exact_accuracy_per_class(y_pred, y_true):
        """
        Compute per-class accuracy for the exact classifier.

        Steps:
        1) Total samples per class = count of y_true == cls
        2) Correct per class = count of indices where (y_true == cls) AND (y_pred == cls)
        3) Accuracy = correct / total * 100
        4) Return both accuracy dict and correctly-labeled counts dict
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"Length mismatch: y_pred={y_pred.shape[0]} vs y_true={y_true.shape[0]}")

        classes = np.unique(y_true)
        acc_per_class = {}
        correctly_labeled_per_class = {}

        for cls in classes:
            idx = (y_true == cls)                 # indices of this class
            total = int(np.sum(idx))              # 1) total per class
            correct = int(np.sum(y_pred[idx] == cls))  # 2) correct per class
            acc = (correct / total) * 100 if total > 0 else 0.0  # 3) accuracy %
            acc_per_class[cls] = acc
            correctly_labeled_per_class[cls] = correct

        return acc_per_class, correctly_labeled_per_class
        


    def update_experiment_csv(
        csv_path,
        mr_order,
        heuristic,
        acc_exact,
        acc_approx,
        correctly_exact,
        correctly_tmr,
        class_counts_train,
        class_counts_test,
        freq_percent_train,
        freq_percent_test,
        class_counts_mop,
        freq_percent_mop
    ):
        """
        Update or create a CSV summarizing per-class statistics for Train, Test, and MOP,
        along with per-class MR/Exact correctness, accuracies, and accuracy losses.

        Columns per class:
            #SamplesTrain_ClassX
            #SamplesTest_ClassX
            FreqTrain_ClassX
            FreqTest_ClassX
            #MOPIndexes_ClassX
            #MOPFrequency_ClassX
            #MOPLabels_ClassX
            #CorrectlyLabeledSamplesMR_ClassX
            #CorrectlyLabeledSamplesExact_ClassX
            #AccuracyMR_ClassX
            #AccuracyExact_ClassX
            #Loss_ClassX

        Global columns:
            mr_order, Heuristic, Average_AccMR_Loss, Average_AccExact
        """

        # ---- 1. Compute loss per class (still per-class, not averaged) ----
        loss_per_class = {}
        for cls in acc_exact.keys():
            acc_e = acc_exact.get(cls, np.nan)
            acc_a = acc_approx.get(cls, np.nan)
            if acc_e > 0 and not np.isnan(acc_a):
                loss_per_class[cls] = acc_a - acc_e
            else:
                loss_per_class[cls] = np.nan

        # ---- 2. Collect all unique class labels (no sorting by frequency) ----
        all_classes = sorted(
            set(
                list(acc_exact.keys())
                + list(acc_approx.keys())
                + list(class_counts_train.keys())
                + list(class_counts_test.keys())
                + list(class_counts_mop.keys())
            )
        )

        # ---- 3. Prepare a single row of data ----
        row = {
            "mr_order": mr_order,
            "Heuristic": heuristic,
            # Global metrics
            "Average_AccMR_Loss": float(np.nanmean(list(loss_per_class.values()))),
            "Average_AccExact": float(np.nanmean(list(acc_exact.values()))),
        }

        for cls in all_classes:
            row[f"#SamplesTrain_Class{cls}"] = class_counts_train.get(cls, np.nan)
            row[f"#SamplesTest_Class{cls}"] = class_counts_test.get(cls, np.nan)
            # row[f"FreqTrain_Class{cls}"] = freq_percent_train.get(cls, np.nan)
            # row[f"FreqTest_Class{cls}"] = freq_percent_test.get(cls, np.nan)
            row[f"#MOPSamples_Class{cls}"] = class_counts_mop.get(cls, np.nan)
            # row[f"MOPFrequency_Class{cls}"] = freq_percent_mop.get(cls, np.nan)
            # row[f"#MOPLabels_Class{cls}"] = class_counts_mop.get(cls, np.nan)
            row[f"#CorrectlyLabeledSamplesMR_Class{cls}"] = correctly_tmr.get(cls, np.nan)
            row[f"#CorrectlyLabeledSamplesExact_Class{cls}"] = correctly_exact.get(cls, np.nan)
            row[f"#AccuracyMR_Class{cls}"] = acc_approx.get(cls, np.nan)   
            row[f"#AccuracyExact_Class{cls}"] = acc_exact.get(cls, np.nan) 
            row[f"#Loss_Class{cls}"] = loss_per_class.get(cls, np.nan)

        row_df = pd.DataFrame([row])

        # ---- 4. Append or create CSV ----
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add missing columns if needed
            for col in row_df.columns:
                if col not in df.columns:
                    df[col] = np.nan
            # Align column order and append
            row_df = row_df.reindex(columns=df.columns, fill_value=np.nan)
            df = pd.concat([df, row_df], ignore_index=True)
        else:
            # Define consistent column order
            col_order = ["mr_order", "Heuristic"]
            for cls in all_classes:
                col_order += [
                    f"#SamplesTrain_Class{cls}",
                    f"#SamplesTest_Class{cls}",
                    # f"FreqTrain_Class{cls}",
                    # f"FreqTest_Class{cls}",
                    f"#MOPIndexes_Class{cls}",
                    # f"#MOPFrequency_Class{cls}",
                    # f"#MOPLabels_Class{cls}",
                    f"#CorrectlyLabeledSamplesMR_Class{cls}",
                    f"#CorrectlyLabeledSamplesExact_Class{cls}",
                    f"AccuracyMR_Class{cls}",      
                    f"AccuracyExact_Class{cls}%",   
                    f"#Loss_Class{cls}",
                ]
            col_order += ["Average_AccMR_Loss", "Average_AccExact"]
            row_df = row_df.reindex(columns=col_order)
            df = row_df

        # ---- 5. Save ----
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Updated {csv_path} with mr_order '{mr_order}' and heuristic '{heuristic}'.")
    
    def update_node_count_savings_csv(
        exact_nodes,
        approximate_nodes,
        mr_order, 
        heuristic,
        out_path="node_count_savings.csv"):
        """
        Update or create a CSV summarizing average node counts and savings per experiment.

        Parameters
        ----------
        exact_nodes : list or np.ndarray
            List or array of node counts from the exact (non-approximate) classifier.
        approximate_nodes : list or np.ndarray
            List or array of node counts from the approximate (MR) classifier.
        mr_order: int
            Order k of modular redundancy for the configuration.
        heuristic: string
            Name of the heuristic algo employed for obtaining the mr_cfg
        out_path : str
            Path to the CSV file to update or create.
        """
        # ---- 1. Compute averages ----
        exact_avg = np.mean(exact_nodes)
        approx_avg = np.mean(approximate_nodes)

        # ---- 2. Compute node savings percentage ----
        if exact_avg > 0:
            node_savings = (1 - (approx_avg / exact_avg)) * 100
        else:
            node_savings = np.nan

        # ---- 3. Prepare new row ----
        row = {
            "Heuristic": heuristic,
            "MrOrder": mr_order,
            "ExactNodes": exact_avg,
            "ApproximateNodes": approx_avg,
            "NodeSavings": node_savings,
        }

        row_df = pd.DataFrame([row])

        # ---- 4. Create or append to CSV ----
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            # Add missing columns if needed
            for col in row_df.columns:
                if col not in df.columns:
                    df[col] = np.nan
            row_df = row_df.reindex(columns=df.columns, fill_value=np.nan)
            df = pd.concat([df, row_df], ignore_index=True)
        else:
            df = row_df[
                ["Heuristic", "MrOrder", "ExactNodes", "ApproximateNodes", "NodeSavings"]
            ]

        # ---- 5. Save ----
        df.to_csv(out_path, index=False)
        print(f"✅ Updated {out_path} with Heuristic '{row['Heuristic']}' and MrOrder '{row['MrOrder']}'.")
            
    # ******************** END UTILITY FUNCTIONS
    logger = logging.getLogger("pyALS-RF")
    logger.info("[MR-HEU-FLOW] Performing additional evaluation on MR experiments.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    
    # Initialize the classifier and all the objects required.
    logger.info("[MR-HEU-FLOW] Initializing the Classifier.")
    
    classifier = ctx.obj["classifier"]
    if quantization_type != None:
        classifier.set_thds_type(quantization_type)
    # Computing classifier prediction Vectors.
    test_samples = classifier.x_test
    
    # Combine training and test labels
    train_labels = classifier.y_train.ravel()
    test_labels  = classifier.y_test.ravel()


    logger.info("[MR-HEU-FLOW] Computing class labels statistics...")

    # Compute unique classes and counts
    unique, train_counts = np.unique(train_labels, return_counts=True)
    _, test_counts = np.unique(test_labels, return_counts=True)
    total_train = len(train_labels)
    total_test = len(test_labels)

    # Dictionaries with frequencies and absolute counts for TRAINING SET
    freq_percent_train = {u: (c / total_train) * 100 for u, c in zip(unique, train_counts)}
    freq_percent_test = {u: (c / total_test) * 100 for u, c in zip(unique, test_counts)}
    
    class_counts_train = {u: c for u, c in zip(unique, train_counts)}
    class_counts_test = {u: c for u, c in zip(unique, test_counts)}

    #sorted_classes = sorted(freq_percent.keys(), key=lambda k: freq_percent[k], reverse=True)

    logger.info("[MR-HEU-FLOW] Initializing the Classifier predictions")
    leaves_per_tree = classifier.compute_leaves_idx(test_samples, disable_tqdm = False)
    

    logger.info("[MR-HEU-FLOW] Transforming the tree leaves")
    # HERE GET THE GENERAL PRUNING CONF ORDERED SIMILARLY TO THE LEAVES.

    votes_vector = classifier.get_votes_vectors_by_leaves_idx(leaves_per_tree, classifier.y_test)
    original_classifier_labels = argmax_with_tie(votes_vector)
    
    logger.info("Estimating node counts for each different leaf")
    nodeCountsPerLeaf, exactNodeCounts = estimate_node_number(leaves_per_tree, classifier)
    
    logger.info("[MR-HEU-FLOW] Initialize Running for cfgs.")
    for mr in range(k_lb, k_ub + k_step, k_step):
        mr_path = os.path.join(exp_path, f"{subpath_k}{mr}")
        for rep in range(1, nreps + 1):
            cfg_path = os.path.join(mr_path, f"{subpath_rep}{rep}")
            logger.info(f"Running for MR order {mr} Rep: {rep} Path {cfg_path}")
            # The output path is in "val_pred_vectors"
            val_pred_vec_path = os.path.join(cfg_path, "val_indexes.txt")
            mop_pred_vec_path = os.path.join(cfg_path, "mop_indexes.txt")
            # Handle the label indexes.
            if not os.path.exists(val_pred_vec_path):
                logger.error("[MR-HEU-FLOW] Invalid path, please provide a structured path !")
                logger.error(val_pred_vec_path)
                assert 1 == 0
            # Transform the prediction vectors for pred vecs to classes.
            else: 
                
                mop_indexes = np.loadtxt(mop_pred_vec_path, dtype=int)
                val_indexes = np.loadtxt(val_pred_vec_path, dtype=int)
                labels_cfg = test_labels[val_indexes]
                # Compute percentages over the MOP labels
                mop_labels = test_labels[mop_indexes]
                _, mop_counts = np.unique(mop_labels, return_counts=True)
                total_mop = len(mop_labels)
                freq_percent_mop = {u: (c / total_mop) * 100 for u, c in zip(unique, mop_counts)}
                class_counts_mop = {u: c for u, c in zip(unique, mop_counts)}
                # Load the configuration
                
                with open(os.path.join(cfg_path, "per_class_cfg.json5"), "r") as f:
                    loaded_cfg = json5.load(f)
                mr_cfg = np.array(loaded_cfg)

                tmr_pred_vecs = np.loadtxt(os.path.join(cfg_path, "val_pred_vectors.txt"), dtype=int)
                if len(val_indexes) == len(tmr_pred_vecs):
                    # ********** PER CLASS ACCURACY CODE
                    vec_results = analyze_vectors(tmr_pred_vecs)
                    acc4ClassTMR, correctlyLabeledTMR = tmr_accuracy_per_class(vec_results=vec_results,y_true=labels_cfg)
                    acc4ClassExact, correctlyLabeledExact= exact_accuracy_per_class(y_pred=original_classifier_labels[val_indexes], y_true=labels_cfg )
                    # losses = accuracy_loss_per_class(acc4ClassExact, acc4ClassTMR)
                    # update_experiment_csv(os.path.join(exp_path, "per_class_acc_report.csv"), mr, "accuracy", acc4ClassExact, acc4ClassTMR, freq_percent)
                    update_experiment_csv(csv_path = os.path.join(exp_path, "per_class_acc_report.csv"),
                                          mr_order=mr, 
                                          heuristic="accuracy", 
                                          acc_exact=acc4ClassExact,
                                          acc_approx=acc4ClassTMR,
                                          correctly_exact=correctlyLabeledExact,
                                          correctly_tmr=correctlyLabeledTMR,
                                          class_counts_train=class_counts_train,
                                          class_counts_test=class_counts_test,
                                          class_counts_mop=class_counts_mop,
                                          freq_percent_train=freq_percent_train,
                                          freq_percent_test=freq_percent_test,
                                          freq_percent_mop=freq_percent_mop
                                          )
                    
                    # ********** AVG NUMBER NODES INFERENCE
                    MRVisitedNodes = estimate_node_number_MR(
                                            nodesPerSamplePerLeaf=nodeCountsPerLeaf[val_indexes], 
                                            mr_cfg=mr_cfg,
                                            tmr_vectors=tmr_pred_vecs)
                    update_node_count_savings_csv(exact_nodes = exactNodeCounts, 
                                                  approximate_nodes = MRVisitedNodes, 
                                                  mr_order= mr,
                                                  heuristic = "accuracy",
                                                  out_path=os.path.join(exp_path, "node_counts_report.csv"))
                else: 
                    logger.error("Incorrect tmr vectors file.....")


                
                
