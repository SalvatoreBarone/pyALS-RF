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
def mr_heu_flow(ctx, method, fraction, mr_order, ncpus, pruning_dir, csv_dir):
    logger = logging.getLogger("pyALS-RF")
    logger.info("[MR-HEU-FLOW] Running the MR Heuristics flow")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    # Initialize the MRAxC object.
    logger.info("[MR-HEU-FLOW] Initializing the MrAxC object..")
    mr_axc = MrAxC(ctx.obj["classifier"], 1, fraction) # Fix the num_cores value to 1.
    logger.info("[MR-HEU-FLOW] MrAxC object initialized!")
    logger.info("[MR-HEU-FLOW] Initializing the MrHeu object")
    mr_heu = MrHeu(mr_order, ncpus, method=method)
    mr_heu.initialize_problem(mr_axc)
    mr_heu.initialize_pruning_cfg_out(pruning_dir)
    mr_heu.initialize_summary_files(csv_dir)
    logger.info("[MR-HEU-FLOW] MrHeu initialized !")
    logger.info("[MR-HEU-FLOW] Running problem!")
    mr_heu.heu_tree_acc()
    #mr_heu.rank_trees_per_margin()
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

        

    