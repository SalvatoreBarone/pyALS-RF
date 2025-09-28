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
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .EnsemblePruning.EnsemblePruner import Pruner
import pandas as pd
import os 


def ensemble_pruning_flow(ctx, method, fraction, n_trees, ncpus, report_path, configuration_path ):
    logger = logging.getLogger("pyALS-RF")
    logger.info("[ENSEMBLE-PRUNING] Running the Ensemble pruning flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    pruner = Pruner(ctx.obj["classifier"], method, n_trees)
    pruner.prune_test_split(fraction)
    pruner.prune()

    # pruner.pruned_trees Questo indica gli indici degli alberi prunati
    # Genera il report e salva la configurazione.
    # Ora ho anche la pruning conf.

    logger.info(f"[ENSEMBLE-PRUNING] Pruning completed!")

    
    # Dump the report
    report = {
        "acc_loss": pruner.loss,
        "remaining_trees": len(pruner.actual_trees),
        "baseline_acc": pruner.baseline_accuracy,
        "pruned_acc": pruner.pruned_accuracy
    }
    report_file_path = os.path.join(report_path, "pruning_report.csv")
    logger.info(f"[ENSEMBLE-PRUNING] Dumping the REPORT in {report_file_path}")    
    add_header = not os.path.exists(report_file_path)
    df = pd.DataFrame(report, index=[0]).to_csv(report_file_path, index = False, header = add_header, mode = "a")
    # Dump the list of pruned trees indexes
    
    pruned_trees_path = os.path.join(configuration_path, "pruned_trees.txt")
    logger.info(f"[ENSEMBLE-PRUNING] Dumping the pruned trees indexes in {pruned_trees_path}")
    np.savetxt(pruned_trees_path, pruner.pruned_trees, fmt = "%d")
    # Dump the Grep-like pruning configuration
    pruning_cfg_path = os.path.join(configuration_path, "pruning_configuration.json5")
    logger.info(f"[ENSEMBLE-PRUNING] Dumping the pruning conf in {pruning_cfg_path}")

    with open(pruning_cfg_path, "w") as f:
        json5.dump(pruner.pruning_conf, f, indent = 2)

