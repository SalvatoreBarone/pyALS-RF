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
import os 


def ensemble_pruning_flow(ctx, method, fraction, n_trees, ncpus, report_path, configuration_path ):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Running the Ensemble pruning flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    pruner = Pruner(ctx.obj["classifier"], method, n_trees)
    pruner.prune_test_split(fraction)
    pruner.prune()

