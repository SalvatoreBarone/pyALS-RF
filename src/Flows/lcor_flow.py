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
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .LCOR.lcor import LCOR
import os 

def leaves_correlation_flow(ctx, output, fraction, maxloss_lb, maxloss_ub, loss_step, ncpus, report ):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    lcor = LCOR (ctx.obj["classifier"], fraction, maxloss_lb, 0, ncpus)
    # *** Useful for testing the score function after an update to the pruning set
    # pruning_configuration = []
    # pruning_configuration.append(('1','4','(Node_0)'))
    # pruning_configuration.append(('6','3','(Node_0)'))
    # lcor.append_pruning_conf(pruning_configuration)
    
    print(type(ctx.obj['configuration'].outdir))
    actual_loss = maxloss_lb
    lcor.predispose_trim()
    while actual_loss <= maxloss_ub:
        lcor.max_loss = actual_loss # reset the loss
        report_path = ctx.obj['configuration'].outdir + "/lcor_" + str(actual_loss) + "/lcor_report.csv"
        pruning_path = ctx.obj['configuration'].outdir + "/lcor_" + str(actual_loss) + "/pruning_configuration.json5"
        if not os.path.exists(report_path): # create the experiments path
            os.makedirs(ctx.obj['configuration'].outdir + "/lcor_" + str(actual_loss))
        lcor.trim(report,f"{report_path}") # trim
        lcor.store_pruning_conf(pruning_path) # store the report
        lcor.restore_bns()
        lcor.pruning_configuration = []
        actual_loss += loss_step
    # *** 
    # print(report_path)
    # print(type(f"{ctx.obj['configuration'].outdir}/lcor_report.csv"))
    # #lcor.trim(report,f"{ctx.obj['configuration'].outdir}/lcor_report.csv")
    # lcor.store_pruning_conf(f"{ctx.obj['configuration'].outdir}/pruning_configuration.json5")

    