"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>
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
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .LCOR.lcor import LCOR
from .LCOR.lcor_axc import LCOR_AXC
import os 


def leaves_correlation_flow(ctx, output, fraction, maxloss_lb, maxloss_ub, loss_step, ncpus, report ):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    lcor = LCOR (ctx.obj["classifier"], fraction, maxloss_lb, 0, ncpus,ctx.obj['configuration'].outdir,ctx.obj["flow"])

    actual_loss = maxloss_lb

    lcor.trim_alternative(report,maxloss_lb,maxloss_ub,loss_step)


def leaves_correlation_flow_2(ctx, fraction, fraction_validation, max_loss , ncpus, report_path, pruning_path ):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    assert report_path != None
    assert pruning_path != None
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    if not os.path.exists(pruning_path):
        os.makedirs(pruning_path)
    lcor = LCOR_AXC(ctx.obj["classifier"], pruning_set_fraction = fraction, validation_set_fraction= fraction_validation, max_loss = max_loss, min_resiliency = 0, ncpus = ncpus, report_path = report_path, pruning_path = pruning_path,flow = ctx.obj["flow"])
    lcor.trim()

