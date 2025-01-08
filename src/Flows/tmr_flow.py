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
import os 
import time

def tmr_flow(ctx, output, fraction,  ncpus, report, it, test_samples):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    tmr = TMR (ctx.obj["classifier"], fraction,  ncpus,ctx.obj['configuration'].outdir,ctx.obj["flow"], it)
    tmr.approx(test_samples = test_samples)



def mr_mop_flow(ctx, alpha : float, beta : float, gamma : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR-MOO flow.")
    load_configuration_ps(ctx)
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)    
    mr_axc = MrAxC(ctx.obj["classifier"])

    # vars = [1 for i in range(0, len(mr_axc.classifier.model_classes) * len(  mr_axc.classifier.trees))]
    # vars = [1 if i %2 == 0 else 0 for i in range(0, len(mr_axc.classifier.model_classes) * len(  mr_axc.classifier.trees))]
    # print(vars)
    # confs = MrMop.get_tree_cfg(mr_axc, vars)
    # print(confs)
    # accs = mr_axc.evaluate_cfg_xmop(confs)
    # print(accs)

    # DA DECOMMENTARE DOPO I TEST SU ACCURACY
    create_problem(ctx, mode = None, alpha = alpha, beta = beta, gamma = gamma)
    ctx.obj["problem"].initialize_problem(mr_axc)
    create_optimizer(ctx)
    can_improve(ctx)
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
    #ctx.obj["optimizer"].archive.plot_front(ctx.obj['problem'].num_of_objectives, f"{ctx.obj['configuration'].outdir}/pareto_front.pdf") # It gives me an error...
    ctx.obj["pareto_front"] = ctx.obj["optimizer"].archive
    logger.info(f"All done! Take a look at the {ctx.obj['configuration'].outdir} directory.")
