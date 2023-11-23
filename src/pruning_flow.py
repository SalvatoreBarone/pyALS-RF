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
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from tqdm import tqdm
from .ctx_factory import load_configuration_ps, create_classifier
from .ConfigParsers.PsConfigParser import *
from .AxCT.HedgeTrimming import HedgeTrimming
from .AxCT.ResiliencyBasedHedgeTrimming import ResiliencyBasedHedgeTrimming
from .AxCT.LossBasedHedgeTrimming import LossBasedHedgeTrimming
from .Model.Classifier import Classifier
from .plot import boxplot


def pruning_flow(ctx : dict, fraction : float, approach: str, cost_criterion: str, minredundancy : int, maxloss : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
       
    trimmer = LossBasedHedgeTrimming(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"]) if approach == "loss" else ResiliencyBasedHedgeTrimming(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"])
    trimmer.trim(HedgeTrimming.get_cost_criterion(cost_criterion))
    trimmer.store_pruning_conf(f"{ctx.obj['configuration'].outdir}/pruning_configuration.json5")
    trimmer.redundancy_boxplot(f"{ctx.obj['configuration'].outdir}/redundancy_boxplot.pdf")
    trimmer.restore_bns()
    
def redundancy_plot(ctx : dict, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
        
    create_classifier(ctx)
    classifier = ctx.obj["classifier"]
    dump_file = f"{ctx.obj['configuration'].outdir}/classifier.joblib"
    skmodel = joblib.load(dump_file)
    predictions = skmodel.predict_proba(np.array(classifier.x_test))
    samples_error = { i: [] for i in range(skmodel.n_classes_) }
    redundancy = []
    for y, p in tqdm( zip(classifier.y_test, predictions), total=len(classifier.y_test), desc="Computing error...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        if  np.argmax(p) == y and not Classifier.check_draw(p)[0]:
            for i in range(skmodel.n_classes_):
                if i != y:
                    samples_error[i].append(np.ceil( (p[y] - p[i]) / 2)[0])
            r = np.sort(np.array(p, copy=True))[::-1]
            redundancy.append((r[0] - r[1] - 1) // 2)

    redundancy_boxplot = f"{ctx.obj['configuration'].outdir}/redundancy_boxplot.pdf"
    error_boxplot = f"{ctx.obj['configuration'].outdir}/error_boxplot.pdf"
    boxplot(redundancy, "", "Redundancy", redundancy_boxplot, figsize = (2, 4), annotate = False, integer_only= True)
    boxplot([ list(v) for v in samples_error.values()], "Classes", r"$E_{p_i}$", error_boxplot, figsize = (skmodel.n_classes_, 4), annotate = False)
                
    
    
