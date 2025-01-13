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
from ..ctx_factory import load_configuration_ps, create_classifier
from ..ConfigParsers.PsConfigParser import *
from .GREP.GREP import GREP
from .GREP.ResiliencyBasedGREP import ResiliencyBasedGREP
from .GREP.LossBasedGREP import LossBasedGREP
from ..Model.Classifier import Classifier
from ..plot import boxplot
import os

def grep_flow(ctx : dict, fraction : float, approach: str, cost_criterion: str, minredundancy : int, maxloss : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
       
    trimmer = LossBasedGREP(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"]) if approach == "loss" else ResiliencyBasedGREP(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"])
    trimmer.trim(GREP.get_cost_criterion(cost_criterion))
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

""" Direction files:
    Such files contains for each leaf (present in a pruning configuration), i.e. the set of 0 ( right node ) or 1 ( left node) required to
    reach that specific leaf.
    In this way, other accellerators and software implementations can uniquely identify the leaf to prune and perform such operation. 
    Moreover, as different implementation can use different logics in terms of operator, this function associates each direction (0 or 1) with the operator
    of the node in question. For instance, if the node condition is greaterThan but the Node of the implementation supports only lerrOrEqual, then 
    the direction must be flipped ( 0 should become 1 and viceversa).
    Optionally ( depending on the value of gen_val_set) this function can generate the tree predictions of each single tree under the pruning configuration.
    In this way, the tree is pruned. 
"""
def pruning_into_directions(ctx, pruning_conf, gen_val_set, val_idx, ncpus, output):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Transforming the pruning configuration into directions file.")
    
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    out = ctx.obj['configuration'].outdir

        
    create_classifier(ctx)
    classifier : Classifier = ctx.obj["classifier"]
    with open(pruning_conf, "r") as f:
        pc = json5.load(f)
    logger.info(f"Initiating transformation...")
    direction_file_json = classifier.transform_assertion_into_directions(pc)
    out_path_directions = os.path.join(out, "leaf_pruning_directions.json5")
    logger.info(f"Dumping direction files...")
    with open(out_path_directions, "w") as f:
        json5.dump(direction_file_json, f, indent = 2)
    logger.info(f"Direction files dumped at {out_path_directions}")
    if gen_val_set: 
        logger.info("Generating prediction vectors.")
        out_pred_vecs_path = os.path.join(out, "axc_pred_vecs.json5")
        # Identify the x_set
        if val_idx is not None:
            x_set = classifier.x_test[val_idx]
        else:
            x_set = classifier.x_test
        # Generate all the testing set files. 
        GREP.set_pruning_conf(classifier, pc)
        logger.info(f"Starting predictions...")
        # Generate and dump the prediction vectors.
        pred_vectors = classifier.predict(x_test = x_set, disable_tqdm = False)   
        logger.info(f"Dumping predictions...")
        with open(out_pred_vecs_path, "w") as f:
            json5.dump(pred_vectors.tolist(), f, indent = 2)
        logger.info(f"Prediction Vectors dumped at {out_pred_vecs_path}")
