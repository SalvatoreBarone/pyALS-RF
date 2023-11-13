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
import logging
from distutils.dir_util import mkpath
from .ctx_factory import load_configuration_ps, create_classifier
from .ConfigParsers.PsConfigParser import *
from .AxCT.LosslessHedgeTrimming import LosslessHedgeTrimming
from .AxCT.LossyHedgeTrimming import LossyHedgeTrimming

def pruning_flow(ctx : dict, use_training_data : bool, max_loss_perc : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
    
    if use_training_data:
        assert ctx.obj["configuration"].error_conf.training_dataset is not None, "You must provide a csv for the training dataset to use this command"
        logger.debug("Reading the traininig data set")
        ctx.obj["classifier"].read_training_set(ctx.obj["configuration"].error_conf.training_dataset)
        logger.debug(f"Read {len(ctx.obj['classifier'].x_train)} samples")
    else:
        logger.debug("Splitting the dataset to perform pruning")
        ctx.obj["classifier"].split_test_dataset()
        logger.debug(f"Pruning set: {len(ctx.obj['classifier'].x_val)} samples")
        logger.debug(f"Test set: {len(ctx.obj['classifier'].x_test)} samples")
        
    trimmer = LosslessHedgeTrimming(ctx.obj["classifier"]) if max_loss_perc is None else LossyHedgeTrimming(ctx.obj["classifier"], False, max_loss_perc)
    trimmer.redundancy_boxplot(f"{ctx.obj['configuration'].outdir}/redundancy_boxplot.pdf")
    trimmer.trim()
    trimmer.store(ctx.obj["configuration"].outdir)

    original = ctx.obj['classifier'].get_assertions_cost()
    after_pruning = ctx.obj['classifier'].get_pruned_assertions_cost()
    logger.info(f"Baseline accuracy: {trimmer.baseline_accuracy} %")
    logger.info(f"Accuracy: {trimmer.accuracy}.")
    logger.info(f"Loss: {trimmer.loss}")
    logger.info(f"Original cost (#literals): {original}") 
    logger.info(f"Prunable assertions: {len(trimmer.candidate_assertions)}")
    logger.info(f"Pruned assertiond {len(trimmer.pruned_assertions)}  ({len(trimmer.pruned_assertions) / len(trimmer.candidate_assertions) * 100}%)")
    logger.info(f"Pruned cost (#literals): {after_pruning} ({(1 - after_pruning / original) * 100})")
    trimmer.compare()
    
