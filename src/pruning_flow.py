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
from distutils.dir_util import mkpath
from .ax_flows import load_configuration_ps, create_classifier
from .ConfigParsers.PsConfigParser import *
from .AxCT.LosslessHedgeTrimming import LosslessHedgeTrimming
from .AxCT.LossyHedgeTrimming import LossyHedgeTrimming

def pruning_flow(ctx : dict, use_training_data : bool, max_loss_perc : float, output : str):
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
    
    if use_training_data:
        assert ctx.obj["configuration"].error_conf.training_dataset is not None, "You must provide a csv for the training dataset to use this command"
        print("Reading the traininig data set")
        ctx.obj["classifier"].read_training_set(ctx.obj["configuration"].error_conf.training_dataset)
        print(f"Read {len(ctx.obj['classifier'].x_train)} samples")
    else:
        print("Splitting the dataset to perform pruning")
        ctx.obj["classifier"].split_test_dataset()
        print(f"Pruning set: {len(ctx.obj['classifier'].x_val)} samples")
        print(f"Test set: {len(ctx.obj['classifier'].x_test)} samples")
        
    trimmer = LosslessHedgeTrimming(ctx.obj["classifier"]) if max_loss_perc is None else LossyHedgeTrimming(ctx.obj["classifier"], False, max_loss_perc)
    trimmer.trim()
    trimmer.store(ctx.obj["configuration"].outdir)

    original = ctx.obj['classifier'].get_assertions_cost()
    after_pruning = ctx.obj['classifier'].get_pruned_assertions_cost()
    print(f"Prunable assertions: {len(trimmer.candidate_assertions)}")
    print(f"Original cost (#literals): {original}") 
    print(f"Pruned assertiond {len(trimmer.pruned_assertions)}  ({len(trimmer.pruned_assertions) / len(trimmer.candidate_assertions) * 100}%)")
    print(f"Pruned cost (#literals): {after_pruning} ({(1 - after_pruning / original) * 100})") 
    print(f"Loss: {trimmer.loss}")

