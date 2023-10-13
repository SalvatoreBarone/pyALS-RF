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
from .ConfigParsers.PsConfigParser import *
from .Model.Classifier import *
from .ax_flows import load_configuration_ps, create_classifier, create_yshelper

def none_hdl_debug_flow(ctx, index, output):
    ctx.obj["classifier"].predict_dump(index, output)

def pruning_hdl_debug_flow(ctx, index, results, output):
    if results is not None:
        ctx.obj['configuration'].outdir = results
        
    pruned_assertions_json = f"{ctx.obj['configuration'].outdir}/pruned_assertions.json5"
    if "pruned_assertions" not in ctx.obj:
        print(f"Reading pruning configuration from {pruned_assertions_json}")
        ctx.obj['pruned_assertions'] = json5.load(open(pruned_assertions_json))
    ctx.obj["classifier"].set_pruning(ctx.obj['pruned_assertions'])
    ctx.obj["classifier"].predict_dump(index, output, True)
    

def ps_hdl_debug_flow(ctx, index, results, variant, output):
    pass


def hdl_debug_flow(ctx, index, axflow, results, variant, output):
    load_configuration_ps(ctx)
    create_classifier(ctx)
    create_yshelper(ctx)
    ctx.obj["classifier"].predict_dump(index, output)
    if axflow == "none":
        none_hdl_debug_flow(ctx, index, output)
    elif axflow == "pruning":
        pruning_hdl_debug_flow(ctx, index, results, output)
    elif  axflow == "ps":
        ps_hdl_debug_flow(ctx, index, results, variant, output)
    