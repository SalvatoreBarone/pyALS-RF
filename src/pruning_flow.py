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
from .AxCT.HedgeTrimming import HedgeTrimming
from .AxCT.ResiliencyBasedHedgeTrimming import ResiliencyBasedHedgeTrimming
from .AxCT.LossBasedHedgeTrimming import LossBasedHedgeTrimming



def pruning_flow(ctx : dict, fraction : float, cost_criterion: str, minredundancy : int, maxloss : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
       
    trimmer = LossBasedHedgeTrimming(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"]) if minredundancy == 0 else ResiliencyBasedHedgeTrimming(ctx.obj["classifier"], fraction, maxloss, minredundancy, ctx.obj["ncpus"])
    trimmer.trim(HedgeTrimming.get_cost_criterion(cost_criterion))
    trimmer.store_pruning_conf(f"{ctx.obj['configuration'].outdir}/pruning_configuration.json5")
    trimmer.redundancy_boxplot(f"{ctx.obj['configuration'].outdir}/redundancy_boxplot.pdf")
    trimmer.restore_bns()
    
    
    
