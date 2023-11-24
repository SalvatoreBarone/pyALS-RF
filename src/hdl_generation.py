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
import json5, logging
from distutils.dir_util import mkpath
from pyamosa.Pareto import Pareto
from .HDLGenerators.HDLGenerator import HDLGenerator
from .HDLGenerators.GREPHdlGenerator import GREPHdlGenerator
from .HDLGenerators.PsHdlGenerator import PsHdlGenerator
from .HDLGenerators.SingleStepAlsHdlGenerator import SingleStepAlsHdlGenerator
from .HDLGenerators.SingleStepAlsWcHdlGenerator import SingleStepAlsWcHdlGenerator
from .HDLGenerators.SingleStepFullHdlGenerator import SingleStepFullHdlGenerator
from .HDLGenerators.TwoStepsAlsHdlGenerator import TwoStepsAlsHdlGenerator
from .HDLGenerators.TwoStepsAlsWcHdlGenerator import TwoStepsAlsWcHdlGenerator
from .HDLGenerators.TwoStepsFullHdlGenerator import TwoStepsFullHdlGenerator
from .ctx_factory import load_configuration_ps, create_classifier, create_yshelper, load_flow, create_problem, create_optimizer

def hdl_generation(ctx, lut_tech, skip_exact : bool, output):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the HDL generation flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)
    create_yshelper(ctx)
    
    if ctx.obj["flow"] is None:
        load_flow(ctx)
        
    hdl_generator = HDLGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    exact_luts, exact_ffs = hdl_generator.get_resource_usage()
    if not skip_exact:
        logger.info("Generating reference (non-approximate) implementation...")
        logger.debug(f"Lut Tech: {lut_tech}")
        hdl_generator.generate_exact_implementation(enable_espresso =  ctx.obj['configuration'].outdir, lut_tech = lut_tech)
    
    logger.info("Generating the approximate implementation...")
    if ctx.obj["flow"] == "pruning":
        pruning_configuration_json = f"{ctx.obj['configuration'].outdir}/pruning_configuration.json5"
        if "pruning_configuration" not in ctx.obj:
            logger.info(f"Reading pruning configuration from {pruning_configuration_json}")
            ctx.obj['pruning_configuration'] = json5.load(open(pruning_configuration_json))
        hdl_generator = GREPHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
        hdl_generator.generate_axhdl(pruning_configuration = ctx.obj['pruning_configuration'], enable_espresso = ctx.obj['espresso'], lut_tech = lut_tech)
        ax_luts, ax_ffs = hdl_generator.get_resource_usage()
        logger.info(f"Expected LUT savings: {(1 - ax_luts / exact_luts) * 100}%\n\tExpected FFs savings: {(1 - ax_ffs / exact_ffs) * 100}%")
    elif ctx.obj["flow"] == "ps":
        if "pareto_front" not in ctx.obj:
            create_problem(ctx, mode = "full")
            create_optimizer(ctx)
            pareto_front_json = f"{ctx.obj['configuration'].outdir}/final_archive.json"
            print(f"Reading pareto front from {pareto_front_json}.")
            ctx.obj["optimizer"].archive = Pareto()
            ctx.obj["optimizer"].archive.read_json(ctx.obj["problem"].types, pareto_front_json)
            ctx.obj["pareto_front"] = ctx.obj["optimizer"].archive
        hdl_generator = PsHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
        hdl_generator.generate_axhdl(pareto_set = ctx.obj['pareto_front'].get_set(), enable_espresso = ctx.obj['espresso'], lut_tech = lut_tech)
    elif ctx.obj["flow"] == "als-onestep":
        hdl_generator = SingleStepAlsHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    elif ctx.obj["flow"] == "als-twosteps":
        hdl_generator = TwoStepsAlsHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    elif ctx.obj["flow"] == "wcals-onestep":
        hdl_generator = SingleStepAlsWcHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    elif ctx.obj["flow"] == "wcals-twosteps":
        hdl_generator = TwoStepsAlsWcHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    elif ctx.obj["flow"] == "full-onestep":
        hdl_generator = SingleStepFullHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    elif ctx.obj["flow"] == "full-twosteps":
        hdl_generator = TwoStepsFullHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    else:
        print(f"{ctx.obj['flow']}: unrecognized approximation flow. Bailing out.")
        exit()
    
    
    logger.info("All done!")