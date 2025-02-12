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
import os
import json5

def hdl_generation(ctx, lut_tech, skip_exact : bool, output, pruning_name):
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
    exact_luts_dbs, exact_luts_bns, exact_ffs_dbs = hdl_generator.get_resource_usage()
    logger.info("Exact implementations expected requirements (voting excluded):"
                f"\n\t- LUTs for decision boxes (exact): {exact_luts_dbs}"
                f"\n\t- FFs for decision boxes (exact): {exact_ffs_dbs}"
                f"\n\t- LUTs for Boolean Networks (exact): {exact_luts_bns}")
    if not skip_exact:
        logger.info("Generating reference (non-approximate) implementation...")
        logger.debug(f"Lut Tech: {lut_tech}")
        hdl_generator.generate_exact_implementation(enable_espresso =  ctx.obj['configuration'].outdir, lut_tech = lut_tech)
    
    logger.info("Generating the approximate implementation...")
    if ctx.obj["flow"] == "pruning":
        if not pruning_name:
            pruning_configuration_json = f"{ctx.obj['configuration'].outdir}/pruning_configuration.json5"
        else:
            pruning_configuration_json = f"{ctx.obj['configuration'].outdir}/{pruning_name}"

        if "pruning_configuration" not in ctx.obj:
            logger.info(f"Reading pruning configuration from {pruning_configuration_json}")
            ctx.obj['pruning_configuration'] = json5.load(open(pruning_configuration_json))
        hdl_generator = GREPHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
        hdl_generator.generate_axhdl(pruning_configuration = ctx.obj['pruning_configuration'], enable_espresso = ctx.obj['espresso'], lut_tech = lut_tech)
        ax_luts_dbs, ax_luts_bns, ax_ffs_dbs = hdl_generator.get_resource_usage()
        
        logger.info("Approximate implementations expected requirements (voting excluded):"
                    f"\n\t- LUTs for decision boxes (approx.): {ax_luts_dbs}"
                    f"\n\t- FFs for decision boxes (approx.): {ax_ffs_dbs}"
                    f"\n\t- LUTs for Boolean Networks (approx.): {ax_luts_bns}")
        logger.info(f"Expected LUT savings for BNs: {(1 - ax_luts_bns / exact_luts_bns) * 100}%"
                    f"\n\tExpected LUT savings for DBs: {(1 - ax_luts_dbs / exact_luts_dbs) * 100}%"
                    f"\n\tExpected FFs savings for DBs: {(1 - ax_ffs_dbs / exact_ffs_dbs) * 100}%")
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
        #ax_luts_dbs, exact_lut_bns, ax_ffs_dbs, exact_lut_dbs, exact_ffs_dbs = hdl_generator.get_resource_usage()
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

def hdl_resource_usage(ctx, pruning_cfg_path : str = None, ps_set_configuration_path: str = None, report_path: str = None):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the HDL generation flow.")
    if pruning_cfg_path != None and ps_set_configuration_path != None:
        assert 1 == 0, "Dual AxC Cfg not yet supported !"
    if ps_set_configuration_path != None and not os.path.exists(ps_set_configuration_path):
        logger.error("Invalid path for precision scaling cfg")
        assert 1 == 0
    if pruning_cfg_path != None and not os.path.exists(pruning_cfg_path):
        logger.error("Invalid path for pruning configuration")
        assert 1 == 0
    load_configuration_ps(ctx)
    create_classifier(ctx)
    create_yshelper(ctx)
    # # Generate the configuration for the exact classifier.
    # hdl_generator = HDLGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
    # exact_luts_dbs, exact_luts_bns, exact_ffs_dbs = hdl_generator.get_resource_usage()
    # logger.info("Exact implementations expected requirements (voting excluded):"
    #             f"\n\t- LUTs for decision boxes (exact): {exact_luts_dbs}"
    #             f"\n\t- FFs for decision boxes (exact): {exact_ffs_dbs}"
    #             f"\n\t- LUTs for Boolean Networks (exact): {exact_luts_bns}")
    if  ps_set_configuration_path  != None:
        with open(ps_set_configuration_path, 'r') as f:
            pareto_set = json5.load(f)
        for cfg_id, cfg in enumerate(pareto_set):
            loss = cfg['f']
            confs = cfg['x']
            nabs = {f["name"]: n for f, n in zip(ctx.obj['classifier'].model_features, confs)}
            ps_ax_hdl_generator = PsHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
            ctx.obj['classifier'].set_nabs(nabs)
            nLUTs_dbs, nLUTs_bns, nFFs_dbs, nLUTs_dbs_exact, nFFs_dbs_exact = ps_ax_hdl_generator.get_resource_usage_custom()
            dbs_lut_savings = ( (nLUTs_dbs_exact - nLUTs_dbs) / nLUTs_dbs_exact) * 100.0
            dbs_ffs_savings = ( (nFFs_dbs_exact - nFFs_dbs) / nFFs_dbs_exact) * 100.0
            total_lut_exact = nLUTs_dbs_exact + nLUTs_bns
            total_luts_ax = nLUTs_dbs + nLUTs_bns
            total_luts_savings = (1 - total_luts_ax /total_lut_exact) * 100.0
            logger.info("Approximate implementations expected requirements (voting excluded):"
                f"\n\t- LUTs for decision boxes (PS): {nLUTs_dbs}"
                f"\n\t- FFs for decision boxes (PS): {nFFs_dbs}"
                f"\n\t- LUTs for Boolean Networks (PS): {nLUTs_bns}")
            logger.info("Exact implementations expected requirements (voting excluded):"
                f"\n\t- LUTs for decision boxes (PS): {nLUTs_dbs_exact}"
                f"\n\t- FFs for decision boxes (PS): {nFFs_dbs_exact}"
                f"\n\t- LUTs for Boolean Networks (PS): {nLUTs_bns}")
            logger.info(f"Savings DBS: LUTS: {dbs_lut_savings} FFS: {dbs_ffs_savings} Total LUTS: {total_luts_savings} Total FFS: {dbs_ffs_savings}")
            print(confs)
    elif pruning_cfg_path != None:
        pass
    else:
        pass