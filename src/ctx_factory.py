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
import os, pyamosa, json5, logging
from pyalslib import YosysHelper, check_for_file
from distutils.dir_util import mkpath
from .logger import configure_logger
from .ConfigParsers.PsConfigParser import *
from .Model.Classifier import *
from .Optimization.PsMop import *

def set_global_options(ctx, confifile, logger_name, verbosity, ncpus, use_espresso, flow = None):
    #assert "flow" not in ctx.obj, f"Approximation flow already set ({ctx.obj['flow']}). You issued more than one approximation command. Bailing out."
    if "logger" not in ctx.obj:
        ctx.obj["logger"] = logger_name
        configure_logger(logger_name, verbosity)
    ctx.obj["flow"] = flow
    if "configfile" not in ctx.obj:
        ctx.obj["configfile"] = confifile
    if "ncpus" not in ctx.obj:
        ctx.obj["ncpus"] = ncpus
    if "espresso" not in ctx.obj:
        ctx.obj["espresso"] = use_espresso

def create_yshelper(ctx):
    if "yshelper" not in ctx.obj:
        assert "logger" in ctx.obj, "Logger not set"
        logger = logging.getLogger(ctx.obj["logger"])
        logger.info("Creating yshelper")
        ctx.obj["yshelper"] = YosysHelper()
        ctx.obj["yshelper"].load_ghdl()
        
def load_configuration_ps(ctx):
    if "configuration" not in ctx.obj:
        assert "configfile" in ctx.obj, "You must provide a JSON configuration file to run this command(s)"
        ctx.obj["configuration"] = PSConfigParser(ctx.obj["configfile"])
        check_for_file(ctx.obj["configuration"].model_source)
        check_for_file(ctx.obj["configuration"].error_conf.test_dataset) 
        if ctx.obj["configuration"].outdir != ".":
            mkpath(ctx.obj["configuration"].outdir)
    else:
        assert isinstance(ctx.obj["configuration"], PSConfigParser), "Unsuitable configuration file"
        
def load_flow(ctx):
    assert "logger" in ctx.obj, "Logger not set"
    logger = logging.getLogger(ctx.obj["logger"])
    logger.info(f"Loading approximation flow from {ctx.obj['configuration'].outdir}/.flow.json5")
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    ctx.obj["flow"] = json5.load(open(f"{ctx.obj['configuration'].outdir}/.flow.json5"))
    
def store_flow(ctx):
    assert "configuration" in ctx.obj, "No configuration. Bailing out."
    with open(f"{ctx.obj['configuration'].outdir}/.flow.json5", "w") as f:
        json5.dump(ctx.obj["flow"], f, indent=2)
        
def create_classifier(ctx):
    if "classifier" not in ctx.obj:
        assert "ncpus" in ctx.obj, "No setting available for 'ncpus'. Bailing out!"
        assert "configuration" in ctx.obj, "No configuration loaded. Bailing out!"
        ctx.obj["classifier"] = Classifier(ctx.obj["ncpus"], ctx.obj["espresso"])
        ctx.obj["classifier"].parse(ctx.obj["configuration"].model_source, ctx.obj["configuration"].error_conf.dataset_description)
        ctx.obj["classifier"].read_test_set(ctx.obj["configuration"].error_conf.test_dataset)
        
def create_alsgraph(ctx):
    pass
    # if "graph" not in ctx.obj:
    #     assert "configuration" in ctx.obj, "You must read the JSON configuration file to run this command(s)"
    #     assert "yshelper" in ctx.obj, "You must create a YosysHelper object first"
    #     print("Graph generation...")
    #     ctx.obj["yshelper"].read_sources(ctx.obj["configuration"].source_hdl, ctx.obj["configuration"].top_module)
    #     ctx.obj["yshelper"].prep_design(ctx.obj["configuration"].als_conf.cut_size)
    #     ctx.obj["graph"] = ALSGraph(ctx.obj["yshelper"].design)
    #     ctx.obj["yshelper"].save_design("original")
    #     print("Done!")
        
def create_catalog(ctx):
    pass
    # if "catalog" not in ctx.obj:
    #     assert "configuration" in ctx.obj, "You must read the JSON configuration file to run this command(s)"
    #     assert "yshelper" in ctx.obj, "You must create a YosysHelper object first"
    #     assert "graph" in ctx.obj, "You must create a ALSGraph object first"
    #     ctx.obj["luts_set"] = ctx.obj["yshelper"].get_luts_set()
    #     print(f"Performing catalog generation using {ctx.obj['ncpus']} threads. Please wait patiently. This may take time.")
    #     ctx.obj["catalog"] = ALSCatalog(ctx.obj["configuration"].als_conf.lut_cache, ctx.obj["configuration"].als_conf.solver).generate_catalog(ctx.obj["luts_set"], ctx.obj["configuration"].als_conf.timeout, ctx.obj['ncpus'])
    #     print("Done!")
              
def create_problem(ctx, **kwargs):
    assert "flow" in ctx.obj, "Unspecified approximation flow. Bailing out."
    assert "classifier" in ctx.obj, "Unspecified classifier model. Bailing out."
    if "problem" not in ctx.obj:
        if ctx.obj["flow"] == "ps":
            ctx.obj["problem"] = PsMop(ctx.obj["classifier"], ctx.obj["configuration"].error_conf.max_loss_perc, ctx.obj["ncpus"]) if kwargs['mode'] == "full" else RankBasedPsMop(ctx.obj["classifier"], ctx.obj["configuration"].error_conf.max_loss_perc, kwargs['alpha'], kwargs['beta'], kwargs['gamma'], ctx.obj["ncpus"])
    
    # assert "configuration" in ctx.obj, "You must read the JSON configuration file to run this command(s)"
    # assert "graph" in ctx.obj, "You must create a ALSGraph object first"
    # if "problem" not in ctx.obj:
    #     ctx.obj["problem"] = MOP(ctx.obj["configuration"].top_module, ctx.obj["graph"], ctx.obj["output_weights"], ctx.obj["catalog"], ctx.obj["configuration"].error_conf, ctx.obj["configuration"].hw_conf, ctx.obj['ncpus']) if ctx.obj['dataset'] is None else IAMOP(ctx.obj["configuration"].top_module, ctx.obj["graph"], ctx.obj["output_weights"], ctx.obj["catalog"], ctx.obj["configuration"].error_conf, ctx.obj["configuration"].hw_conf, ctx.obj['dataset'], ctx.obj['ncpus'])
        
def create_optimizer(ctx):
    if "optimizer" not in ctx.obj:
        print("Creating optimizer...")
        grp = ctx.obj["configuration"].variable_grouping_strategy 
        tso = ctx.obj["configuration"].transfer_strategy_objectives
        tsv = ctx.obj["configuration"].transfer_strategy_variables
        if grp is None:
            ctx.obj["optimizer"] = pyamosa.Optimizer(ctx.obj["configuration"].amosa_conf)
        elif grp in ["DRG", "drg", "random"]:
            print("Using dynamic random grouping")
            ctx.obj["optimizer"] = pyamosa.DynamicRandomGroupingOptimizer(ctx.obj["configuration"].amosa_conf)
        elif grp in ["dvg", "DVG", "dvg2", "DVG2", "differential"]:
            print(f"Using differential grouping with TSO {tso} and TSV {tsv}")
            variable_decomposition_cache = f"{ctx.obj['configuration'].output_dir}/dvg2_{tso}_{tsv}.json5"
            grouper = pyamosa.DifferentialVariableGrouping2(ctx.obj["problem"])
            if os.path.exists(variable_decomposition_cache):
                grouper.load(variable_decomposition_cache)
            else:
                grouper.run(ctx.obj["configuration"].tso_selector[tso], ctx.obj["configuration"].tsv_selector[tsv])
                grouper.store(variable_decomposition_cache)
            ctx.obj["optimizer"] = pyamosa.GenericGroupingOptimizer(ctx.obj["configuration"], grouper)
        ctx.obj["final_archive_json"] = f"{ctx.obj['configuration'].output_dir}/final_archive.json"
        
def improve(ctx):
    if "improve" not in ctx.obj and os.path.exists(ctx.obj["final_archive_json"]):
        print("Using results from previous runs as a starting point.")
        ctx.obj["improve"] = ctx.obj["final_archive_json"]