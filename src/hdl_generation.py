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
import json5
from .HDLGenerators.PruningHdlGenerator import PruningHdlGenerator
from .HDLGenerators.PsHdlGenerator import PsHdlGenerator
from .HDLGenerators.SingleStepAlsHdlGenerator import SingleStepAlsHdlGenerator
from .HDLGenerators.SingleStepAlsWcHdlGenerator import SingleStepAlsWcHdlGenerator
from .HDLGenerators.SingleStepFullHdlGenerator import SingleStepFullHdlGenerator
from .HDLGenerators.TwoStepsAlsHdlGenerator import TwoStepsAlsHdlGenerator
from .HDLGenerators.TwoStepsAlsWcHdlGenerator import TwoStepsAlsWcHdlGenerator
from .HDLGenerators.TwoStepsFullHdlGenerator import TwoStepsFullHdlGenerator
from .ax_flows import load_configuration_ps, create_classifier, create_yshelper, load_flow

def hdl_generation(ctx):
    load_configuration_ps(ctx)
    create_classifier(ctx)
    create_yshelper(ctx)
    
    if ctx.obj["flow"] is None:
        load_flow(ctx)
    
    print("Generating the approximate implementation...")
    if ctx.obj["flow"] == "pruning":
        pruned_assertions_json = f"{ctx.obj['configuration'].outdir}/pruned_assertions.json5"
        if "pruned_assertions" not in ctx.obj:
            print(f"Reading pruning configuration from {pruned_assertions_json}")
            ctx.obj['pruned_assertions'] = json5.load(open(pruned_assertions_json))
        hdl_generator = PruningHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
        hdl_generator.generate_axhdl(pruned_assertions = ctx.obj['pruned_assertions'])
    elif ctx.obj["flow"] == "ps":
        hdl_generator = PsHdlGenerator(ctx.obj["classifier"], ctx.obj["yshelper"], ctx.obj['configuration'].outdir)
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
    
    print("Generating reference (non-approximate) implementation...")
    hdl_generator.generate_exact_implementation()
    print("All done!")