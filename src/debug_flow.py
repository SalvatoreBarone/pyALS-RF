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
import joblib
from tabulate import tabulate
from .ConfigParsers.PsConfigParser import *
from .Model.Classifier import *
from .ctx_factory import load_configuration_ps, create_classifier, create_yshelper
from .dtgen import print_nodes

def debug_with_scikit(ctx, output):
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
    create_classifier(ctx)
    dump_file = f"{ctx.obj['configuration'].outdir}/classifier.joblib"
    model = joblib.load(dump_file)
    classifier = ctx.obj["classifier"]
    acc_pyals = 0
    acc_scikit = 0
    mismatches = []
    for x, y in tqdm(zip(classifier.x_test, classifier.y_test), total = len(classifier.y_test), desc="Testing accuracy...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
        rho_pyals = classifier.get_score(x)
        rho_scikit = model.predict_proba(np.array(x).reshape((1, -1)))
        assert all(i == int(j) for i, j in zip(rho_pyals, rho_scikit[0])), f"Error in model response: {rho_pyals} {rho_scikit}"
        outcome_pyals, draw_pyals = classifier.predict(x)
        draw_scikit, _ = classifier.check_draw(rho_scikit[0].tolist())
        if np.argmax(rho_scikit) == y and not draw_scikit:
            acc_scikit += 1
        if np.argmax(outcome_pyals) == y:
            acc_pyals += 1
        if (np.argmax(outcome_pyals) != np.argmax(rho_scikit)) and (draw_pyals != draw_scikit):
            mismatches.append((', '.join(str(s) for s in rho_pyals), draw_pyals, ', '.join(str(s) for s in outcome_pyals), np.argmax(outcome_pyals), ', '.join(f'{q:.2f}' for q in rho_scikit[0]), np.argmax(rho_scikit), y))
            
    print(tabulate(mismatches, headers=["Score", "Draw", "Outcome", "argmax", "Scikit Rho", "argmax", "Label"]))
    print(f"{len(mismatches)} mismatches")
    print(f"Accuracy pyALS : {acc_pyals / len(classifier.y_test)}")
    print(f"Accuracy scikit : {acc_scikit / len(classifier.y_test)}")

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
    
def print_model(dump_file, pmml_file):
    model = joblib.load(dump_file)
    print_nodes(model)
    if pmml_file is not None:
        from sklearn2pmml.pipeline import PMMLPipeline
        from sklearn2pmml import sklearn2pmml   
        pipeline = PMMLPipeline([("classifier", model)])
        sklearn2pmml(pipeline, pmml_file, with_repr = True)
    
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
    