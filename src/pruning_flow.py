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
import json5, os
from distutils.dir_util import mkpath
from .ConfigParsers.PsConfigParser import *
from .Model.Classifier import *
from .ax_flows import load_configuration_ps, create_classifier

def pruning_flow(ctx, use_training_data, output):
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    
    create_classifier(ctx)
    
    if use_training_data:
        assert ctx.obj["configuration"].training_dataset is not None, "You must provide a csv for the training dataset to use this command"
        print("Reading the traininig data set")
        ctx.obj["classifier"].read_training_set(ctx.obj["configuration"].training_dataset)
        print(f"Read {len(ctx.obj['classifier'].x_train)} samples")
    else:
        print("Splitting the dataset to perform pruning")
        ctx.obj["classifier"].split_test_dataset()
        print(f"Pruning set: {len(ctx.obj['classifier'].x_val)} samples")
        print(f"Test set: {len(ctx.obj['classifier'].x_test)} samples")
        
    
    print("Computing the baseline accuracy...")
    baseline_accuracy = ctx.obj["classifier"].evaluate_test_dataset()
    print(f"Baseline accuracy: {baseline_accuracy} %")
    # active_assertions_json = f"{ctx.obj['configuration'].outdir}/active_assertion.json5"
    # redundancy_json = f"{ctx.obj['configuration'].outdir}/redundancy.json5"
    # pruning_json = f"{ctx.obj['configuration'].outdir}/pruning.json5"
    pruned_assertions_json = f"{ctx.obj['configuration'].outdir}/pruned_assertions.json5"
    # if os.path.exists(active_assertions_json) and os.path.exists(redundancy_json) and os.path.exists(pruning_json):
    #     print("Reading pruning from JSON files...")
    #     active_assertions = json5.load(open(active_assertions_json))
    #     redundancy_table = json5.load(open(redundancy_json))
    #     pruning_table = json5.load(open(pruning_json))
    # else:
    active_assertions, redundancy_table, pruning_table = get_pruning_table(ctx.obj["classifier"], use_training_data)
        # with open(active_assertions_json, "w") as f:
        #     json5.dump(active_assertions, f, indent=2)
        # with open(redundancy_json, "w") as f:
        #     json5.dump(redundancy_table, f, indent=2)
        # with open(pruning_json, "w") as f:
        #     json5.dump(pruning_table, f, indent=2)
        
    # hist = redundancy_histogram(redundancy_table)
    # threshold = int(np.ceil( len(ctx.obj["classifier"].trees) / 2 ))
    # print(f"Trees: {len(ctx.obj['classifier'].trees)}, threshold: {threshold}")
    # print("Redundancy:")
    # for k, v in hist.items():
    #     print(f"{k}: {v}%")
   
    total_cost, candidate_assertions, pruned_assertions = lossless_hedge_trimming(redundancy_table, pruning_table)
    savings = sum( i[3] for i in pruned_assertions ) / total_cost * 100
    print(f"Prunable assertions: {len(candidate_assertions)}")
    print(f"Original cost (#literals): {total_cost}") 
    print(f"Pruned assertiond {len(pruned_assertions)}")
    print(f"Savings (%): {savings}")
    
    acc = ctx.obj["classifier"].test_pruning(pruned_assertions)
    print(f"Loss: {baseline_accuracy - acc}")
    ctx.obj['pruned_assertions'] = pruned_assertions
    with open(pruned_assertions_json, "w") as f:
        json5.dump(pruned_assertions, f, indent=2)

def get_pruning_table(classifier, use_training_data):
    active_assertions = classifier.get_assertion_activation(use_training_data)
    redundancy_table = {}
    pruning_table = { c : {t.name : {} for t in classifier.trees } for c in classifier.model_classes }
    for m in active_assertions:
        for tree, path in m["outcomes"].items():
            if path["correct"]:
                if path["assertion"] not in pruning_table[m["y"]][tree]:
                    pruning_table[m["y"]][tree][path["assertion"]] = []
                sample_id = ';'.join([str(x) for x in m["x"]])
                pruning_table[m["y"]][tree][path["assertion"]].append(sample_id)
                redundancy_table[sample_id] = m["redundancy"]
    return active_assertions, redundancy_table, pruning_table

def redundancy_histogram(redundancy_table):
    hist = {}
    for r in redundancy_table.values():
        if r not in hist:
            hist[r] = 0
        hist[r] += 1
    for k in hist:
        hist[k] = hist[k] * 100 / len(redundancy_table)
    return hist

def lossless_approximable_assertions(redundancy_table, pruning_table):
    assertion_cost = []
    total_cost = 0
    for class_label, trees in pruning_table.items():
        for tree_name, assertions in trees.items():
            for assertion, samples in assertions.items():
                approximable = all([ redundancy_table[sample] > 0 for sample in samples ])
                literals = len(assertion.split("and"))
                total_cost += literals
                if approximable:
                    assertion_cost.append((class_label, tree_name, assertion, literals / len(samples)) )
    assertion_cost.sort(key=lambda x: x[3], reverse = True)
    return total_cost, assertion_cost

def lossless_hedge_trimming(redundancy_table, pruning_table):
    total_cost, candidate_assertions = lossless_approximable_assertions(redundancy_table, pruning_table)
    pruned_assertions = []
    for class_label, tree_name, assertion, cost in candidate_assertions:
        samples = pruning_table[class_label][tree_name][assertion]
        approximable = all([ redundancy_table[sample] > 0 for sample in samples ])
        if approximable:
            for sample in samples:
                redundancy_table[sample] -= 1
            pruned_assertions.append((class_label, tree_name, assertion, cost))
    return total_cost, candidate_assertions, pruned_assertions