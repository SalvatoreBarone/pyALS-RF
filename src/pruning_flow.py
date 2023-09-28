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
from pyalslib import check_for_file
from multiprocessing import cpu_count
from .PsConfigParser import *
from .Classifier import *
import json5

def pruning_flow(configfile, active_assertions_json, redundancy_json, pruning_json, relative_cost, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(ncpus)
    classifier.parse(configuration.pmml)
    classifier.read_dataset(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    classifier.reset_nabs_configuration()
    classifier.reset_assertion_configuration()
    print("Computing the baseline accuracy...")
    baseline_accuracy = classifier.evaluate_test_dataset()
    print(f"Baseline accuracy: {baseline_accuracy} %")
    
    if os.path.exists(active_assertions_json) and os.path.exists(redundancy_json) and os.path.exists(pruning_json):
        print("Reading pruning from JSON files...")
        active_assertions = json5.load(open(active_assertions_json))
        redundancy_table = json5.load(open(redundancy_json))
        pruning_table = json5.load(open(pruning_json))
    else:
        active_assertions, redundancy_table, pruning_table = get_pruning_table(classifier)
        with open(active_assertions_json, "w") as f:
            json5.dump(active_assertions, f, indent=2)
        with open(redundancy_json, "w") as f:
            json5.dump(redundancy_table, f, indent=2)
        with open(pruning_json, "w") as f:
            json5.dump(pruning_table, f, indent=2)
        
    hist = redundancy_histogram(redundancy_table)
    threshold = int(np.ceil( len(classifier.trees) / 2 ))
    print(f"Trees: {len(classifier.trees)}, threshold: {threshold}")
    print("Redundancy:")
    for k, v in hist.items():
        print(f"{k}: {v}%")
   
    total_cost, candidate_assertions, pruned_assertions = lossless_hedge_trimming(redundancy_table, pruning_table, relative_cost)
    savings = sum( i[3] for i in pruned_assertions ) / total_cost
    print(len(candidate_assertions), len(pruned_assertions), savings)
    
    acc = classifier.test_pruning(pruned_assertions)
    print(f"Loss: {baseline_accuracy - acc}")
    

def get_pruning_table(classifier):
    active_assertions = classifier.get_mintems()
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

def lossless_approximable_assertions(redundancy_table, pruning_table, relative_cost):
    assertion_cost = []
    total_cost = 0
    for class_label, trees in pruning_table.items():
        for tree_name, assertions in trees.items():
            for assertion, samples in assertions.items():
                approximable = all([ redundancy_table[sample] > 0 for sample in samples ])
                literals = len(assertion.split("and"))
                total_cost += literals
                if approximable:
                    assertion_cost.append((class_label, tree_name, assertion, literals / len(samples) if relative_cost else literals ) )
    assertion_cost.sort(key=lambda x: x[3], reverse = True)
    return total_cost, assertion_cost

def lossless_hedge_trimming(redundancy_table, pruning_table, relative_cost):
    total_cost, candidate_assertions = lossless_approximable_assertions(redundancy_table, pruning_table, relative_cost)
    pruned_assertions = []
    for class_label, tree_name, assertion, cost in candidate_assertions:
        samples = pruning_table[class_label][tree_name][assertion]
        approximable = all([ redundancy_table[sample] > 0 for sample in samples ])
        if approximable:
            for sample in samples:
                redundancy_table[sample] -= 1
            pruned_assertions.append((class_label, tree_name, assertion, cost))
    return total_cost, candidate_assertions, pruned_assertions