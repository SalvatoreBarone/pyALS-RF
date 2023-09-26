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

def pruning_flow(configfile, active_minterms_json, pruning_table_json, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(ncpus)
    classifier.parse(configuration.pmml)
    classifier.read_dataset(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    active_minterms, pruning_table = get_pruning_table(classifier)
    hist = redundancy_histogram(active_minterms)
    threshold = int(np.ceil( len(classifier.trees) / 2 ))
    print(f"Trees: {len(classifier.trees)}, threshold: {threshold}")
    print("Redundancy:")
    for k, v in hist.items():
        print(f"{k}: {v}%")
    with open(active_minterms_json, "w") as f:
        json5.dump(active_minterms, f, indent=2)
    with open(pruning_table_json, "w") as f:
        json5.dump(pruning_table, f, indent=2)

def get_pruning_table(classifier):
    active_minterms = classifier.get_mintems()
    print(active_minterms)
    pruning_table = { c : {t.name : {} for t in classifier.trees } for c in classifier.model_classes }
    for m in active_minterms:
        for tree, path in m["outcomes"].items():
            if path["correct"]:
                if path["minterm"] not in pruning_table[m["y"]][tree]:
                    pruning_table[m["y"]][tree][path["minterm"]] = {}
                sample_id = ';'.join([str(x) for x in m["x"]])
                pruning_table[m["y"]][tree][path["minterm"]][sample_id] = m["redundancy"]
    print(pruning_table)
    return active_minterms, pruning_table

def redundancy_histogram(active_minterms):
    hist = {}
    for m in active_minterms:
        if m["redundancy"] not in hist:
            hist[m["redundancy"]] = 0
        hist[m["redundancy"]] += 1
    for k in hist:
        hist[k] = hist[k] * 100 / len(active_minterms)
    return hist