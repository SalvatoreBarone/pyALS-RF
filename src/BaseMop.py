"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

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
import copy
from multiprocessing import cpu_count, Pool
from pyalslib import list_partitioning
from .Classifier import *

def evaluate_preloaded_dataset(classifier, samples):
    return classifier.evaluate_preloaded_dataset(samples)

def evaluate_preloaded_dataset_noals(classifier, samples):
    return classifier.evaluate_preloaded_dataset_noals(samples)

def evaluate_eprob(graph, samples, configuration):
    lut_info = {}
    return sum(0 if sample["output"] == graph.evaluate(sample["input"], lut_info, configuration)[0] else 1 for sample in samples)

class BaseMop:
    def __init__(self, classifier, dataset_csv, ncpus):
        self.classifier = classifier
        self.features = self.classifier.get_features()
        self.dataset = classifier.preload_dataset(dataset_csv)
        self.n_samples = len(self.dataset)
        self.ncpus = ncpus
        classifier.reset_assertion_configuration()
        classifier.reset_nabs_configuration()
        classifiers = [copy.deepcopy(classifier) for _ in range(ncpus)]
        self.args = [[c, d] for c, d in zip(classifiers, list_partitioning(self.dataset, ncpus))]
        self.baseline_accuracy = self.evaluate_dataset()
        print(f"Baseline accuracy: {self.baseline_accuracy} %")

    def evaluate_dataset(self):
        with Pool(self.ncpus) as pool:
            res = pool.starmap(evaluate_preloaded_dataset, self.args)
        return sum(res) * 100 / self.n_samples
    
    def evaluate_dataset_noals(self):
        with Pool(self.ncpus) as pool:
            res = pool.starmap(evaluate_preloaded_dataset_noals, self.args)
        return sum(res) * 100 / self.n_samples

    def get_accuracy_loss(self):
        return self.baseline_accuracy - self.evaluate_dataset()
    
    def get_accuracy_loss_noals(self):
        return self.baseline_accuracy - self.evaluate_dataset_noals()
