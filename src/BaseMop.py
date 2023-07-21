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
import copy
from multiprocessing import cpu_count, Pool
from pyalslib import list_partitioning
from .Classifier import *

class BaseMop:
    def __init__(self, classifier, ncpus):
        self.classifier = classifier
        self.ncpus = ncpus
        self.classifier.reset_assertion_configuration()
        self.classifier.reset_nabs_configuration()
        classifiers = [copy.deepcopy(classifier) for _ in range(ncpus)]
        self.baseline_accuracy = self.classifier.evaluate_test_dataset()
        print(f"Baseline accuracy: {self.baseline_accuracy} %")