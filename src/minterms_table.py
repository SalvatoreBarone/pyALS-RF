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

def get_minterms_table(configfile, outfile, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(ncpus)
    classifier.parse(configuration.pmml)
    classifier.read_dataset(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    minterms = classifier.get_mintems()
    with open(outfile, "w") as f:
        json5.dump(minterms, f)