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
import json
from .ConfigParser import *
from pyamosa import Optimizer
from pyalslib import ALSConfig
from .ErrorConfig import ErrorConfig
class TwoStepsConfigParser:
    def __init__(self, configfile):
        configuration = json.load(open(configfile))

        self.pmml = search_field_in_config(configuration, "model", True)
        self.outdir = search_field_in_config(configuration, "outdir", True)

        self.error_conf = ErrorConfig(
            max_loss_perc = search_subfield_in_config(configuration, "error", "max_loss_perc", True),
            test_dataset = search_subfield_in_config(configuration, "error", "test_dataset", True),
            max_eprob = search_subfield_in_config(configuration, "error", "max_eprob_perc", True),
            nvectors = search_subfield_in_config(configuration, "error", "nvectors", True),
            dataset = search_subfield_in_config(configuration, "error", "dataset", False))

        self.als_conf = ALSConfig(
            lut_cache = search_subfield_in_config(configuration, "als", "cache", True),
            cut_size = str(search_subfield_in_config(configuration, "als", "cut_size", True)),
            solver = search_subfield_in_config(configuration, "als", "solver", True),
            timeout = int(search_subfield_in_config(configuration, "als", "timeout", False, 60000)))

        optimizer_conf = search_field_in_config(configuration, "optimizer", True)
        assert isinstance(optimizer_conf, (list, tuple)), "the 'optimizer' field of the config file is not valid"
        assert len(optimizer_conf)  == 2, "the 'optimizer' field of the config file is not valid"

        self.fst_optimizer_conf = Optimizer.Config(
                archive_hard_limit = int(search_field_in_config(optimizer_conf[0], "archive_hard_limit", True)),
                archive_soft_limit = int(search_field_in_config(optimizer_conf[0], "archive_soft_limit", True)),
                archive_gamma = int(search_field_in_config(optimizer_conf[0], "archive_gamma", True)),
                clustering_max_iterations = int(search_field_in_config(optimizer_conf[0], "clustering_iterations", True)),
                hill_climbing_iterations = int(search_field_in_config(optimizer_conf[0], "hill_climbing_iterations", True)),
                initial_temperature = float(search_field_in_config(optimizer_conf[0], "initial_temperature", True)),
                final_temperature = float(search_field_in_config(optimizer_conf[0], "final_temperature", True)),
                cooling_factor = float(search_field_in_config(optimizer_conf[0], "cooling_factor", True)),
                annealing_iterations = int(search_field_in_config(optimizer_conf[0], "annealing_iterations", True)),
                annealing_strength = int(search_field_in_config(optimizer_conf[0], "annealing_strength", True)),
                early_termination_window = int(search_field_in_config(optimizer_conf[0], "early_termination", True)),
                multiprocessing_enabled = bool(search_field_in_config(optimizer_conf[0], "multiprocess_enabled", True)))
        
        self.snd_optimizer_conf = Optimizer.Config(
                archive_hard_limit = int(search_field_in_config(optimizer_conf[1], "archive_hard_limit", True)),
                archive_soft_limit = int(search_field_in_config(optimizer_conf[1], "archive_soft_limit", True)),
                archive_gamma = int(search_field_in_config(optimizer_conf[1], "archive_gamma", True)),
                clustering_max_iterations = int(search_field_in_config(optimizer_conf[1], "clustering_iterations", True)),
                hill_climbing_iterations = int(search_field_in_config(optimizer_conf[1], "hill_climbing_iterations", True)),
                initial_temperature = float(search_field_in_config(optimizer_conf[1], "initial_temperature", True)),
                final_temperature = float(search_field_in_config(optimizer_conf[1], "final_temperature", True)),
                cooling_factor = float(search_field_in_config(optimizer_conf[1], "cooling_factor", True)),
                annealing_iterations = int(search_field_in_config(optimizer_conf[1], "annealing_iterations", True)),
                annealing_strength = int(search_field_in_config(optimizer_conf[1], "annealing_strength", True)),
                early_termination_window = int(search_field_in_config(optimizer_conf[1], "early_termination", True)),
                multiprocessing_enabled = bool(search_field_in_config(optimizer_conf[1], "multiprocess_enabled", True)))