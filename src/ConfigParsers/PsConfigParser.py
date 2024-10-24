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
import json, pyamosa
from pyalslib import ALSConfig
from ..Model.ErrorConfig import ErrorConfig
from .ConfigParser import *
from .DtGenConfigParser import *

class PSConfigParser(ConfigParser):
    def __init__(self, config_file):
        super().__init__(config_file)
        

        self.model_source = search_field_in_config(self.configuration, "model", True)
        self.outdir = search_field_in_config(self.configuration, "outdir", True)

        self.als_conf = ALSConfig(
            lut_cache = search_subfield_in_config(self.configuration, "als", "cache", True),
            cut_size = str(search_subfield_in_config(self.configuration, "als", "cut_size", True)),
            solver = search_subfield_in_config(self.configuration, "als", "solver", True),
            timeout = int(search_subfield_in_config(self.configuration, "als", "timeout", False, 60000)))

        self.error_conf = ErrorConfig(
            max_loss_perc = search_subfield_in_config(self.configuration, "error", "max_loss_perc", True),
            test_dataset = search_subfield_in_config(self.configuration, "error", "test_dataset", True),
            dataset_description = DtGenConfigParser(search_subfield_in_config(self.configuration, "error", "dataset_description", True)))
        
        
        optimizer_conf = search_field_in_config(self.configuration, "optimizer", True)
        assert isinstance(optimizer_conf, dict), "the 'optimizer' field of the config file is not valid"

        self.optimizer_conf = pyamosa.Config(
                archive_hard_limit = int(search_field_in_config(optimizer_conf, "archive_hard_limit", True)),
                archive_soft_limit = int(search_field_in_config(optimizer_conf, "archive_soft_limit", True)),
                archive_gamma = int(search_field_in_config(optimizer_conf, "archive_gamma", True)),
                clustering_max_iterations = int(search_field_in_config(optimizer_conf, "clustering_iterations", True)),
                hill_climbing_iterations = int(search_field_in_config(optimizer_conf, "hill_climbing_iterations", True)),
                initial_temperature = float(search_field_in_config(optimizer_conf, "initial_temperature", True)),
                cooling_factor = float(search_field_in_config(optimizer_conf, "cooling_factor", True)),
                annealing_iterations = int(search_field_in_config(optimizer_conf, "annealing_iterations", True)),
                annealing_strength = int(search_field_in_config(optimizer_conf, "annealing_strength", True)),
                multiprocessing_enabled = bool(search_field_in_config(optimizer_conf, "multiprocess_enabled", False, False)),
                hill_climb_checkpoint_file = f"{self.outdir}/.hill_climb_checkpoint.json",
                minimize_checkpoint_file = f"{self.outdir}/.annealing_checkpoint.json",
                cache_dir = f"{self.outdir}/.cache")
        
        self.variable_grouping_strategy = search_subfield_in_config(optimizer_conf, "amosa", "grouping", False, None)
        self.transfer_strategy_objectives = search_subfield_in_config(optimizer_conf, "amosa", "tso", False, "all")
        self.transfer_strategy_variables =  search_subfield_in_config(optimizer_conf, "amosa", "tsv", False, "any")
        
        optimizer_min_temperature = search_field_in_config(optimizer_conf, "final_temperature", True, 1e-7)
        optimizer_stop_phy_window = search_field_in_config(optimizer_conf, "early_termination", False, 20)
        optimizer_max_duration = search_field_in_config(optimizer_conf, "max_duration", False, None)
        self.termination_criterion = pyamosa.CombinedStopCriterion(optimizer_max_duration, optimizer_min_temperature, optimizer_stop_phy_window)
        