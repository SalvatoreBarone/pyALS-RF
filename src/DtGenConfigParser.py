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


class DtGenConfigParser:
	__allowed_criterion = ["gini", "entropy", "log_loss"]
	__allowed_max_features = ["sqrt", "log2"]


	def __init__(self, config_file):
		configuration = json.load(open(config_file))
		self.dataset = {
			"path": search_subfield_in_config(configuration, "dataset", "path"),
			"classes" : search_subfield_in_config(configuration, "dataset", "classes"),
			"fraction" : search_subfield_in_config(configuration, "dataset", "fraction", False, 0.9)
		}
		self.output_dir = search_field_in_config(configuration, "output", False, "output")
		self.model = {
			"predictors" : search_subfield_in_config(configuration, "model", "predictors", False, 1),
            "bootstrap" : search_subfield_in_config(configuration, "model", "bootstrap", False, True),
            "random_features" : search_subfield_in_config(configuration, "model", "random_features", False, True),
			"depth" : search_subfield_in_config(configuration, "model", "depth", False, None),
			"criterion" : search_subfield_in_config(configuration, "model", "criterion", False, "gini"),
			"min_samples_split" : search_subfield_in_config(configuration, "model", "min_samples_split", False, 2),
			"min_samples_leaf" : search_subfield_in_config(configuration, "model", "min_samples_leaf", False, 1),
			"max_features" : search_subfield_in_config(configuration, "model", "max_features", False, "sqrt"),
			"max_leaf_nodes" : search_subfield_in_config(configuration, "model", "max_leaf_nodes", False, None),
			"min_impurity_decrease" : search_subfield_in_config(configuration, "model", "min_impurity_decrease", False, 0.0),
			"ccp_alpha" : search_subfield_in_config(configuration, "model", "ccp_alpha", False, 0.0)
		}
		if self.model["criterion"] not in self.__allowed_criterion:
			print(f"{self.criterion} is not allowed as splitting criterion. Allowed ones are {self.__allowed_criterion}")
			exit()
		if self.model["max_features"] not in self.__allowed_max_features and not isinstance(self.model["max_features"], (int, float)):
			print(f"{self.max_features} is not allowed for the \"max_features\" field. Allowed ones are {self.__allowed_max_features}, integer and float values")
			exit()
