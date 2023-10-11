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
import json
from .ConfigParser import *


class DtGenConfigParser(ConfigParser):
	def __init__(self, config_file):
		super().__init__(config_file)
		self.separator = search_field_in_config(self.configuration, "separator", False, ",")
		self.outcome_col = search_field_in_config(self.configuration, "outcome_col", False, None)
		self.skip_header = search_field_in_config(self.configuration, "skip_header", False, True)
		self.attributes_name = search_field_in_config(self.configuration, "attributes_name", True)
		self.classes_name = search_field_in_config(self.configuration, "classes_name", True)
