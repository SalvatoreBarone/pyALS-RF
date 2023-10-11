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
import json5
class ConfigParser:
    def __init__(self, configfile : str):
        self.configuration = json5.load(open(configfile))

def search_field_in_config(configuration, field, mandatory = True, default_value = None):
    try:
        return configuration[field]
    except KeyError as e:
        if not mandatory:
            return default_value
        print(f"{e} not found in the configuration")
        exit()

def search_subfield_in_config(configuration, section, field, mandatory = True, default_value = None):
    try:
        return configuration[section][field]
    except KeyError as e:
        if not mandatory:
            return default_value
        print(f"{e} not found in the configuration")
        exit()

