"""
Copyright 2021 Salvatore Barone <salvatore.barone@unina.it>

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
from enum import Enum

class AxConfig:
    class Technique(Enum):
        ALS = 1,
        PS = 2,
        FULL = 3

    class Strategy(Enum):
        ONE_STEP = 1,
        TWO_STEPS = 2

    def __init__(self, technique, strategy):
        ax_technique = {"als": AxConfig.Technique.ALS, "ps": AxConfig.Technique.PS, "full": AxConfig.Technique.FULL}
        if technique not in ["als", "ps", "full"]:
            raise ValueError("{}: Approximation technique not recognized".format(technique))
        else:
            self.technique = ax_technique[technique]
        ax_strategy = {"one": AxConfig.Strategy.ONE_STEP, "two": AxConfig.Strategy.TWO_STEPS}
        if strategy not in ["one", "two"]:
            raise ValueError("{}: approximation strategy not recognized".format(strategy))
        else:
            self.strategy = ax_strategy[strategy]

class ALSConfig:
    def __init__(self, luttech, catalog, timeout):
        self.luttech = luttech
        self.catalog = catalog
        self.timeout = timeout


class NSGAConfig:
    def __init__(self, pop_size, iterations, cross_p, cross_eta, mut_p, mut_eta, max_err):
        self.pop_size = pop_size
        self.iterations = iterations
        self.cross_p = cross_p
        self.cross_eta = cross_eta
        self.mut_p = mut_p
        self.mut_eta = mut_eta
        self.max_error = max_err
