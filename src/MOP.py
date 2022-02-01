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
from .AMOSA import *
from liberty.parser import parse_liberty

class AxConfig:
    class Technique(Enum):
        ALS = 1,
        PS = 2,
        FULL = 3

    class Strategy(Enum):
        ONE_STEP = 1,
        TWO_STEPS = 2

    def __init__(self, technique, strategy):
        ax_technique = {
            "als": AxConfig.Technique.ALS,
            "ps": AxConfig.Technique.PS,
            "full": AxConfig.Technique.FULL
        }
        if technique not in ax_technique.keys():
            raise ValueError("{}: Approximation technique not recognized".format(technique))
        else:
            self.technique = ax_technique[technique]
        ax_strategy = {
            "one": AxConfig.Strategy.ONE_STEP,
            "single": AxConfig.Strategy.ONE_STEP,
            "two": AxConfig.Strategy.TWO_STEPS,
        }
        if strategy not in ax_strategy.keys():
            raise ValueError("{}: approximation strategy not recognized".format(strategy))
        else:
            self.strategy = ax_strategy[strategy]

class ErrorConfig:
    class Metric(Enum):
        EPROB = 1
        AWCE = 2
        MED = 3
    def __init__(self, metric, threshold, vectors, weights = None):
        error_metrics = {
            "eprob": ErrorConfig.Metric.EPROB,
            "EProb": ErrorConfig.Metric.EPROB,
            "EPROB": ErrorConfig.Metric.EPROB,
            "awce": ErrorConfig.Metric.AWCE,
            "AWCE": ErrorConfig.Metric.AWCE,
            "med" : ErrorConfig.Metric.MED,
            "MED" : ErrorConfig.Metric.MED}
        if metric not in error_metrics.keys():
            raise ValueError(f"{metric}: error-metric not recognized")
        else:
            self.metric = error_metrics[metric]
        self.threshold = threshold
        self.n_vectors = vectors
        self.weights = weights

class HwConfig:
    class Metric(Enum):
        GATES = 1
        DEPTH = 5
    def __init__(self, metrics):
        hw_metrics = {
            "gates" : HwConfig.Metric.GATES,
            "depth": HwConfig.Metric.DEPTH
        }
        self.metrics = []
        for metric in metrics:
            if metric not in hw_metrics.keys():
                raise ValueError(f"{metric}: hw-metric not recognized")
            else:
                self.metrics.append(hw_metrics[metric])

class SingleStepOptimizerConf:
    def __init__(self, error_conf, hw_conf, amosa_conf):
        self.error_conf = error_conf
        self.hw_conf = hw_conf
        self.amosa_conf = amosa_conf

class TwoStepsOptimizerConf:
    def __init__(self, fst_error_conf, fst_hw_conf, fst_amosa_conf, snd_error_conf, snd_hw_config, snd_amosa_conf):
        self.fst_error_conf = fst_error_conf
        self.fst_hw_conf = fst_hw_conf
        self.fst_amosa_conf = fst_amosa_conf
        self.snd_error_conf = snd_error_conf
        self.snd_hw_conf = snd_hw_config
        self.snd_amosa_conf = snd_amosa_conf

