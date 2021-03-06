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
import itertools, numpy, enum, json
from multiprocessing import cpu_count, Pool
from pyAMOSA.AMOSA import *
from .Utility import *
from enum import Enum


def evaluate_preloaded_dataset(classifier, samples):
    return classifier.evaluate_preloaded_dataset(samples)


def evaluate_eprob(graph, samples, configuration):
    return sum([0 if sample["output"] == graph.evaluate(sample["input"], configuration) else 1 for sample in samples])


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


class SingleStepOptimizerConf:
    def __init__(self, error_conf, amosa_conf):
        self.error_conf = error_conf
        self.amosa_conf = amosa_conf


class TwoStepsOptimizerConf:
    def __init__(self, fst_error_conf, fst_amosa_conf, snd_error_conf, snd_amosa_conf):
        self.fst_error_conf = fst_error_conf
        self.fst_amosa_conf = fst_amosa_conf
        self.snd_error_conf = snd_error_conf
        self.snd_amosa_conf = snd_amosa_conf


class OptimizationBaseClass:
    def __init__(self, classifier, dataset_csv, config):
        self.classifier = classifier
        self.features = self.classifier.get_features()
        self.dataset = classifier.preload_dataset(dataset_csv)
        self.n_samples = len(self.dataset)
        self.config = config
        classifier.reset_assertion_configuration()
        classifier.reset_nabs_configuration()
        classifiers = [copy.deepcopy(classifier)] * cpu_count()
        self.args = [[c, d] for c, d in zip(classifiers, list_partitioning(self.dataset, cpu_count()))]
        self.baseline_accuracy = self.evaluate_dataset()
        print(f"Baseline accuracy: {self.baseline_accuracy}.")

    def evaluate_dataset(self):
        with Pool(cpu_count()) as pool:
            res = pool.starmap(evaluate_preloaded_dataset, self.args)
        return sum(res) * 100 / self.n_samples

    def get_accuracy_loss(self):
        return self.baseline_accuracy - self.evaluate_dataset()


class SingleStepPsOnly(OptimizationBaseClass, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config):
        OptimizationBaseClass.__init__(self, classifier, dataset_csv, config)
        n_vars = len(classifier.get_features())
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, [53] * n_vars, 2, 1)

    def __set_matter_configuration(self, x):
        nabs = {f["name"]: n for f, n in zip(self.features, x[:len(self.features)])}
        for item in self.args:
            item[0].set_nabs(nabs)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.config.error_conf.threshold]


class SingleStepAlsOnly(OptimizationBaseClass, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config):
        OptimizationBaseClass.__init__(self, classifier, dataset_csv, config)
        self.cells_per_tree = classifier.get_als_cells_per_tree()
        n_vars = sum(self.cells_per_tree)
        ub = classifier.get_als_dv_upper_bound()
        print(f"{len(ub)} d.v.: {ub}")
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        configurations = []
        count = 0
        for size in self.cells_per_tree:
            configurations.append([x[i + count] for i in range(size)])
            count += size
        for item in self.args:
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.config.error_conf.threshold]


class SingleStepCombined(OptimizationBaseClass, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config):
        OptimizationBaseClass.__init__(self, classifier, dataset_csv, config)
        self.cells_per_tree = classifier.get_als_cells_per_tree()
        n_features = len(self.features)
        n_cells = sum(self.cells_per_tree)
        n_vars = n_features + n_cells
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars,  [53] *  n_features + classifier.get_als_dv_upper_bound(), 3, 1)

    def __set_matter_configuration(self, x):
        configurations = []
        count = 0
        for size in self.cells_per_tree:
            configurations.append([x[i + count + len(self.features)] for i in range(size)])
            count += size
        nabs = {f["name"]: n for f, n in zip(self.features, x[:len(self.features)])}
        for item in self.args:
            item[0].set_nabs(nabs)
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.error_conf.threshold]


class FirstStepOptimizer(AMOSA.Problem):
    def __init__(self, decision_tree, preloaded_dataset, error_config):
        self.decision_tree = decision_tree
        graph = self.decision_tree.get_graph()
        n_vars = graph.get_num_cells()
        ub = self.decision_tree.get_als_dv_upper_bound()

        self.error_config = error_config
        self.decision_tree.reset_assertion_configuration()
        self.samples = self.generate_samples(graph, preloaded_dataset)
        self.total_samples = len(self.samples)
        self.args = [[g, s, [0] * n_vars] for g, s in zip([copy.deepcopy(graph)] * cpu_count(), list_partitioning(self.samples, cpu_count()))]

        print(f"Tree {self.decision_tree.get_name()}. d.v. #{len(ub)}: {ub}")
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def generate_samples(self, graph, preloaded_dataset):
        samples = []
        for sample in preloaded_dataset:
            inputs = self.decision_tree.get_boxes_output(sample["input"])
            samples.append({"input": inputs, "output": graph.evaluate(inputs)})
        return samples

    def __set_matter_configuration(self, x):
        self.decision_tree.set_assertions_configuration(x)
        for a in self.args:
            a[2] = self.decision_tree.get_assertions_configuration()

    def __get_eprob(self):
        with Pool(cpu_count()) as pool:
            error = pool.starmap(evaluate_eprob, self.args)
        rs = sum(error) * 100 / len(self.samples)
        return rs

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.__get_eprob()
        f2 = self.decision_tree.get_current_required_aig_nodes()
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.error_config.threshold]


class SecondStepOptimizerBase(OptimizationBaseClass):
    def __init__(self, classifier, dataset_csv, config, improve, out_dir):
        OptimizationBaseClass.__init__(self, classifier, dataset_csv, config)
        self.opt_solutions_for_trees = []
        for t in self.classifier.get_trees():
            problem = FirstStepOptimizer(t, self.dataset, config.fst_error_conf)
            optimizer = AMOSA(self.config.fst_amosa_conf)
            optimizer.hill_climb_checkpoint_file = f"{out_dir}/first_step_hillclimb_checkpoint_{t.get_name()}.json"
            optimizer.minimize_checkpoint_file = f"{out_dir}/first_step_hminimize_checkpoint{t.get_name()}.json"
            optimizer.run(problem, improve, False)
            optimizer.save_results(problem, f"{out_dir}/report_{t.get_name()}.csv")
            optimizer.plot_pareto(problem, f"{out_dir}/pareto_front_{t.get_name()}.pdf")
            optimizer.save_pareto_set(problem, f"{out_dir}/pareto_set_{t.get_name()}.csv")
            optimizer.archive_to_json(f"{out_dir}/final_archive_{t.get_name()}.json")
            self.opt_solutions_for_trees.append(optimizer.pareto_set())


class SecondStepOptimizerAlsOnly(SecondStepOptimizerBase, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config, improve, outdir):
        SecondStepOptimizerBase.__init__(self, classifier, dataset_csv, config, improve, outdir)
        n_vars = self.classifier.get_num_of_trees()
        ub = [ len(i)-1 for i in self.opt_solutions_for_trees ]
        print(f"Baseline accuracy: {self.baseline_accuracy}.")
        print(f"d.v. #{len(ub)}, {ub}")
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, ub, 2, 1)

    def __set_matter_configuration(self, x):
        configurations = [ s[c] for s, c in zip(self.opt_solutions_for_trees, x)  ]
        for item in self.args:
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.config.snd_error_conf.threshold]


class SecondStepOptimizerCombined(SecondStepOptimizerBase, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config, improve, outdir):
        SecondStepOptimizerBase.__init__(self, classifier, dataset_csv, config, improve, outdir)
        n_vars = len(self.features) + self.classifier.get_num_of_trees()
        ub = [53] * len(self.features) + [ len(i)-1 for i in self.opt_solutions_for_trees ]
        print(f"Baseline accuracy: {self.baseline_accuracy}.")
        print(f"d.v. #{len(ub)}, {ub}")
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, ub, 3, 1)

    def __set_matter_configuration(self, x):
        nabs = { f["name"] : n for f, n in zip(self.features, x[:len(self.features)]) }
        configurations = [ s[c] for s, c in zip(self.opt_solutions_for_trees, x[len(self.features):])  ]
        for item in self.args:
            item[0].set_nabs(nabs)
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.snd_error_conf.threshold]
