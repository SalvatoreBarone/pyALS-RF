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
import itertools, numpy
from multiprocessing import cpu_count, Pool
from .AMOSA import *
from .Utility import *

def evaluate_preloaded_dataset(classifier, samples):
    return classifier.evaluate_preloaded_dataset(samples)

def evaluate_eprob(graph, samples, configuration):
    return sum([0 if sample["output"] == graph.evaluate(sample["input"], configuration) else 1 for sample in samples])

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
        dataset = classifier.preload_dataset(dataset_csv)
        self.n_samples = len(dataset)
        self.config = config
        classifier.reset_assertion_configuration()
        classifier.reset_nabs_configuration()
        classifiers = [copy.deepcopy(classifier)] * cpu_count()
        self.args = [[c, d] for c, d in zip(classifiers, list_partitioning(dataset, cpu_count()))]
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
        for item in self.args:
            item[0].set_nabs([{"name": f["name"], "nab": x} for f, x in zip(self.features, x)])

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
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, classifier.get_als_dv_upper_bound(), 2, 1)

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
        for item in self.args:
            item[0].set_nabs([{"name": f["name"], "nab": x} for f, x in zip(self.features, x[:len(self.features)])])
            item[0].set_assertions_configuration(configurations)

    def evaluate(self, x, out):
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.error_conf.threshold]

class FirstStepOptimizer(AMOSA.Problem):
    def __init__(self, decision_tree, error_config):
        self.decision_tree = decision_tree
        self.error_config = error_config
        self.catalog = self.decision_tree.get_catalog_for_assertions()
        self.graph = self.decision_tree.get_graph()
        self.graphs = [copy.deepcopy(self.graph)] * cpu_count()
        n_vars = self.decision_tree.get_als_num_of_dv()
        self.decision_tree.reset_assertion_configuration()
        self.samples = []
        self.__generate_samples()
        self.total_samples = len(self.samples)
        self.args = [[g, s, [0] * n_vars] for g, s in zip(self.graphs, list_partitioning(self.samples, cpu_count()))]
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, self.decision_tree.get_als_dv_upper_bound(), 2, 1)

    def __generate_samples(self):
        PI = self.graph.get_pi()
        if self.error_config.n_vectors != 0:
            for _ in range(self.error_config.n_vectors):
                inputs = {i["name"]: bool(random.getrandbits(1)) for i in PI}
                self.samples.append({"input": inputs, "output": self.graph.evaluate(inputs)})
        else:
            self.error_config.n_vectors = 2 ** len(PI)
            permutations = [list(i) for i in itertools.product([False, True], repeat=len(PI))]
            for perm in permutations:
                inputs = {i["name"]: p for i, p in zip(PI, perm)}
                self.samples.append({"input": inputs, "output": self.graph.evaluate(inputs)})

    def __set_matter_configuration(self, x):
        self.decision_tree.set_assertions_configuration(x)
        for a in self.args:
            a[2] = x

    def __get_eprob(self):
        with Pool(cpu_count()) as pool:
            error = pool.starmap(evaluate_eprob, self.args)
        rs = sum(error) / self.error_config.n_vectors
        if self.error_config.n_vectors != 0:
            return rs + 4.5 / self.error_config.n_vectors * (1 + np.sqrt(1 + 4 / 9 * self.error_config.n_vectors * rs * (1 - rs)))
        else:
            return rs

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.__get_eprob()
        f2 = self.decision_tree.get_current_required_aig_nodes()
        out["F"] = [f1, f2]
        out["G"] = [f1 - self.error_config.threshold]

class SecondStepOptimizerBase(OptimizationBaseClass):
    def __init__(self, classifier, dataset_csv, config):
        OptimizationBaseClass.__init__(self, classifier, dataset_csv, config)
        self.opt_solutions_for_trees = []
        for t in self.classifier.get_trees():
            problem = FirstStepOptimizer(t, config.fst_error_config)
            optimizer = AMOSA(self.config.fst_amosa_conf)
            optimizer.minimize(problem)
            self.opt_solutions_for_trees.append(optimizer.pareto_set())

class SecondStepOptimizerAlsOnly(SecondStepOptimizerBase, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config):
        SecondStepOptimizerBase.__init__(self, classifier, dataset_csv, config)
        n_vars = self.classifier.get_num_of_trees()
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, [ len(i) for i in self.opt_solutions_for_trees ], 2, 1)

    def __set_matter_configuration(self, x):
        pass

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2]
        out["g"] = [f1 - self.config.error_conf.threshold]


class SecondStepOptimizerCombined(SecondStepOptimizerBase, AMOSA.Problem):
    def __init__(self, classifier, dataset_csv, config):
        SecondStepOptimizerBase.__init__(self, classifier, dataset_csv, config)
        n_vars = len(self.features) + self.classifier.get_num_of_trees()
        upper_bound = [53] * len(self.features) + [ len(i) for i in self.opt_solutions_for_trees ]
        AMOSA.Problem.__init__(self, n_vars, [AMOSA.Type.INTEGER] * n_vars, [0] * n_vars, upper_bound, 3, 1)

    def __set_matter_configuration(self, x):
        pass

    def evaluate(self, x, out):
        self.__set_matter_configuration(x)
        f1 = self.get_accuracy_loss()
        f2 = self.args[0][0].get_total_retained()
        f3 = sum(self.args[0][0].get_current_required_aig_nodes())
        out["f"] = [f1, f2, f3]
        out["g"] = [f1 - self.config.error_conf.threshold]

