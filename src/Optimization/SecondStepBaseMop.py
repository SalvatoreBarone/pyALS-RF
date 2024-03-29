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
import os
from distutils.dir_util import mkpath
from .BaseMop import *
from .FirstStepAlsMop import *

class SecondStepBaseMop(BaseMop):
    def __init__(self, classifier, error_conf, opt_conf, term_criterion, out_dir, ncpus):
        self.error_conf = error_conf
        self.opt_conf = opt_conf
        BaseMop.__init__(self, classifier, self.error_conf.test_dataset, ncpus)
        self.opt_solutions_for_trees = []
        assert len(self.classifier.get_trees()) > 1, "The two steps approach is available only for random forest/bagging classifiers"
        for t in self.classifier.get_trees():
            problem = FirstStepAlsMop(t, self.dataset, self.error_conf)
            optimizer = pyamosa.Optimizer(self.opt_conf)
            t_outdir = f"{out_dir}/{t.get_name()}"
            mkpath(t_outdir)
            optimizer.hill_climb_checkpoint_file = f"{t_outdir}/first_step_hillclimb_checkpoint.json"
            optimizer.minimize_checkpoint_file = f"{t_outdir}/first_step_hminimize_checkpoint.json"
            optimizer.cache_dir = f"{t_outdir}/.cache"
            improve = None
            if os.path.exists(f"{t_outdir}/final_archive.json"):
                print("Using results from previous runs as a starting point.")
                improve = f"{t_outdir}/final_archive.json"
            term_criterion.info()
            print(f"Evaluating EP using {len(problem.samples)} vectors")
            optimizer.run(problem, termination_criterion = term_criterion, improve = improve)
            optimizer.archive_to_csv(problem, f"{t_outdir}/report.csv")
            optimizer.plot_pareto(problem, f"{t_outdir}/pareto_front.pdf")
            optimizer.archive_to_json(f"{t_outdir}/final_archive.json")
            self.opt_solutions_for_trees.append(optimizer.pareto_set())
