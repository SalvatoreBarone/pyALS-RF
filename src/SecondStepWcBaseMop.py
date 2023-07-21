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
from .BaseMop import *
from .FirstStepAlsMop import *

class SecondStepWcBaseMop(BaseMop):
    def __init__(self, classifier, error_conf, fst_opt_conf, fst_opt_term_criterion, out_dir, ncpus):
        self.error_conf = error_conf
        self.fst_opt_conf = fst_opt_conf
        BaseMop.__init__(self, classifier, self.error_conf.test_dataset, ncpus)
        assert len(self.classifier.get_trees()) > 1, "The two steps approach is available only for random forest/bagging classifiers"

        t = self.classifier.get_trees()[0]
        self.first_step_problem = FirstStepAlsMop(t, self.dataset, self.error_conf)
        self.first_step_optimizer = pyamosa.Optimizer(self.fst_opt_conf)
        t_outdir = f"{out_dir}/wc_tree"
        print(t_outdir)
        mkpath(t_outdir)
        self.first_step_optimizer.hill_climb_checkpoint_file = f"{t_outdir}/first_step_hillclimb_checkpoint.json"
        self.first_step_optimizer.minimize_checkpoint_file = f"{t_outdir}/first_step_minimize_checkpoint.json"
        self.first_step_optimizer.cache_dir = f"{t_outdir}/.first_step_cache"
        improve = None
        if os.path.exists(f"{t_outdir}/final_archive.json"):
            print("Using results from previous runs as a starting point.")
            improve = f"{t_outdir}/final_archive.json"
        fst_opt_term_criterion.info()
        print(f"Evaluating EP using {len(self.first_step_problem.samples)} vectors")
        self.first_step_optimizer.run(self.first_step_problem, termination_criterion = fst_opt_term_criterion, improve = improve)
        self.first_step_optimizer.archive_to_csv(self.first_step_problem, f"{t_outdir}/report.csv")
        self.first_step_optimizer.plot_pareto(self.first_step_problem, f"{t_outdir}/pareto_front.pdf")
        self.first_step_optimizer.archive_to_json(f"{t_outdir}/final_archive.json")
