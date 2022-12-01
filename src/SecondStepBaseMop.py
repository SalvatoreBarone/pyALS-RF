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
from .BaseMop import *
from .FirstStepAlsMop import *

class SecondStepBaseMop(BaseMop):
    def __init__(self, classifier, dataset_csv, config, improve, out_dir):
        BaseMop.__init__(self, classifier, dataset_csv, config)
        self.opt_solutions_for_trees = []
        for t in self.classifier.get_trees():
            problem = FirstStepAlsMop(t, self.dataset, config.fst_error_conf)
            optimizer = Optimizer(self.config.fst_amosa_conf)
            optimizer.hill_climb_checkpoint_file = f"{out_dir}/first_step_hillclimb_checkpoint_{t.get_name()}.json"
            optimizer.minimize_checkpoint_file = f"{out_dir}/first_step_hminimize_checkpoint{t.get_name()}.json"
            optimizer.cache_dir = f"{out_dir}/.cache_{t.get_name()}"
            optimizer.run(problem, improve, False)
            optimizer.save_results(problem, f"{out_dir}/report_{t.get_name()}.csv")
            optimizer.plot_pareto(problem, f"{out_dir}/pareto_front_{t.get_name()}.pdf")
            optimizer.archive_to_json(f"{out_dir}/final_archive_{t.get_name()}.json")
            self.opt_solutions_for_trees.append(optimizer.pareto_set())
