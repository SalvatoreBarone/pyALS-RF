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
import os, pyamosa, time, numpy as np
from pyalslib import check_for_file
from src.PsConfigParser import *
from src.PsMop import *

def ps_flow(configfile, mode, alpha, beta, gamma, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(ncpus)
    classifier.parse(configuration.pmml)
    classifier.read_dataset(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, ncpus) if mode == "full" else RankBasedPsMop(classifier, configuration.error_conf.max_loss_perc, alpha, beta, gamma, ncpus)
    optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
    improve = None
    if os.path.exists(f"{configuration.outdir}/final_archive.json"):
        print("Using results from previous runs as a starting point.")
        improve = f"{configuration.outdir}/final_archive.json"
    print("Termination criterion:")
    configuration.termination_criterion.info()
    init_t = time.time()
    optimizer.run(problem, termination_criterion = configuration.termination_criterion, improve = improve)
    optimizer.archive_to_json(f"{configuration.outdir}/final_archive.json")
    dt = time.time() - init_t
    print(f"AMOSA heuristic completed in {dt} seconds")
    hours = int(optimizer.duration / 3600)
    minutes = int((optimizer.duration - hours * 3600) / 60)
    print(f"Took {hours} hours, {minutes} minutes")
    print(f"Cache hits: {problem.cache_hits} over {problem.total_calls} evaluations.")
    print(f"{len(problem.cache)} cache entries collected")
    if mode == "rank":
        print(f"Average samples: {np.mean(problem.sample_count)} (Total #of samples: {len(classifier.y_test)})")
        optimizer.archive = problem.archived_actual_accuracy(optimizer.archive)
        optimizer.archive_to_json(f"{configuration.outdir}/final_archive.json")
    
    optimizer.archive_to_csv(problem, f"{configuration.outdir}/report.csv")
    optimizer.plot_pareto(problem, f"{configuration.outdir}/pareto_front.pdf")
    classifier.pool.close()
    print(f"All done! Take a look at the {configuration.outdir} directory.")