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
import os, pyamosa
from pyalslib import check_for_file
from src.PsConfigParser import *
from src.PsMop import *

def ps_flow(configfile, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(configuration.als_conf)
    classifier.parse(configuration.pmml, ncpus)
    classifier.generate_hdl_exact_implementations(configuration.outdir)
    problem = PsMop(classifier, configuration.error_conf, ncpus)
    optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
    improve = None
    if os.path.exists(f"{configuration.outdir}/final_archive.json"):
        print("Using results from previous runs as a starting point.")
        improve = f"{configuration.outdir}/final_archive.json"
    optimizer.hill_climb_checkpoint_file = f"{configuration.outdir}/{optimizer.hill_climb_checkpoint_file}"
    optimizer.minimize_checkpoint_file = f"{configuration.outdir}/{optimizer.minimize_checkpoint_file}"
    optimizer.cache_dir = f"{configuration.outdir}/{optimizer.cache_dir}"
    print("Termination criterion:")
    configuration.termination_criterion.info()
    optimizer.run(problem, termination_criterion = configuration.termination_criterion, improve = improve)
    optimizer.archive_to_csv(problem, f"{configuration.outdir}/report.csv")
    optimizer.plot_pareto(problem, f"{configuration.outdir}/pareto_front.pdf")
    optimizer.archive_to_json(f"{configuration.outdir}/final_archive.json")
    classifier.generate_hdl_ps_ax_implementations(configuration.outdir, optimizer.pareto_set())
    print(f"All done! Take a look at the {configuration.outdir} directory.")