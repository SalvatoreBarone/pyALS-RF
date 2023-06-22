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
from distutils.dir_util import mkpath
from pyalslib import check_for_file, check_for_optional_file
from .Classifier import Classifier
from .OneStepConfigParser import OneStepConfigParser
from .TwoStepsConfigParser import TwoStepsConfigParser
from .SingleStepAlsWcMop import SingleStepAlsWcMop
from .SecondStepAlsWcMop import SecondStepAlsWcMop

def als_wc_one_step(configfile, ncpus):
    configuration = OneStepConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    check_for_file(configuration.als_conf.lut_cache)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(configuration.als_conf, ncpus)
    classifier.wc_parse(configuration.pmml)
    classifier.generate_hdl_exact_implementations(configuration.outdir)
    problem = SingleStepAlsWcMop(classifier, configuration.error_conf, ncpus)
    optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
    improve = None
    if os.path.exists(f"{configuration.outdir}/final_archive.json"):
        print("Using results from previous runs as a starting point.")
        improve = f"{configuration.outdir}/final_archive.json"
    optimizer.hill_climb_checkpoint_file = f"{configuration.outdir}/{optimizer.hill_climb_checkpoint_file}"
    optimizer.minimize_checkpoint_file = f"{configuration.outdir}/{optimizer.minimize_checkpoint_file}"
    optimizer.cache_dir = f"{configuration.outdir}/{optimizer.cache_dir}"
    configuration.termination_criterion.info()
    optimizer.run(problem,  termination_criterion = configuration.termination_criterion, improve = improve)
    optimizer.archive_to_csv(problem, f"{configuration.outdir}/report.csv")
    optimizer.plot_pareto(problem, f"{configuration.outdir}/pareto_front.pdf")
    optimizer.archive_to_json(f"{configuration.outdir}/final_archive.json")
    classifier.wc_fix_ys_helper()
    classifier.generate_hdl_onestep_asl_wc_ax_implementations(configuration.outdir, optimizer.pareto_set())
    print(f"All done! Take a look at the {configuration.outdir} directory.")


def als_wc_two_steps(configfile, ncpus):
    configuration = TwoStepsConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    check_for_file(configuration.als_conf.lut_cache)
    check_for_optional_file(configuration.error_conf.dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(configuration.als_conf)
    print("Creating classifier object...")
    classifier.wc_parse(configuration.pmml, ncpus)
    print("PMML parsing completed")
    classifier.generate_hdl_exact_implementations(configuration.outdir)
    print("HDL generation (accurate) completed")
    second_step_problem = SecondStepAlsWcMop(classifier, configuration.error_conf, configuration.fst_optimizer_conf, configuration.fst_termination_criterion, configuration.outdir, ncpus)
    print("Assertion generation (approximate) completed")
    second_step_optimizer = pyamosa.Optimizer(configuration.snd_optimizer_conf)
    second_step_optimizer.hill_climb_checkpoint_file = f"{configuration.outdir}/second_step_hillclimb_checkpoint.json"
    second_step_optimizer.minimize_checkpoint_file = f"{configuration.outdir}/second_step_minimize_checkpoint.json"
    second_step_optimizer.cache_dir = f"{configuration.outdir}/.second_step_cache"
    configuration.snd_termination_criterion.info()
    second_step_optimizer.run(second_step_problem, termination_criterion = configuration.snd_termination_criterion)
    second_step_optimizer.archive_to_csv(second_step_problem, f"{configuration.outdir}/report.csv")
    second_step_optimizer.plot_pareto(second_step_problem, f"{configuration.outdir}/pareto_front.pdf")
    second_step_optimizer.archive_to_json(f"{configuration.outdir}/final_archive.json")
    classifier.wc_fix_ys_helper()
    classifier.generate_hdl_twostep_asl_wc_ax_implementations(configuration.outdir, second_step_optimizer.pareto_set(), second_step_problem.first_step_optimizer.pareto_set())
    print(f"All done! Take a look at the {configuration.outdir} directory.")
