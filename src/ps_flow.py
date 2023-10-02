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
import os, pyamosa, time, numpy as np, matplotlib.pyplot as plt, matplotlib.lines as mlines
from pyalslib import check_for_file
from src.PsConfigParser import *
from src.PsMop import *
from .rank_based import softmax, dist_gini
from .plot import scatterplot, boxplot
from multiprocessing import cpu_count

def ps_flow(configfile, mode, alpha, beta, gamma, ncpus):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(ncpus)
    classifier.parse(configuration.pmml)
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
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

def ps_eval(configfile, nabs):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(cpu_count())
    classifier.parse(configuration.pmml)
    nabs = nabs.replace(" ", "").split(",")
    assert len(nabs) == len(classifier.model_features), f"You must set nabs for each of the model featuer (you provided {len(nabs)}, but {len(classifier.model_features)} are required"
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    classifier.reset_nabs_configuration()
    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
    print(f"Computing the accuracy for {nabs}")
    classifier.set_nabs({f["name"]: n for f, n in zip(classifier.model_features, nabs[:len(classifier.model_features)])})
    ax_acc = classifier.evaluate_test_dataset()
    acc_loss = problem.baseline_accuracy - ax_acc
    print(f"Ax accuracy: {ax_acc}, loss: {acc_loss}. #bits: {classifier.get_total_retained()}")

def ps_distance(configfile, pareto = None):
    def plot(outdir, data, samples, label):
        plt.figure(figsize=[8,4])
        y = [ ax[samples] for ax in data ]
        x = [ f"{ax['loss']:.2f}" for ax in data ]
        plt.boxplot(y, labels=x)
        plt.ylabel(label)
        plt.xlabel("Accuracy loss (%)")
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.savefig(f"{outdir}/boxplot_{samples}.pdf", bbox_inches='tight', pad_inches=0)

    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(cpu_count())
    classifier.parse(configuration.pmml)
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    archive_json = f"{configuration.outdir}/final_archive.json" if pareto is None else pareto
    assert os.path.exists(archive_json), f"No {archive_json} file found"

    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
    optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
    print("Reading the Pareto front.")
    optimizer.read_final_archive_from_json(problem, archive_json)

    print(f"{len(optimizer.archive)} solutions read from {archive_json}")

    classifier.reset_nabs_configuration()
    rhos = [ softmax(classifier.predict_mt(x)) for x in tqdm(classifier.x_test, desc="Evaluating rho ...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False) ]
    data_to_plot = []
    for ax in  tqdm(optimizer.archive, desc="Analysing ACSs...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False):
        nabs = {f["name"]: n for f, n in zip(classifier.model_features, ax["x"][:len(classifier.model_features)])}
        classifier.set_nabs(nabs)
        rhos_prime = [ softmax(classifier.predict_mt(x)) for x in tqdm(classifier.x_test, desc="Evaluating rho' ...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False) ]
        data_to_plot.append({
            "loss": ax["f"][0], 
            "mae_p_pp_T": [ np.mean(np.abs(rho - ax_rho)) for rho, ax_rho in zip(rhos, rhos_prime) ], 
            "rhotheta_rhoptheta" : [ float(rho[y] - ax_rho[y]) for rho, ax_rho, y in zip(rhos, rhos_prime, classifier.y_test)]})
        
    data_to_plot = sorted(data_to_plot, key=lambda d: d["loss"]) 
    plot(configuration.outdir, data_to_plot, "mae_p_pp_T", r"$\frac{1}{M}\sum_{i=0}^{M-1} |\rho_i - \rho'_i|,\; \tau \in T$")
    plot(configuration.outdir, data_to_plot, "rhotheta_rhoptheta", r"$ \rho_{\theta^*} - \rho'_{\theta^*},\; \tau \in T$")
    classifier.pool.close()
    print(f"All done! Take a look at the {configuration.outdir} directory.")
    
def ps_compare(configfile, outdir, pareto, alpha, beta, gamma, maxloss, neval):
            
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(cpu_count())
    classifier.parse(configuration.pmml)
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
    problem.load_cache(f"{configuration.outdir}/.cache")
    archive_json = f"{configuration.outdir}/final_archive.json" if pareto is None else pareto
    
    n_vars = len(classifier.model_features)
    classifier.reset_nabs_configuration()
    classifier.reset_assertion_configuration()
    C, M = datasetRanking(classifier)
    baseline_accuracy = len(C) / (len(C) + len(M)) * 100
    print(f"Baseline accuracy: {baseline_accuracy} %")
    
    legend_markers = [
         mlines.Line2D([],[], color='crimson', marker='d', linestyle='None', label='Reference'),
         mlines.Line2D([],[], color='mediumblue', marker='o', linestyle='None', label='Rank-based')]
    
    maxMiss = int((len(C) + len(M)) * (100 - baseline_accuracy + maxloss) / 100)
    
    estimation_error = []
    evaluated_samples = [maxMiss] * len(problem.cache)
    if os.path.exists(archive_json):
        actual_pareto = []
        estimated_pareto = []
    
        problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
        optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
        print("Reading the Pareto front.")
        optimizer.read_final_archive_from_json(problem, archive_json)
        print(f"{len(optimizer.archive)} solutions read from {archive_json}")
        classifier.reset_nabs_configuration()
        
        for ax in  tqdm(optimizer.archive, desc="Analysing ACSs...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False):
            nabs = {f["name"]: n for f, n in zip(classifier.model_features, ax["x"][:len(classifier.model_features)])}
            classifier.set_nabs(nabs)
            
            estloss, nsamples = estimateLoss(baseline_accuracy, 2 * maxloss, alpha, beta, gamma, classifier, C, M)
            actual_pareto.append(ax["f"])
            estimated_pareto.append(np.array([estloss, ax["f"][1]]))
            estimation_error.append(ax["f"][0] - estloss)
            evaluated_samples.append(nsamples)
            
        scatterplot([np.array(actual_pareto), np.array(estimated_pareto)], legend_markers, "Accuracy loss (%)", "Power consumption (mW)", f"{outdir}/actual_vs_est_pareto_comparison.pdf")
        
    boxplot(estimation_error, "", "", f"{outdir}/estimation_error.pdf", annotate = True, figsize = (3, 4))
    boxplot(evaluated_samples, "", "", f"{outdir}/evaluated_samples.pdf", annotate = True, figsize = (3, 4), float_format = "%.0f")
    classifier.pool.close()
    print(f"All done! Take a look at the {outdir} directory.")

def compute_gini_dist(configfile, outdir):
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.pmml)
    check_for_file(configuration.error_conf.test_dataset)
    classifier = Classifier(cpu_count())
    classifier.parse(configuration.pmml)
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    classifier.reset_nabs_configuration()
    classifier.reset_assertion_configuration()
    dist_gini(classifier, outdir)