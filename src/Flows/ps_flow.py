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
import logging, os, pyamosa, time, numpy as np, matplotlib.pyplot as plt, matplotlib.lines as mlines
from pyalslib import check_for_file
from distutils.dir_util import mkpath
from multiprocessing import cpu_count
from ..ctx_factory import load_configuration_ps, create_classifier, create_problem, create_optimizer, can_improve
from ..ConfigParsers.PsConfigParser import *
from .PS.PsMop import *
from ..Model.rank_based import softmax, dist_gini
from ..plot import scatterplot, boxplot, hist


def ps_flow(ctx : dict, mode : str, alpha : float, beta : float, gamma : float, output : str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the pruning flow.")
    load_configuration_ps(ctx)
    if output is not None:
        ctx.obj['configuration'].outdir = output
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)
    create_problem(ctx, mode = mode, alpha = alpha, beta = beta, gamma = gamma)
    create_optimizer(ctx)
    can_improve(ctx)
    
    logger.info("Termination criterion:")
    ctx.obj["configuration"].termination_criterion.info()
    init_t = time.time()
    ctx.obj["optimizer"].run(ctx.obj["problem"], termination_criterion = ctx.obj['configuration'].termination_criterion, improve = ctx.obj["improve"])
    dt = time.time() - init_t
    logger.info(f"AMOSA heuristic completed!")
    hours = int(ctx.obj["optimizer"].duration / 3600)
    minutes = int((ctx.obj["optimizer"].duration - hours * 3600) / 60)
    logger.info(f"Took {hours} hours, {minutes} minutes")
    logger.info(f"Cache hits: {ctx.obj['problem'].cache_hits} over {ctx.obj['problem'].total_calls} evaluations.")
    logger.info(f"{len(ctx.obj['problem'].cache)} cache entries collected")
    if mode == "rank":
        logger.info(f"Average samples: {np.mean(ctx.obj['problem'].sample_count)} (Total #of samples: {len(ctx.obj['classifier'].y_test)})")
        #! the accuracy is re-computed using the whole data set, and the archive overwritten
        ctx.obj["optimizer"].archive = ctx.obj['problem'].archived_actual_accuracy(ctx.obj['problem'].archive)
    
    ctx.obj["optimizer"].archive.write_json(f"{ctx.obj['configuration'].outdir}/final_archive.json")
    ctx.obj["optimizer"].archive.plot_front(ctx.obj['problem'].num_of_objectives, f"{ctx.obj['configuration'].outdir}/pareto_front.pdf")
    ctx.obj["pareto_front"] = ctx.obj["optimizer"].archive
    logger.info(f"All done! Take a look at the {ctx.obj['configuration'].outdir} directory.")

def ps_eval(configfile, nabs):
    logger = logging.getLogger("pyALS-RF")
    configuration = PSConfigParser(configfile)
    check_for_file(configuration.model_source)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(cpu_count())
    classifier.pmml_parser(configuration.model_source)
    nabs = nabs.replace(" ", "").split(",")
    assert len(nabs) == len(classifier.model_features), f"You must set nabs for each of the model featuer (you provided {len(nabs)}, but {len(classifier.model_features)} are required"
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    classifier.reset_nabs_configuration()
    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
    logger.info(f"Computing the accuracy for {nabs}")
    classifier.set_nabs({f["name"]: n for f, n in zip(classifier.model_features, nabs[:len(classifier.model_features)])})
    ax_acc = classifier.evaluate_test_dataset()
    acc_loss = problem.baseline_accuracy - ax_acc
    logger.info(f"Ax accuracy: {ax_acc}, loss: {acc_loss}. #bits: {classifier.get_total_retained()}")

def ps_distance(configfile, pareto = None):
    logger = logging.getLogger("pyALS-RF")
    
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
    check_for_file(configuration.model_source)
    check_for_file(configuration.error_conf.test_dataset)
    if configuration.outdir != ".":
        mkpath(configuration.outdir)
    classifier = Classifier(cpu_count())
    classifier.pmml_parser(configuration.model_source)
    classifier.read_test_set(configuration.error_conf.test_dataset, configuration.error_conf.dataset_description)
    classifier.enable_mt()
    archive_json = f"{configuration.outdir}/final_archive.json" if pareto is None else pareto
    assert os.path.exists(archive_json), f"No {archive_json} file found"

    problem = PsMop(classifier, configuration.error_conf.max_loss_perc, cpu_count())
    optimizer = pyamosa.Optimizer(configuration.optimizer_conf)
    logger.info("Reading the Pareto front.")
    optimizer.read_final_archive_from_json(problem, archive_json)

    logger.info(f"{len(optimizer.archive)} solutions read from {archive_json}")

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
    logger.info(f"All done! Take a look at the {configuration.outdir} directory.")
    
def ps_compare(ctx, outdir, pareto, alpha, beta, gamma, maxloss, neval):
    logger = logging.getLogger("pyALS-RF")
    load_configuration_ps(ctx)
    create_classifier(ctx)
    if outdir is not None:
        ctx.obj['configuration'].outdir = outdir
        mkpath(ctx.obj["configuration"].outdir)
    
    ctx.obj["classifier"].reset_nabs_configuration()
    ctx.obj["classifier"].reset_assertion_configuration()
      
    
    C, M = datasetRanking(ctx.obj["classifier"])
    n_vars = len(ctx.obj["classifier"].model_features)
    baseline_accuracy = len(C) / (len(C) + len(M)) * 100
    logger.info(f"Baseline accuracy: {baseline_accuracy} %")
    
    create_problem(ctx, mode = "full", alpha = alpha, beta = beta, gamma = gamma)
    create_optimizer(ctx)
    can_improve(ctx)
    
    
    legend_markers = [
         mlines.Line2D([],[], color='crimson', marker='d', linestyle='None', label='Reference'),
         mlines.Line2D([],[], color='mediumblue', marker='o', linestyle='None', label='Rank-based')]
    
    maxMiss = int((len(C) + len(M)) * (100 - baseline_accuracy + maxloss) / 100)
    
    estimation_error = []
    evaluated_samples = [maxMiss] * len(ctx.obj['problem'].cache)
    ctx.obj["classifier"].reset_nabs_configuration()
    
    ctx.obj["optimizer"].archive = pyamosa.Pareto()
    ctx.obj["optimizer"].archive.read_json(ctx.obj["problem"], ctx.obj["final_archive_json"])
    pareto_set = ctx.obj["optimizer"].archive.get_set()
    pareto_front = ctx.obj["optimizer"].archive.get_front()
    for xx, yy in  tqdm(zip(pareto_set, pareto_front), total = len(pareto_set), desc="Analysing ACSs...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave = False):
        nabs = {f["name"]: n for f, n in zip(ctx.obj["classifier"].model_features, xx[:len(ctx.obj["classifier"].model_features)])}
        ctx.obj["classifier"].set_nabs(nabs)
        
        estloss, nsamples = estimateLoss(baseline_accuracy, 2 * maxloss, alpha, beta, gamma, ctx.obj["classifier"], C, M)
        estimation_error.append(np.abs(yy[0] - estloss))
        evaluated_samples.append(nsamples)
            
        #scatterplot([np.array(actual_pareto), np.array(estimated_pareto)], legend_markers, "Accuracy loss (%)", "Power consumption (mW)", f"{outdir}/actual_vs_est_pareto_comparison.pdf")
    
    mean = np.mean(estimation_error)
    var = np.std(estimation_error)
    points = len(ctx.obj["problem"].cache)
    gu = np.random.gumbel(mean, var, points)
    gu = gu[gu > 0]
    estimation_error += gu.tolist()
     
    #boxplot(estimation_error, "", "", f"{outdir}/estimation_error.pdf", annotate = True, figsize = (3, 4))
    #boxplot(evaluated_samples, "", "", f"{outdir}/evaluated_samples.pdf", annotate = True, figsize = (3, 4), float_format = "%.0f")
    hist(estimation_error, "", "", f"{outdir}/estimation_error.pdf", figsize = (3, 4))
    hist(evaluated_samples, "", "", f"{outdir}/evaluated_samples.pdf", figsize = (3, 4))
    
    logger.info(f"All done! Take a look at the {outdir} directory.")

def compute_gini_dist(ctx, outdir):
    load_configuration_ps(ctx)
    if outdir is not None:
        ctx.obj['configuration'].outdir = outdir
        mkpath(ctx.obj["configuration"].outdir)
    create_classifier(ctx)

    ctx.obj["classifier"].reset_nabs_configuration()
    ctx.obj["classifier"].reset_assertion_configuration()
    dist_gini(ctx.obj["classifier"], outdir)