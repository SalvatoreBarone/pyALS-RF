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
import numpy as np
from distutils.dir_util import mkpath
from tqdm import tqdm
from .Classifier import *
from ..plot import boxplot, hist_and_boxplot

def softmax(x):
    e_x = np.exp(np.array(x, dtype = np.float64))
    return e_x / e_x.sum()

def dispersion(x, theta):
    return 1 - (1/(len(x)-1)) * np.sum(  (x - theta) ** 2 )

def giniImpurity(x):
    return len(x) / (len(x) - 1) * (1 - np.sum(np.square(x)))

def dist_gini(classifier, outdir):
    C = []
    M = []
    for tau, theta_star in tqdm(zip(classifier.x_test, classifier.y_test), total=len(classifier.y_test), desc="Computing the Gini index...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
        rho = softmax(classifier.predict_mt(tau))
        Ig = giniImpurity(rho)
        if  np.argmax(rho) == theta_star:
            C.append(Ig)
        else:
            M.append(Ig)
    mkpath(outdir)
    #boxplot(C, "", "", f"{outdir}/C_ginidist.pdf", figsize= (2, 4), annotate = True, float_format = "%.2f", fontsize = 13)
    #boxplot(M, "", "", f"{outdir}/M_ginidist.pdf", figsize= (2, 4), annotate = True, float_format = "%.2f", fontsize = 13) 
    hist_and_boxplot(C, r"$I_G({\rho})$", "# of samples", f"{outdir}/C_ginihist.pdf", figsize= (2, 4), annotate = True, float_format = "%.2f", fontsize = 13)
    hist_and_boxplot(M, r"$I_G({\rho})$", "# of samples", f"{outdir}/M_ginihist.pdf", figsize= (2, 4), annotate = True, float_format = "%.2f", fontsize = 13)
   

def datasetRanking(classifier):
    C = []
    M = []
    for index, (tau, theta_star) in tqdm(enumerate(zip(classifier.x_test, classifier.y_test)), total=len(classifier.y_test), desc="Dataset ranking...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
        rho = softmax(classifier.predict_mt(tau))
        Ig = giniImpurity(rho)
        if  np.argmax(rho) == theta_star:
            C.append((index, Ig))
        else:
            M.append((index, Ig))
    C.sort(key = lambda a: a[-1], reverse = True) # correctly classified samples have to be sorted in descending order
    M.sort(key = lambda a: a[-1])                 # misclassified samples have to be sorted in ascending order
    return C, M

def estimateLoss(eta, nu, alpha, beta, gamma, classifier, C, M):
    maxMiss = int((len(C) + len(M)) * (100 - eta + nu) / 100)
    tested_samples = 0
    miss = 0
    a = 0
    b = 0
    alpha_max = int(alpha * len(C))
    beta_max = int(beta * len(M))
    for index, _ in M: 
        tested_samples += 1
        if np.argmax(classifier.predict_mt(classifier.x_test[index])) == classifier.y_test[index]:
            miss -= 1
            b = 0
        else:
            b +=1
        if b >= beta_max:
            break
    for index, _ in C:
        tested_samples += 1
        if np.argmax(classifier.predict_mt(classifier.x_test[index])) == classifier.y_test[index]:
            a += 1
        else:
            miss += 1
            a /= gamma
        if miss >= maxMiss or a >= alpha_max:
            break
    return miss / (len(C) + len(M)) * 100, tested_samples

