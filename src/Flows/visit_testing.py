import logging, joblib, numpy as np
from distutils.dir_util import mkpath
from itertools import combinations, product
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from .TMR.tmr import TMR
import os 


def visit_test(ctx):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)

    create_classifier(ctx)   
    classifier = ctx.obj["classifier"]
    classifier.get_dbs_vectors()
   