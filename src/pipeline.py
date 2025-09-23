"""
Class implementing the pipeline for Bayesian models

by Mathis DA SILVA
"""


from data_processing.data_processing import DataProcessing
from models.poisson_model import PoissonModel
from models.zero_inflated_poisson import ZeroInflatedPoissonModel
from visualisation import Visualisation
import arviz as az

class Pipeline:

    def __init__(self):
        pass