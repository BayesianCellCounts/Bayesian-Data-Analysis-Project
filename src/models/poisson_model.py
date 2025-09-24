"""
Class implementing the Poisson model

by Mathis DA SILVA
"""


from .models import BayesianModel
import pymc as pm
import numpy as np

class PoissonModel(BayesianModel):


    def __init__(self):
        super().__init__(name = "Poisson")


    def build_model(self, data):


        with pm.Model() as model:
            # Hyperpriors pour chaque combinaison région-groupe
            theta = pm.Normal('theta', mu=5, sigma=2,
                              shape=(data['n_regions'], data['n_groups']))

            tau = pm.HalfNormal('tau', sigma=0.5,
                                shape=(data['n_regions'], data['n_groups']))

            gamma_raw = pm.Normal('gamma_raw', mu=0,sigma=1,
                                shape=(data['n_regions'], data['n_groups']))

            gamma = pm.Deterministic('gamma', theta + tau * gamma_raw)

            # Paramètre du taux de Poisson
            lambda_i = pm.math.exp(gamma[data['region_idx'], data['group_idx']])

            # Vraisemblance
            y_obs = pm.Poisson('y_obs', mu=lambda_i, observed=data['counts'])

        self.model = model
        return model