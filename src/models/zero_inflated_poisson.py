"""
Class implementing the Zero Inflated Poisson model

by Mathis DA SILVA
"""

from models import BayesianModel
import pymc as pm
import numpy as np


class ZeroInflatedPoissonModel(BayesianModel):


    def __init__(self):
        super().__init__(name = "Zero-Inflated Poisson")


    def build_model(self, data):

        with pm.Model() as model:

            pi = pm.Beta('pi', alpha=1, beta=5)

            theta = pm.Normal('theta', mu=5, sigma=2,
                              shape=(data['n_regions'], data['n_groups']))

            tau = pm.HalfNormal('tau', sigma=np.log(1.05),
                                shape=(data['n_regions'], data['n_groups']))

            gamma = pm.Normal('gamma',
                              mu=theta[data['region_idx'], data['group_idx']],
                              sigma=tau[data['region_idx'], data['group_idx']],
                              shape=len(data['counts']))

            lambda_i = pm.math.exp(gamma)

            y_obs = pm.ZeroInflatedPoisson('y_obs', mu=lambda_i, psi=pi,
                                           observed=data['counts'])

        self.model = model
        return model