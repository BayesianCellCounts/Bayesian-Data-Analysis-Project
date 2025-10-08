"""
Class implementing the Poisson model

by Mathis DA SILVA
"""


from models.models import BayesianModel
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

            tau = pm.HalfNormal('tau', sigma=np.log(1.05),
                                shape=(data['n_regions'], data['n_groups']))

            # Effets individuels pour chaque observation
            gamma = pm.Normal('gamma',
                              mu=theta[data['region_idx'], data['group_idx']],
                              sigma=tau[data['region_idx'], data['group_idx']],
                              shape=len(data['counts']))

            # Paramètre du taux de Poisson
            lambda_i = pm.math.exp(gamma)

            # Vraisemblance
            y_obs = pm.Poisson('y_obs', mu=lambda_i, observed=data['counts'])

            pm.Deterministic('log_likelihood', pm.logp(y_obs, data['counts']))

        self.model = model
        return model