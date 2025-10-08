"""
Class implementing the base of all models

by Mathis DA SILVA
"""


import pymc as pm


class BayesianModel:


    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.trace = None


    def build_model(self, datasets: dict):
        raise NotImplementedError


    def fit(self, draws=500, chains=2, target_accept=0.85, tune=1000):
        if self.model is None:
            raise RuntimeError('model not initialized')

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                chains=chains,
                tune=tune,
                target_accept=target_accept,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': True}  # Force le calcul de log_likelihood
            )
        return self.trace