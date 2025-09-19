"""
Class implementing the base of all models

by Mathis DA SILVA
"""


import pymc as pm


class BayesianModel:


    def __init__(self, name: str, model, trace):
        self.name = name
        self.model = model
        self.trace = trace


    def build_model(self, datasets: dict):
        pass


    def fit(self, draws = 1000, chains = 4, target_accept = 0.8):
        with self.model:
            self.trace = pm.sample(
                draws = draws,
                chains = chains,
                target_accept = target_accept,
                return_inferencedata = True
            )
        return self.trace