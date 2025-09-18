"""
Class implementing the base of all models

by Mathis DA SILVA
"""

class BayesianModel:

    def __init__(self, name, fitted = False, parameters = None):
        self.name = name
        self.fitted = fitted
        self.parameters = parameters
