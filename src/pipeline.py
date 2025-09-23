"""
Class implementing the pipeline for Bayesian models

by Mathis DA SILVA
"""


from .data_processing.data_processing import DataProcessing
from .models.poisson_model import PoissonModel
from .models.zero_inflated_poisson import ZeroInflatedPoissonModel
from .visualisation import Visualisation
import arviz as az

class Pipeline:

    def __init__(self, data_processor: None):
        self.data_processor = data_processor
        self.models = {}
        self.traces = {}
        self.visualizer = Visualisation()

    def load_and_process_data(self, file_path):
        self.data_processor = DataProcessing(file_path)
        self.data_processor.load_data(file_path)
        return self.data_processor.process_data()

    def initialize_models(self):
        self.models = {
            'Poisson': PoissonModel(),
            'Zero-Inflated Poisson': ZeroInflatedPoissonModel()
        }

    def fit_models(self, data, draws = 1000, chains = 4):
        for name, model in self.models.items():
            print(f"Ajusting {name} model : \n")
            model.build_model(data)
            self.traces[name] = model.fit(draws = draws, chains = chains)

    def compare_models(self):
        comparison = az.compare(self.traces)
        self.visualizer.plot_model_comparison(self.traces, comparison)
        return comparison

    def analyse_best_model(self, data, comparison):
        best_model_name = comparison.index[0]
        best_trace = self.traces[best_model_name]

        self.visualizer.plot_trace(best_trace, best_model_name)

        self.visualizer.plot_results(best_trace, data, best_model_name)

    def run_pipeline(self, file_path, draws = 1000, chains = 4):
        data = self.load_and_process_data(file_path)

        self.initialize_models()
        self.fit_models(data, draws, chains)

        comparison = self.compare_models()

        self.analyse_best_model(data, comparison)