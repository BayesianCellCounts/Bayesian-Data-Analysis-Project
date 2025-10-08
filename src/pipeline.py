"""
Class implementing the pipeline for Bayesian models

by Mathis DA SILVA
"""


from data_processing.data_processing import DataProcessing
from models.poisson_model import PoissonModel
from models.zero_inflated_poisson import ZeroInflatedPoissonModel
from visualisation import Visualisation
import arviz as az
from multiprocessing import freeze_support


class Pipeline:

    def __init__(self):
        self.data_processor = None
        self.models = {}
        self.traces = {}
        self.visualizer = Visualisation()

    def load_and_process_data(self, file_path):
        self.data_processor = DataProcessing(file_path)
        self.data_processor.load_data(file_path)
        return self.data_processor.prepare_data()

    def initialize_models(self):
        self.models = {
            'Poisson': PoissonModel(),
            'Zero-Inflated Poisson': ZeroInflatedPoissonModel()
        }

    def fit_models(self, data, draws=500, chains=2):
        for name, model in self.models.items():
            print(f"Ajusting {name} model : \n")
            model.build_model(data)

            # Ajustement simple - les paramètres sont gérés dans model.fit()
            self.traces[name] = model.fit(
                draws=draws,
                chains=chains,
                target_accept=0.90,
                tune=1000  # Plus de tuning pour meilleure convergence
            )

    def compare_models(self):
        if len(self.traces) < 2:
            print("Comparaison impossible : moins de 2 modèles ajustés")
            return None

        try:
            comparison = az.compare(self.traces)
            self.visualizer.plot_model_comparison(self.traces, comparison)
            return comparison
        except Exception as e:
            print(f"Erreur lors de la comparaison : {e}")
            print("Calcul des WAIC individuels à la place...")
            for name, trace in self.traces.items():
                try:
                    waic = az.waic(trace)
                    print(f"{name}: WAIC = {waic.waic:.1f}")
                except Exception as e2:
                    print(f"{name}: Erreur WAIC - {e2}")
            return None

    def analyse_best_model(self, data, comparison):
        best_model_name = comparison.index[0]
        best_trace = self.traces[best_model_name]

        self.visualizer.plot_trace_diagnostics(best_trace, best_model_name)
        self.visualizer.plot_region_effects(best_trace, data, best_model_name)

    def run_pipeline(self, file_path, draws = 500, chains = 2):
        data = self.load_and_process_data(file_path)

        self.initialize_models()
        self.fit_models(data, draws, chains)

        comparison = self.compare_models()

        self.analyse_best_model(data, comparison)

if __name__ == '__main__':
    freeze_support()  # Nécessaire pour Windows
    pipeline = Pipeline()
    pipeline.run_pipeline('../data/dataset_neuroscience_3.xlsx')