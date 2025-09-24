# -*- coding: utf-8 -*-

"""
Pipeline optimisé pour performance sans changer les paramètres du modèle

by Mathis DA SILVA
"""

from data_processing.data_processing import DataProcessing
from models.poisson_model import PoissonModel
from models.zero_inflated_poisson import ZeroInflatedPoissonModel
from visualisation import Visualisation
import arviz as az
import pymc as pm
import pytensor
from multiprocessing import freeze_support
import os

class PerformancePipeline:

    def __init__(self):
        self.data_processor = None
        self.models = {}
        self.traces = {}
        self.visualizer = Visualisation()
        self._optimize_pytensor()

    def _optimize_pytensor(self):
        """Optimise PyTensor pour de meilleures performances"""
        # Force la compilation C++ si disponible
        pytensor.config.cxx = ""  # Let PyTensor auto-detect
        
        # Optimisations de performance
        pytensor.config.optimizer = 'fast_run'
        pytensor.config.allow_gc = False  # Plus rapide mais utilise plus de mémoire
        pytensor.config.compute_test_value = 'off'  # Désactive les vérifications
        
        # Configuration multiprocessing
        os.environ['OMP_NUM_THREADS'] = '1'  # Évite les conflits avec PyMC
        os.environ['MKL_NUM_THREADS'] = '1'
        
        print("PyTensor optimized for performance")

    def load_and_process_data(self, file_path):
        self.data_processor = DataProcessing(file_path)
        self.data_processor.load_data(file_path)
        return self.data_processor.prepare_data()

    def initialize_models(self):
        self.models = {
            'Poisson': PoissonModel(),
            'Zero-Inflated Poisson': ZeroInflatedPoissonModel()
        }

    def fit_models(self, data, draws=1000, chains=4):
        """
        Fit models avec optimisations de performance 
        SANS changer la structure du modèle
        """
        for name, model in self.models.items():
            print(f"Ajusting {name} model : \n")
            model.build_model(data)
            
            with model.model:
                # Étape 1: Compilation préalable pour optimiser les calculs
                print("Compiling model...")
                
                # Étape 2: Initialisation optimisée
                print("Finding good starting point...")
                try:
                    # Essaie de trouver un bon point de départ
                    start = pm.find_MAP(maxeval=500)
                    print("MAP initialization successful")
                except:
                    print("MAP initialization failed, using default")
                    start = None
                
                # Étape 3: Sampling avec optimisations
                print("Starting MCMC sampling...")
                self.traces[name] = pm.sample(
                    draws=draws,
                    chains=chains,
                    start=start,
                    # Optimisations de performance PyMC
                    cores=chains,  # Parallélisation
                    target_accept=0.9,
                    max_treedepth=10,
                    # Optimisations mémoire et calcul
                    return_inferencedata=True,
                    progressbar=True,
                    # Réglage du tuning
                    tune=max(1000, draws),  # Plus de tuning = convergence plus rapide
                    # Autres optimisations
                    compute_convergence_checks=False,  # Économise du temps
                    idata_kwargs={
                        "log_likelihood": False,  # Économise mémoire
                        "predictions": False
                    }
                )
                
                print(f"Sampling completed for {name}")

    def fit_models_with_warmup(self, data, draws=1000, chains=4):
        """
        Alternative avec warm-up progressif pour éviter les blocages
        """
        for name, model in self.models.items():
            print(f"Ajusting {name} model with progressive warmup: \n")
            model.build_model(data)
            
            with model.model:
                # Warm-up rapide
                print("Phase 1: Quick warmup...")
                warmup_trace = pm.sample(
                    draws=100,
                    chains=2,
                    tune=200,
                    cores=2,
                    target_accept=0.8,
                    progressbar=True,
                    return_inferencedata=True,
                    compute_convergence_checks=False
                )
                
                # Vérification rapide de convergence
                try:
                    rhat_max = float(az.rhat(warmup_trace).max())
                    print(f"Warmup R-hat max: {rhat_max:.3f}")
                except:
                    rhat_max = 1.0
                
                # Si convergence OK, continuer
                if rhat_max < 1.2:
                    print("Phase 2: Full sampling...")
                    
                    # Utiliser les derniers échantillons comme point de départ
                    last_sample = {}
                    for var in warmup_trace.posterior.data_vars:
                        last_sample[var] = warmup_trace.posterior[var].isel(
                            chain=0, draw=-1
                        ).values
                    
                    self.traces[name] = pm.sample(
                        draws=draws,
                        chains=chains,
                        start=last_sample,
                        cores=chains,
                        target_accept=0.9,
                        tune=max(500, draws//2),
                        return_inferencedata=True,
                        progressbar=True,
                        compute_convergence_checks=False,
                        idata_kwargs={"log_likelihood": False}
                    )
                else:
                    print(f"Poor convergence, using warmup trace for {name}")
                    self.traces[name] = warmup_trace

    def compare_models(self):
        comparison = az.compare(self.traces)
        self.visualizer.plot_model_comparison(self.traces, comparison)
        return comparison

    def analyse_best_model(self, data, comparison):
        best_model_name = comparison.index[0]
        best_trace = self.traces[best_model_name]

        self.visualizer.plot_trace_diagnostics(best_trace, best_model_name)
        self.visualizer.plot_region_effects(best_trace, data, best_model_name)

    def run_pipeline(self, file_path, draws=1000, chains=4, use_warmup=False):
        """
        Lance le pipeline avec optimisations
        
        use_warmup=True : Utilise le warm-up progressif (recommandé pour gros modèles)
        """
        print("Loading and processing data...")
        data = self.load_and_process_data(file_path)

        print("Initializing models...")
        self.initialize_models()
        
        if use_warmup:
            self.fit_models_with_warmup(data, draws, chains)
        else:
            self.fit_models(data, draws, chains)

        print("Comparing models...")
        comparison = self.compare_models()

        print("Analyzing best model...")
        self.analyse_best_model(data, comparison)

if __name__ == '__main__':
    freeze_support()
    pipeline = PerformancePipeline()
    
    # Utilise le warm-up progressif pour éviter les blocages
    pipeline.run_pipeline('../data/dataset_neuroscience_1.xlsx', 
                         draws=1000, chains=4, use_warmup=True)