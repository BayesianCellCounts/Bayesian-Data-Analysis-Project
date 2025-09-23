"""
Class implementing model visualisations with matplotlib.pyplot

by Mathis DA SILVA
"""


import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd


class Visualisation:


    def __init__(self):
        self.figure_size = (12, 8)

    def plot_model_comparison(self, traces_dict, comparison):
        """
        :param traces_dict: 
        :param comparison: 
        :return: 
        """
        plt.figure(figsize=self.figure_size)
        az.plot_comparision(comparison)
        plt.title("Comparison between models")
        plt.show()


    def plot_trace_diagnostics(self, trace, model_name):
        """
        :param trace:
        :param model_name:
        :return:
        """

        az.plot_trace(trace, var_names=['theta', 'tau'], compact = True)
        plt.suptitle(f"Diagnostics for {model_name}")
        plt.show()


    def plot_region_effects(self, trace, data, model_name):
        """
        :param trace:
        :param data:
        :param model_name:
        :return:
        """

        posterior = trace.posterior
        theta_mean = posterior['theta'].mean(dim = ['chain', 'draw'])

        results = []
        for r in range(data['n_regions']):
            for g in range(data['n_groups']):
                results.append({
                    'Region': data['region_names'][r],
                    'Group': data['group_names'][g],
                    'Theta': float(theta_mean[r,g])
                })
        results = pd.DataFrame(results)

        plt.figure(figsize=self.figure_size)
        sns.barplot(data = results, x='Region', y = 'Theta', hue = 'Group')
        plt.xticks(rotation = 45)
        plt.title(f"Effects on region and group of {model_name}")
        plt.show()