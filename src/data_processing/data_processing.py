"""
Class implementing the processing of data to be provided for each model

by Mathis DA SILVA
"""


import pandas as pd


class DataProcessing:


    def __init__(self, data, processed_data):
        self.data = data
        self.processed_data = processed_data


    def load_data(self, file_path:str):
        self.data = pd.read_excel(file_path)
        return self.data

