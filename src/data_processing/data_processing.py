"""
Class implementing the processing of data to be provided for each model

by Mathis DA SILVA
"""


import pandas as pd
import numpy as np


class DataProcessing:


    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.processed_data = None


    def load_data(self, file_path:str):
        self.data = pd.read_excel(self.file_path)
        return self.data


    def prepare_data(self):
        dataset = self.data

        data_clean = dataset.iloc[:-1].copy()

        metadata = data_clean[['abbreviation','region name','brain area']]
        count_data = data_clean.iloc[:, 3:]

        count_data_long = []
        region_id = []
        group_id = []

        for region_idx, region in enumerate(metadata['abbreviation']):
            for col_idx, column in enumerate(count_data.columns):
                group = column.split(' ')[0]
                count = count_data.iloc[region_idx, col_idx]

                count_data_long.append(count)
                region_id.append(region_idx)
                group_id.append(0 if  'A-SSRI' in column else 1)

        self.processed_data = {
            'counts': np.array(count_data_long),
            'region_idx': np.array(region_id),
            'groupe_idx': np.array(group_id),
            'n_regions': len(metadata),
            'n_groups': 2,
            'region_names': metadata['abbreviation'].tolist(),
            'group_names': ['A-SSRI','C-SSRI']
        }

        return self.processed_data