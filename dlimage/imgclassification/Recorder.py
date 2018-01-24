import pandas as pd
import numpy as np


class Recorder(object):
    def __init__(self, labels):
        labels.append('file')
        self.df = pd.DataFrame(columns=labels)

    def add_result(self, result, file_name=""):
        result = list(np.squeeze(result))
        result.append(file_name)
        self.df.loc[len(self.df)] = result
        return self.df

    def sort_result(self, result):
        result = np.squeeze(result)
        index = result.argsort()[-5:][::-1]
        return index

    def save(self, file_name=''):
        self.df.to_csv(file_name)
