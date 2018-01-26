"""Analysis classification result data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(file_name: str):
    df = pd.read_csv(file_name, index_col=0)
    return df


def _guess_top_category(df: pd.DataFrame):
    columns = df.columns.drop('file', errors='ignore')
    avg = [df[col].sum() for col in columns]
    idx = max(range(len(avg)), key=avg.__getitem__)
    return columns[idx], idx


def summarize(df: pd.DataFrame, save_false_result=False):
    df_new = df.drop(labels='file', axis=1, errors='ignore')
    name, col_idx = _guess_top_category(df_new)
    total = len(df_new)
    avg = df_new[name].sum() / total
    maximal = df_new[name].max()
    minimal = df_new[name].min()
    print("Top-1 confidence, avg: {0}, max: {1}, min:{2}".format(avg, maximal, minimal))
    row_correct = np.argmax(df_new.values, axis=1) == col_idx
    num_correct = sum(row_correct)
    print("Pass rate: {0}%. {1}/{2}".format(num_correct / total, num_correct, total))
    if save_false_result is True:
        incorrect = df.loc[row_correct == False]
        incorrect.to_csv('false_result.csv')


def plot_csv(df: pd.DataFrame):
    columns = df.columns.drop('file', errors='ignore')
    for c in columns:
        plt.plot(df[c])
    plt.show()


if __name__ == "__main__":
    df = load_csv("result.csv")
    summarize(df, True)
