"""Analysis classification result data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_csv_files(path):
    path = os.path.abspath(path)
    results = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]
    print(results)
    return results


def load_csv(file_name: str):
    df = pd.read_csv(file_name, index_col=0)
    return df


def _guess_top_category(df: pd.DataFrame):
    columns = df.columns.drop('file', errors='ignore')
    avg = [df[col].sum() for col in columns]
    idx = max(range(len(avg)), key=avg.__getitem__)
    return columns[idx], idx


def summarize_label(df_in, label='', df_out=None, save_false_result=False, path='false_result.csv'):
    df_new = df_in.drop(labels='file', axis=1, errors='ignore')
    if len(df_new) <= 0:
        return

    if label == '' or label is None:
        label, col_idx = _guess_top_category(df_new)
    else:
        try:
            col_idx = df_new.columns.get_loc(label)
        except KeyError:
            print("Can't find label '{0}'".format(label))
            return

    total = len(df_new)
    row_correct = np.argmax(df_new.values, axis=1) == col_idx
    num_correct = sum(row_correct)
    print("category: {0}".format(label))
    print("Pass rate: {0:.3f}%. {1}/{2}".format(num_correct * 100.0 / total, num_correct, total))

    avg = df_new[label].sum() / total
    maximal = df_new[label].max()
    minimal = df_new[label].min()
    print("Top-1 confidence: (avg: {0:.3f}%, max: {1:.3f}%, min:{2:.3f}%)\n".format(avg * 100,
          maximal * 100, minimal * 100))

    if save_false_result is True:
        incorrect = df_in.loc[row_correct == False]
        incorrect.to_csv(path)
    if df_out is not None:
        row = {'label': label, 'pass-rate': num_correct / total, 'avg': avg,
                'max': maximal, 'min': minimal, 'passed': num_correct, 'total': total}
        df_out.loc[len(df_out)] = row


def plot_csv(df: pd.DataFrame):
    columns = df.columns.drop('file', errors='ignore')
    for c in columns:
        plt.plot(df[c])
    plt.show()


def _get_image_label(path):
    assert path.endswith('.csv')
    return os.path.basename(path)[:-4]


def summarize(path):
    files = get_csv_files(path)
    df_out = pd.DataFrame(columns=['label', 'pass-rate', 'avg', 'max', 'min', 'passed', 'total'])
    for f in files:
        df = load_csv(f)
        label = _get_image_label(f)
        summarize_label(df, label, df_out)
    return df_out


if __name__ == "__main__":
    df = summarize('data/')
    print(df)

