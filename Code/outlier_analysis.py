# -*- coding: utf-8 -*-
"""
Erstellt am 02.07.2022

@author: Nils Heimbach, Christian T. Seidler

Funktionen zur Überprüfung auf Ausreißer (Spikes und Flats).
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def show_boxplot(y, data, title):
    """Erzeugen und Anzeigen eines Boxplots für das angegebene Merkmal."""

    plt.figure()
    sns.boxplot(y=y, data=data)
    plt.title(title)
    plt.show()


def show_histogramm(x, data, title, kde=True):
    """Erzeugen und Anzeigen eines Histogramms für das angegebene Merkmal."""

    plt.figure()
    sns.histplot(x=x, data=data, kde=kde)
    plt.title(title)
    plt.show()


def calc_and_print_outliers_with_z_score(data, column, threshold=3):
    """Berechnen und Anzeigen der Datenpunkte, deren z-Score größer als der
    angegebene Threshold ist (--> Spikes)."""

    z_scores = calc_outliers_with_z_score(data, column)
    print_outliers_from_z_score(z_scores, threshold, column)


def calc_outliers_with_z_score(data, column):
    """Berechnen des z-Scores für jeden Datenwert der angegebenen Spalte."""

    z = np.abs(stats.zscore(data[column]))
    return z


def print_outliers_from_z_score(z_scores, threshold, column):
    """Ausgeben der Datenwerte, die einen z-Score größer als den threshold
    haben."""

    print("Outliers in Column", column + ":", np.where(z_scores > threshold))
    logging.info("Anzahl der Ausreißer über dem Treshhold von %i sind %i .",
                 threshold, len(np.where(z_scores > threshold))-1)


def calc_and_print_outliers_with_iqr(data, column, threshold=1.5):
    """Berechnen und Anzeigen der Datenpunkte, deren Interquartilabstand größer
    als der angegebene Threshold ist (--> Spikes)."""

    upper, lower = calc_outliers_with_iqr(data, column, threshold)
    print_outliers_from_iqr(upper, lower, column)


def calc_outliers_with_iqr(data, column, threshold):
    """Berechnen der Datenwerte, deren IQR größer als der angegebene Threshold
    ist."""

    q1, q3 = calc_q1_and_q3(data, column)
    iqr = q3 - q1
    # Above Upper bound
    upper = data[column] >= (q3 + threshold*iqr)
    # Below Lower bound
    lower = data[column] <= (q1 - threshold*iqr)
    logging.info('Anzahl der Ausreißer über dem Treshhold %i .', upper.sum())
    logging.info("Anzahl der Ausreißer unter dem Treshhold %i .", lower.sum())
    return upper, lower


def calc_q1_and_q3(data, column):
    """Berechnen von 25-Percentil und 75-Percentil für die anegegebene
    Spalte."""

    q1 = np.percentile(data[column], 25, method='midpoint')
    q3 = np.percentile(data[column], 75, method='midpoint')
    return q1, q3


def print_outliers_from_iqr(upper, lower, column):
    """Anzeigen der Datenwerte, deren IQR größer dem angegebenen Threshold
    ist."""

    print("Upper Outliers in Column", column + ":", np.where(upper))
    print("Lower Outliers in Column", column + ":", np.where(lower))


def print_num_of_flats(data, column, threshold=0):
    """Berechnen und Anzeigen der Datenwerte, deren Differenz zum Vorwert
    kleiner gleich dem angegebenen Threshold ist (--> Flats)."""

    df = data.copy()
    df[column + '_diff'] = df[column].diff()
    print("Flats in column", column + ":",
          df[df[column + '_diff'].abs() <= threshold])
    logging.info('Die Anzahl an Flats der Spalte %s sind %i.', column,
                 len(df[df[column + '_diff'].abs() <= threshold]))
