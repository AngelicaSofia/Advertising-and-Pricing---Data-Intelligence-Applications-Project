import numpy as np
import pandas as pd


def create_table(n_rows, n_columns):
    """Function to create an empty matrix for bid price"""
    n_rows += 1
    n_columns += 1
    table = np.zeros((n_rows, n_columns))
    return table


def update_table(table, row, column, content):
    """Function to update the matrix content"""
    table[row][column] = content


def table_to_csv(table, name):
    """Function to save the table into a csv file"""
    pd.DataFrame(table).to_csv(name+".csv")

