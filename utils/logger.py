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

def create_dict(keys):
    myDict = {key: None for key in keys}
    return myDict

def update_dict(file, key, val):
    file[key] = value

def dict_to_csv(file, keys, name):
    csv_file = name+".csv"
    with open(file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
