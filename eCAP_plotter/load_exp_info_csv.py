# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:24:28 2023

@author: Rex Chen

Description: This function loads the experimental information from a CSV file and assigns the values to corresponding NumPy arrays. The input parameter is the name of the CSV file. The output of the function is five NumPy arrays containing the experimental information: file_, Dir, Status, Date, and Path.

Input: csv_file: A string representing the name of the CSV file containing the experimental information.

Output:
    file_: A NumPy array of shape (50, 50) containing the filenames of the experimental data.
    Dir: A NumPy array of shape (50, 50) containing the directories of the experimental data.
    Status: A NumPy array of shape (50, 50) containing the status of the experimental data.
    Date: A NumPy array of shape (50,) containing the date of the experiment.
    Path: A NumPy array of shape (50,) containing the path of the experiment.
"""

import pandas as pd
import numpy as np
import os

def load_exp_info_csv(csv_file,file_location):
    # Load the DataFrame from a CSV file
    exp_info = pd.read_csv(csv_file, encoding='ISO-8859-1')
    
    file_ = np.empty((50, 50), dtype=object)
    Dir = np.empty((50, 50), dtype=object)
    Status = np.empty((50, 50), dtype=object)
    Date = np.empty((50), dtype=object)
    Path = np.empty((50), dtype=object)

    # Iterate through the rows of the loaded DataFrame
    for index_, row in exp_info.iterrows():
        # Skip empty rows (where exp_no is NaN)
        if pd.isna(row['exp_no']):
            continue

        exp_no = int(row['exp_no']) - 1
        cond_no = int(row['cond_no'])

        # Assign values from the DataFrame to the corresponding arrays
        file_[exp_no, cond_no] = row['file']
        Dir[exp_no, cond_no] = file_location + row['path'] + row['file']
        Status[exp_no, cond_no] = row['status']
        Date[exp_no] = row['date']
        Path[exp_no] = row['path']
        
    df_exp_info = exp_info
    return file_, Dir, Status, Date, Path, df_exp_info
