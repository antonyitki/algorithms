# -*- coding: utf-8 -*-
"""part1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d7a7IM_lFCHjwj_qlAxycQ0OBJNCshT8
"""

################################################################################
#                                                                              #
# University of Roehampton, London (UK)                                        #
# YIT 19488399, Antonio Ramon Oliver                                           #
# Module: Algorithms                                                           #
# Coursework part 1 (20% value). Weeks 1 -3. Date: Monday, 25 October 2021.    #
#                                                                              #
################################################################################


import pandas as pd #module for files and data
import os #this module works with files


#validate user input. filename is the variable for name of the file
while (os.path.isfile("/uk_glx_open_retail_points_v20_202104.csv")):
    filename = (input('File should be readable by a computer: '))
    if filename == "/uk_glx_open_retail_points_v20_202104.csv":
        print("\n\n\n Well done!\n\n")
        break
    print('Try again')

# read specific columns of csv file using Pandas
df = pd.read_csv(filename, usecols = ['long_wgs','lat_wgs'])
print(df)

#calculation for total number of rows/records
num_lines = -1 #first line is name of the columns
with open(filename, 'r') as f:
    for line in f:
        num_lines += 1
print("Number of lines:\n", num_lines)


#selecting one row randomly (2 values from same row)
random_row = df.sample()
print("\nHere the lucky selection:")
print(random_row)