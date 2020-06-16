#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:16:01 2020

@author: sahil308
"""

import os
os.chdir('/Users/sahil308/Desktop/dummy')
import pandas as pd

from os import chdir
from glob import glob
import pandas as pdlib
import datetime

import re


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    # return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    return [tryint(c) for c in re.split('r[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l


# Produce a single CSV after combining all files
def produceOneCSV(list_of_files, file_out):
   # Consolidate all CSV files into one object
   result_obj = pd.concat((pd.read_csv(file, skiprows=6) for file in list_of_files), ignore_index=True)
   #result_obj = pd.concat((pd.read_csv(file, skiprows=6).assign(filename=file) for file in list_of_files), ignore_index=True)
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

# List all CSV files in the working dir
file_pattern = "csv"
x_files = [file for file in glob('*.{}'.format(file_pattern))]
sort_nicely(x_files)


file_out = "ConsolidateOutput2.csv"
produceOneCSV(list_of_files, file_out)
import numpy as np
def read_data(list_of_files, axistype, file_out):    
    for index, file in enumerate(list_of_files):
        timestamp = pd.read_csv(file, nrows= 1, skiprows=3)
        start=timestamp['ms'][0]
        print(start)
        data = pd.read_csv(file, skiprows=6)
        data.columns = data.columns.str.lower()
        data.columns = data.columns.str.replace('-', '_').str.replace(' ', '_')
        data['ref_timestamp'] = datetime.datetime.strptime(start, '%m/%d/%Y  %H:%M')
        data[f'timestamp_{axistype}'] = data['ref_timestamp'] + pd.to_timedelta(data['x_axis_value'], 'milliseconds')
        data.drop(columns=['ref_timestamp', 'x_axis_value'], inplace=True)
        data.rename({'y_axis_value': f'{axistype}_axis_value'}, axis='columns', inplace=True)
        data = data.iloc[:, np.r_[1,0]]
        
#        data['ts_str'] = data[f'timestamp_{axistype}'].apply(lambda x:datetime.datetime.strftime(x, '%m/%d/%Y  %H:%M:%S.%f'))
        
        if index == 0:
            data.to_csv(file_out, index=False)
        else:
            data.to_csv(file_out, index=False, mode='a', header=False)
    return data

print
            
dt = read_data(x_files, 'x', "ConsolidateOutput2.csv")
read_data(x_files, 'x', "ConsolidateOutput2.csv")
read_data(x_files, 'x', "ConsolidateOutput2.csv")



dt.info()

v = pd.read_csv('ConsolidateOutput2.csv')            




timestamp = pd.read_csv('1.csv', nrows= 5, skiprows=3)
start=timestamp['ms'][0]
type(start)

d1 = pd.read_csv('1.csv', skiprows=6)
d1.columns = d1.columns.str.lower()
d1.columns = d1.columns.str.replace('-', '_').str.replace(' ', '_')
d1.columns
d1['ref_timestamp'] = datetime.datetime.strptime(start, '%m/%d/%Y  %H:%M')
d1['timestamp'] = d1['ref_timestamp'] + datetime.timedelta(d1['x_axis_value'])

d1['timestamp'] = d1['ref_timestamp'] + pd.to_timedelta(d1['x_axis_value'], 'milliseconds')




df1 = pd.read_csv('1.csv', skiprows=5)


export-0
export-1







# Move to the path that holds our CSV files
#csv_file_path = 'c:/temp/csv_dir/'
#chdir(csv_file_path)

#c