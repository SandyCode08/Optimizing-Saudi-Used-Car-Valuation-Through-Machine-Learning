# custom_transformers.py
import pandas as pd

TOP_15_MAKE = ['Toyota',
 'Hyundai',
 'Ford',
 'Chevrolet',
 'Nissan',
 'GMC',
 'Kia',
 'Mercedes',
 'Lexus',
 'Mazda',
 'Honda',
 'BMW',
 'Dodge',
 'Mitsubishi',
 'Land Rover']

TOP_20_TYPE = ['Land Cruiser',
 'Camry',
 'Hilux',
 'Accent',
 'Yukon',
 'Sonata',
 'Tahoe',
 'Taurus',
 'Elantra',
 'Corolla',
 'Expedition',
 'Furniture',
 'Suburban',
 'Prado',
 'Patrol',
 'Accord',
 'S',
 'Range Rover',
 'ES',
 'Yaris']

def reduce_make(data_saudi_feature_selection):
    data_saudi_feature_selection['Make_reduced'] = data_saudi_feature_selection['Make'].where(data_saudi_feature_selection['Make'].isin(TOP_15_MAKE), 'Other')
    return data_saudi_feature_selection[['Make_reduced']]

def reduce_type(data_saudi_feature_selection):
    data_saudi_feature_selection['Type_reduced'] = data_saudi_feature_selection['Type'].where(data_saudi_feature_selection['Type'].isin(TOP_20_TYPE), 'Other')
    return data_saudi_feature_selection[['Type_reduced']]

def to_int_transform(x):
    return x.astype(int)
