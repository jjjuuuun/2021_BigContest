import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# user_path = 'C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/'
# =============================================================================
user_path = input()
os.chdir(user_path)

########################################################
batter_df = pd.read_csv('bat_except_all.csv')

batter_2018 = batter_df[(batter_df['GYEAR'] == 2018)]
batter_2019 = batter_df[(batter_df['GYEAR'] == 2019)]
batter_2020 = batter_df[(batter_df['GYEAR'] == 2020)]
batter_2021 = batter_df[(batter_df['GYEAR'] == 2021)]

list_2018 = list(batter_2018['PCODE'].values)
list_2019 = list(batter_2019['PCODE'].values)
list_2020 = list(batter_2020['PCODE'].values)
list_2021 = list(batter_2021['PCODE'].values)

up_slope = 0.15562915209790212
up_intercept = 32.48832867132867
low_slope = -0.14348317307692315
low_intercept = 23.852230769230772

def limit_line(x):
    global up_slope, up_intercept, low_slope, low_intercept
    
    return up_slope * x + up_intercept, low_slope * x + low_intercept

def check_barrel(df):
    upper, lower = limit_line(df.HIT_VEL)
    if upper >= df.HIT_ANG_VER and lower <= df.HIT_ANG_VER:
        return 1
    else:
        return 0

hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

hts_2018 = hts_2018[hts_2018['PCODE'].isin(list_2018)]
hts_2019 = hts_2019[hts_2019['PCODE'].isin(list_2019)]
hts_2020 = hts_2020[hts_2020['PCODE'].isin(list_2020)]
hts_2021 = hts_2021[hts_2021['PCODE'].isin(list_2021)]

hts_2018.drop(['T_ID', 'INN'], axis=1, inplace=True)
hts_2019.drop(['T_ID', 'INN'], axis=1, inplace=True)
hts_2020.drop(['T_ID', 'INN'], axis=1, inplace=True)
hts_2021.drop(['T_ID', 'INN'], axis=1, inplace=True)

hts_2018['barrel'] = hts_2018.apply(check_barrel, axis = 1)
hts_2019['barrel'] = hts_2019.apply(check_barrel, axis = 1)
hts_2020['barrel'] = hts_2020.apply(check_barrel, axis = 1)
hts_2021['barrel'] = hts_2021.apply(check_barrel, axis = 1)

hts_2018.to_csv('hts_barrel_2018.csv', index = False)
hts_2019.to_csv('hts_barrel_2019.csv', index = False)
hts_2020.to_csv('hts_barrel_2020.csv', index = False)
hts_2021.to_csv('hts_barrel_2021.csv', index = False)