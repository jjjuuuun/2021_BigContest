import pandas as pd
import os
import numpy as np

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)

hts.drop(['G_ID', 'PIT_ID', 'T_ID', 'INN', 'STADIUM', 'PIT_VEL', 'HIT_RESULT'], axis=1, inplace=True)

sort_hts = hts.sort_values(by=['PCODE','GYEAR'],ascending=True,ignore_index=True)

pcode_list = []
for i in range(len(sort_hts)):
    if not int(sort_hts.loc[i]['PCODE']) in pcode_list:
        pcode_list.append(int(sort_hts.loc[i]['PCODE']))

# =============================================================================
# new_sort_hts = pd.DataFrame()
# 
# for i in pcode_list:
#     new_sort_hts = sort_hts[sort_hts['PCODE'] == i]
#     
# =============================================================================   
    
    
def limit_line(x):
    up_slope = 0.15562915209790212
    up_intercept = 32.48832867132867
    low_slope = -0.14348317307692315
    low_intercept = 23.852230769230772
    
    return up_slope * x + up_intercept, low_slope * x + low_intercept


def barrel(df):
    upper, lower = limit_line(df.HIT_VEL)
    if upper >= df.HIT_ANG_VER and lower <= df.HIT_ANG_VER:
        return 1
    else:
        return 0
    

sort_hts['barrel'] = sort_hts.apply(barrel, axis = 1)
    
    
    
    
    
    
    
    