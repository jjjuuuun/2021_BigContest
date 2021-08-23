# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:07:41 2021

@author: sj545
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir("C:/Users/sj545/OneDrive/바탕 화면/2021_BigContest")

batter = pd.read_csv("add_potential.csv")
batter['PCODE'] = round(batter['PCODE'])
data = batter[['PCODE','GYEAR']]
#print(data)
#print(len(data['PCODE'].unique()))
def count_year(data):
    until_2018 = []
    until_2019 = []
    until_2020 = []
    until_2021 = []
    only_2021 = []
    for idx in range(1,data.shape[0]-1):
        if data.PCODE[idx] != data.PCODE[idx-1] and data.PCODE[idx] != data.PCODE[idx+1] and data.GYEAR[idx] == 2021:
            only_2021.append(data.GYEAR[idx])
        elif data.PCODE[idx] != data.PCODE[idx+1]:
            if data.GYEAR[idx] == 2018:
                until_2018.append(data.GYEAR[idx])
            elif data.GYEAR[idx] == 2019:
                until_2019.append(data.GYEAR[idx])
            elif data.GYEAR[idx] == 2020:
                until_2020.append(data.GYEAR[idx])
            elif data.GYEAR[idx] == 2021:
                until_2021.append(data.GYEAR[idx])
        else:
            pass
        if idx == data.shape[0]-2:
            until_2018.append(data.GYEAR[idx+1])
            
    until_2018 = len(until_2018)
    until_2019 = len(until_2019)
    until_2020 = len(until_2020)
    until_2021 = len(until_2021)
    only_2021 = len(only_2021)
    return until_2018, until_2019, until_2020, until_2021, only_2021

until_2018,until_2019,until_2020,until_2021,only_2021 = count_year(data)

count_df = pd.DataFrame({'year' : [2018,2019,2020,2021,'only_2021'],
                        'count' : [until_2018,until_2019,until_2020,until_2021,only_2021]})

count_df.plot(kind='bar')
plt.xticks([0,1,2,3,4],['2018','2019','2020','2021','only_2021'])
plt.xticks(fontsize=10,rotation=45)