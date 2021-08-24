import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)

hts.drop(['GYEAR', 'G_ID', 'PIT_ID', 'PCODE', 'T_ID', 'INN', 'STADIUM','HIT_RESULT', 'PIT_VEL'], axis=1, inplace=True)

hts_shuffled=hts.sample(frac=1).reset_index(drop=True)

len_hts_shuffled = len(hts_shuffled)//4

hts_1 = hts_shuffled.iloc[:len_hts_shuffled]
hts_2 = hts_shuffled.iloc[len_hts_shuffled : len_hts_shuffled*2]
hts_3 = hts_shuffled.iloc[len_hts_shuffled*2 : len_hts_shuffled*3]
hts_4 = hts_shuffled.iloc[len_hts_shuffled*3:]

hts_1.reset_index(drop=True, inplace=True)
hts_2.reset_index(drop=True, inplace=True)
hts_3.reset_index(drop=True, inplace=True)
hts_4.reset_index(drop=True, inplace=True)

hts_1.to_csv('hts_1.csv', index = False)
hts_2.to_csv('hts_2.csv', index = False)
hts_3.to_csv('hts_3.csv', index = False)
hts_4.to_csv('hts_4.csv', index = False)