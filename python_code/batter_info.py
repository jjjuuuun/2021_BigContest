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

hts.drop(['G_ID', 'PIT_ID', 'T_ID', 'INN', 'HIT_RESULT', 'STADIUM', 'PIT_VEL'], axis=1, inplace=True)

hts.to_csv('hts_1.csv', index = False)

sort_hts = hts.sort_values(by=['PCODE', 'GYEAR'], ascending=[True, True], ignore_index = True)

sort_hts.dtypes

pcode_50054 = sort_hts[(sort_hts['PCODE'] == 71564)]

plt.plot('HIT_VEL', 'HIT_ANG_VER',
         data = pcode_50054,
         linestyle='none',
         marker='o')
plt.show()