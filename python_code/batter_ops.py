import pandas as pd
import os
import numpy as np

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
bat_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2018.csv')
bat_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2019.csv')
bat_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2020.csv')
bat_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2021.csv')

bat = pd.concat([bat_2018, bat_2019, bat_2020, bat_2021], ignore_index=True)


def obp(df):
    obp = (df.HIT + df.BB + df.HP) / (df.AB + df.BB + df.HP + df.SF)
    return obp

bat['OBP'] = bat.apply(obp, axis = 1)
bat['OPS'] = bat.SLG + bat.OBP

bat = bat[(bat['PA'] >= 100)]

bat.to_csv('bat.csv', index=False)
