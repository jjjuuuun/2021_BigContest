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

player_2018 = pd.read_csv('player_2018.csv')
player_2019 = pd.read_csv('player_2019.csv')
player_2020 = pd.read_csv('player_2020.csv')
player_2021 = pd.read_csv('player_2021.csv')

# LG, KT, NC, SK, KIA, 한화, 롯데, 넥센, 삼성, 두산
player_2018['OPPOSITE'].value_counts()

# LG, KT, NC, SK, KIA, 한화, 롯데, 키움, 삼성, 두산
player_2019['OPPOSITE'].value_counts()

# LG, KT, NC, SK, KIA, 한화, 롯데, 키움, 삼성, 두산
player_2020['OPPOSITE'].value_counts()

# LG, KT, NC, SSG, KIA, 한화, 롯데, 키움, 삼성, 두산
player_2021['OPPOSITE'].value_counts()

# =============================================================================
# LG : LG
# KT : KT
# NC : NC
# SK, SSG : SK
# KIA : HT
# 한화 : HH
# 롯데 : LT
# 키움,넥센 : WO
# 삼성 : SS
# 두산 : OB
# =============================================================================

team_id = {'LG' : 'LG', 'KT' : 'KT', 'NC' : 'NC', 'SK' : 'SK', 'SSG' : 'SK', 'KIA' : 'HT', 
           '한화' : 'HH', '롯데' : 'LT', '키움' : 'WO', '넥센' : 'WO', '삼성' : 'SS', '두산' : 'OB'}

def unity_team(x):
    return team_id[x]

player_2018['OPPOSITE'] = player_2018['OPPOSITE'].apply(unity_team)
player_2019['OPPOSITE'] = player_2019['OPPOSITE'].apply(unity_team)
player_2020['OPPOSITE'] = player_2020['OPPOSITE'].apply(unity_team)
player_2021['OPPOSITE'] = player_2021['OPPOSITE'].apply(unity_team)

player_2018['GYEAR'] = 2018
player_2019['GYEAR'] = 2019
player_2020['GYEAR'] = 2020
player_2021['GYEAR'] = 2021

player_2018['DATE'] = player_2018['DATE'].apply(str)
player_2019['DATE'] = player_2019['DATE'].apply(str)
player_2020['DATE'] = player_2020['DATE'].apply(str)
player_2021['DATE'] = player_2021['DATE'].apply(str)

player_2018['GYEAR'] = player_2018['GYEAR'].apply(str)
player_2019['GYEAR'] = player_2019['GYEAR'].apply(str)
player_2020['GYEAR'] = player_2020['GYEAR'].apply(str)
player_2021['GYEAR'] = player_2021['GYEAR'].apply(str)

player_2018['DATE'] = player_2018['DATE'] + '.' + player_2018['GYEAR']
player_2019['DATE'] = player_2019['DATE'] + '.' + player_2019['GYEAR']
player_2020['DATE'] = player_2020['DATE'] + '.' + player_2020['GYEAR']
player_2021['DATE'] = player_2021['DATE'] + '.' + player_2021['GYEAR']


def game_id(x):
    date_list = x.split('.')
    if int(date_list[0]) < 10:
        date_list[0] = '0' + str(int(date_list[0]))
    if len(date_list[1]) == 1 and int(date_list[1]) < 10:
        date_list[1] = str(int(date_list[1])) + '0'
    g_id = date_list[2] + date_list[0] + date_list[1]
    return g_id
        
player_2018['G_ID'] = player_2018['DATE'].apply(game_id)
player_2019['G_ID'] = player_2019['DATE'].apply(game_id)
player_2020['G_ID'] = player_2020['DATE'].apply(game_id)
player_2021['G_ID'] = player_2021['DATE'].apply(game_id)

player_2018.drop(['OPPOSITE', 'DATE'], inplace = True, axis = 1)
player_2019.drop(['OPPOSITE', 'DATE'], inplace = True, axis = 1)
player_2020.drop(['OPPOSITE', 'DATE'], inplace = True, axis = 1)
player_2021.drop(['OPPOSITE', 'DATE'], inplace = True, axis = 1)

player_2018.to_csv('daily_batter_2018.csv', index = False)
player_2019.to_csv('daily_batter_2019.csv', index = False)
player_2020.to_csv('daily_batter_2020.csv', index = False)
player_2021.to_csv('daily_batter_2021.csv', index = False)