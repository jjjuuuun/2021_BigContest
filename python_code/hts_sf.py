import urllib.request as req
import requests as reqs
from bs4 import BeautifulSoup
import urllib.parse as par
import os
import openpyxl
from openpyxl.drawing.image import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
import pyperclip
from selenium.webdriver import ActionChains
import time
from selenium.webdriver.common.keys import Keys

  
# =============================================================================
# user_path = 'C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/'
# =============================================================================
user_path = input()
os.chdir(user_path)

hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

bat_except_all = pd.read_csv('bat_except_all.csv')
bat_2018 = bat_except_all[bat_except_all['GYEAR'] == 2018]
bat_2019 = bat_except_all[bat_except_all['GYEAR'] == 2019]
bat_2020 = bat_except_all[bat_except_all['GYEAR'] == 2020]
bat_2021 = bat_except_all[bat_except_all['GYEAR'] == 2021]

list_2018 = list(bat_2018['PCODE'].values)
list_2019 = list(bat_2019['PCODE'].values)
list_2020 = list(bat_2020['PCODE'].values)
list_2021 = list(bat_2021['PCODE'].values)

hts_2018 = hts_2018[hts_2018['PCODE'].isin(list_2018)]
hts_2019 = hts_2019[hts_2019['PCODE'].isin(list_2019)]
hts_2020 = hts_2020[hts_2020['PCODE'].isin(list_2020)]
hts_2021 = hts_2021[hts_2021['PCODE'].isin(list_2021)]

hts_2018 = hts_2018[hts_2018['HIT_RESULT'] == 17]
hts_2019 = hts_2019[hts_2019['HIT_RESULT'] == 17]
hts_2020 = hts_2020[hts_2020['HIT_RESULT'] == 17]
hts_2021 = hts_2021[hts_2021['HIT_RESULT'] == 17]

hts_2018.dtypes


def game_id(x):
    return x[0:8]
        
hts_2018['G_ID'] = hts_2018['G_ID'].apply(game_id)
hts_2019['G_ID'] = hts_2019['G_ID'].apply(game_id)
hts_2020['G_ID'] = hts_2020['G_ID'].apply(game_id)
hts_2021['G_ID'] = hts_2021['G_ID'].apply(game_id)

hts_2018.drop(['PIT_ID', 'T_ID', 'INN', 'HIT_VEL', 'HIT_ANG_VER', 'PIT_VEL', 'STADIUM'], inplace = True, axis = 1)
hts_2019.drop(['PIT_ID', 'T_ID', 'INN', 'HIT_VEL', 'HIT_ANG_VER', 'PIT_VEL', 'STADIUM'], inplace = True, axis = 1)
hts_2020.drop(['PIT_ID', 'T_ID', 'INN', 'HIT_VEL', 'HIT_ANG_VER', 'PIT_VEL', 'STADIUM'], inplace = True, axis = 1)
hts_2021.drop(['PIT_ID', 'T_ID', 'INN', 'HIT_VEL', 'HIT_ANG_VER', 'PIT_VEL', 'STADIUM'], inplace = True, axis = 1)

hts_2018.to_csv('hts_sf_2018.csv', index = False)
hts_2019.to_csv('hts_sf_2019.csv', index = False)
hts_2020.to_csv('hts_sf_2020.csv', index = False)
hts_2021.to_csv('hts_sf_2021.csv', index = False)