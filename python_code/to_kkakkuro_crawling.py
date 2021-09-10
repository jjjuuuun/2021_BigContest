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

pcode_list = [76232, 68050, 75847, 67341, 79192, 78224, 78513, 76290, 79215, 67872]


player_2018 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2019 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2020 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2021 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])


options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

cnt = 1
for year in [2018, 2019, 2020, 2021]:
    for pcode in pcode_list:    
        browser = webdriver.Chrome('C:/Users/KIMJUNYOUNG/Desktop/chromedriver.exe', chrome_options=options)
        browser.implicitly_wait(1)
        browser.get('https://www.koreabaseball.com/Record/Player/HitterDetail/Daily.aspx?playerId={}'.format(pcode))
    
        browser.find_element_by_class_name('select02').click()
        xpath_format = '//option[@value="{year}"]'
        browser.find_element_by_xpath(xpath_format.format(year = year)).click()
        
        game_info = browser.find_elements_by_css_selector('div.tbl-type02.tbl-type02-pd0 > table > tbody > tr')
    
        # 빈 데이터 프레임 넣기
        player_df = pd.DataFrame(index=range(len(game_info)),
                                 columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2'])
        
        idx = 0
        for info in game_info:
            row = info.text.split(' ')
            player_df.iloc[idx] = row
            idx += 1
        
        player_df['PCODE'] = pcode
        
        if year == 2018:
            player_2018 = pd.concat([player_2018, player_df], ignore_index=True)
        elif year == 2019:
            player_2019 = pd.concat([player_2019, player_df], ignore_index=True)
        elif year == 2020:
            player_2020 = pd.concat([player_2020, player_df], ignore_index=True)
        elif year == 2021:
            player_2021 = pd.concat([player_2021, player_df], ignore_index=True)
            
        print(cnt)
        cnt += 1
    
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

player_2018.to_csv('to_kkakuro_2018.csv', index = False)
player_2019.to_csv('to_kkakuro_2019.csv', index = False)
player_2020.to_csv('to_kkakuro_2020.csv', index = False)
player_2021.to_csv('to_kkakuro_2021.csv', index = False)