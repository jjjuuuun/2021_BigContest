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

batter_df = pd.read_csv('bat_except_all.csv')
pcode_list = list(batter_df['PCODE'].values)


player_2018 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2019 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2020 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2021 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])


options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

for i in range(len(batter_df)):
    year = batter_df['GYEAR'].loc[i]
    pcode = batter_df['PCODE'].loc[i]

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
        
    print(i)
    
player_2018.to_csv('player_2018.csv', index = False)
player_2019.to_csv('player_2019.csv', index = False)
player_2020.to_csv('player_2020.csv', index = False)
player_2021.to_csv('player_2021.csv', index = False)