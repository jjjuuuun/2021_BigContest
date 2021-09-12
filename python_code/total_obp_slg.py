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

# =============================================================================
# barrel_per을 구해 상관계수 확인
# barrel_per_hit : 타구당 배럴 비율
# barrel_per_pa : 타석당 배럴 비율
# barrel_per_ab : 타수당 배럴 비율
# =============================================================================

batter_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2018.csv')
batter_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2019.csv')
batter_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2020.csv')
batter_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2021.csv')

# 정규타석 못채운 pcode 제거
batter_2018 = batter_2018[(batter_2018['PA'] >= 446)]
batter_2019 = batter_2019[(batter_2019['PA'] >= 446)]
batter_2020 = batter_2020[(batter_2020['PA'] >= 446)]
batter_2021 = batter_2021[(batter_2021['PA'] >= 190)]

# define_barrel에서 저장한 sort_hts 사용
sort_hts = pd.read_csv('sort_hts.csv')
sort_hts = sort_hts.sort_values(by = 'GYEAR', ascending = True, ignore_index = True)
sort_hts_18 = sort_hts.iloc[:35029]
sort_hts_19 = sort_hts.iloc[35029:68279]
sort_hts_20 = sort_hts.iloc[68279:102781]
sort_hts_21 = sort_hts.iloc[102781:]

# barrel_per_hit 구하기
def append_barrel(df):
    barrel_per_df = pd.DataFrame(columns = ['PCODE', 'len_barrel', 'barrel_per_hit'])
    pcode_list = set(list(df.PCODE))
    idx = 0    
    for pcode in pcode_list:
        pcode_df = df[(df['PCODE'] == pcode)]
        len_barrel = len(pcode_df[(pcode_df['barrel'] == 1)])
        barrel_per_hit = len_barrel / len(pcode_df) 
        barrel_per_df.loc[idx] = pcode, len_barrel, barrel_per_hit
        idx += 1
    return barrel_per_df

barrel_18 = append_barrel(sort_hts_18)
barrel_19 = append_barrel(sort_hts_19)
barrel_20 = append_barrel(sort_hts_20)
barrel_21 = append_barrel(sort_hts_21)

# barrel_per과 hts 데이터프레임 합치기
batter_2018 = pd.merge(batter_2018, barrel_18, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2019 = pd.merge(batter_2019, barrel_19, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2020 = pd.merge(batter_2020, barrel_20, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2021 = pd.merge(batter_2021, barrel_21, left_on = 'PCODE', right_on='PCODE', how = 'inner')

bat = pd.concat([batter_2018, batter_2019, batter_2020, batter_2021], ignore_index=True)

# barrel_per_pa, barrel_per_ab 구하기
bat['barrel_per_pa'] = bat['len_barrel'] / bat['PA']
bat['barrel_per_ab'] = bat['len_barrel'] / bat['AB']

# OPS와의 상관계수를 알기 위해 OBP, OPS 계산
def obp(df):
    obp = (df.HIT + df.BB + df.HP) / (df.AB + df.BB + df.HP + df.SF)
    return obp

bat['OBP'] = bat.apply(obp, axis = 1)
bat['OPS'] = bat.SLG + bat.OBP

# OPS와의 상관계수 확인
bat_corr = bat.corr()
bat_corr_ops = bat.corr().OPS
bat_corr_ops.drop(['OPS'], inplace = True)

# OPS와 가장 상관관계가 있는 columns 10개 확인
top_corr_ops = bat_corr_ops.sort_values(ascending = False).head(10)

# OPS와의 상관계수 시각화
plt.bar(x = top_corr_ops.index, height = top_corr_ops.values)
plt.tight_layout()
plt.show()

# 불필요한 columns 제거
bat.drop(['len_barrel', 'OBP', 'OPS'], axis = 1, inplace = True)

bat.to_csv('bat_all.csv', index = False)

# =============================================================================
# KBO 사이트로부터 일자별 선수별 데이터 Crawling
# =============================================================================

# 정규타석을 채운 pcode들을 알기 위해 bat_all을 불러옴
bat = pd.read_csv('bat_all.csv')

# Crawling한 데이터들을 넣은 프레임 만들기
player_2018 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2019 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2020 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])
player_2021 = pd.DataFrame(columns=['DATE', 'OPPOSITE', 'AVG1', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'AVG2', 'PCODE'])


# web crawling option(창 안띄우기)
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

for i in range(len(bat)):
    year = bat['GYEAR'].loc[i]
    pcode = bat['PCODE'].loc[i]

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
    
# Crawling 결과 csv 파일로 저장
player_2018.to_csv('player_2018.csv', index = False)
player_2019.to_csv('player_2019.csv', index = False)
player_2020.to_csv('player_2020.csv', index = False)
player_2021.to_csv('player_2021.csv', index = False)

# =============================================================================
# Crawling 한 데이터프레임 전처리
# 데이터 타입 변경
# 수치화
# 필요한 데이터로 전환
# =============================================================================
player_2018 = pd.read_csv('player_2018.csv')
player_2019 = pd.read_csv('player_2019.csv')
player_2020 = pd.read_csv('player_2020.csv')
player_2021 = pd.read_csv('player_2021.csv')

# LG, KT, NC, SK, KIA, 한화, 롯데, 넥센, 삼성, 두산
print(player_2018['OPPOSITE'].value_counts())

# LG, KT, NC, SK, KIA, 한화, 롯데, 키움, 삼성, 두산
print(player_2019['OPPOSITE'].value_counts())

# LG, KT, NC, SK, KIA, 한화, 롯데, 키움, 삼성, 두산
print(player_2020['OPPOSITE'].value_counts())

# LG, KT, NC, SSG, KIA, 한화, 롯데, 키움, 삼성, 두산
print(player_2021['OPPOSITE'].value_counts())

# DATE를 G_ID 형태로 변환
def game_id(x):
    date_list = x.split('.')
    if int(date_list[0]) < 10:
        date_list[0] = '0' + str(int(date_list[0]))
    if len(date_list[1]) == 1 and int(date_list[1]) < 10:
        date_list[1] = str(int(date_list[1])) + '0'
    g_id = date_list[2] + date_list[0] + date_list[1]
    return g_id

# 데이터프레임을 원하는 형태로 변경
def renewal_df(df, year):
    df['GYEAR'] = year
    df['DATE'] = df['DATE'].apply(str)
    df['GYEAR'] = df['GYEAR'].apply(str)
    df['DATE'] = df['DATE'] + '.' + str(year)
    df['G_ID'] = df['DATE'].apply(game_id)
    df['G_ID'] = df['G_ID'].apply(int)
    df.drop(['OPPOSITE', 'DATE'], inplace = True, axis = 1)    
    return df

daily_batter_2018 = renewal_df(player_2018, 2018)
daily_batter_2019 = renewal_df(player_2019, 2019)
daily_batter_2020 = renewal_df(player_2020, 2020)
daily_batter_2021 = renewal_df(player_2021, 2021)

daily_batter_2018.to_csv('daily_batter_2018.csv', index = False)
daily_batter_2019.to_csv('daily_batter_2019.csv', index = False)
daily_batter_2020.to_csv('daily_batter_2020.csv', index = False)
daily_batter_2021.to_csv('daily_batter_2021.csv', index = False)

# =============================================================================
# 해당 타구가 barrel 인지 아닌지 확인
# =============================================================================
# define_barrel에서 찾은 계수들 사용
# 나중에 class로 만들어 불러와야 함
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

# 각 년도에 정규타석을 채운 pcode를 얻기 위함    
bat = pd.read_csv('bat_all.csv')

bat_2018 = bat[(bat['GYEAR'] == 2018)]
bat_2019 = bat[(bat['GYEAR'] == 2019)]
bat_2020 = bat[(bat['GYEAR'] == 2020)]
bat_2021 = bat[(bat['GYEAR'] == 2021)]

list_2018 = list(bat_2018['PCODE'].values)
list_2019 = list(bat_2019['PCODE'].values)
list_2020 = list(bat_2020['PCODE'].values)
list_2021 = list(bat_2021['PCODE'].values)

hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

# hts file에 해당 타구가 barrel 인지 아닌지 확인할 수 있는 column을 만듬
def hts_barrel(df, pcode_list):
    df = df[df['PCODE'].isin(pcode_list)]
    df['barrel'] = df.apply(check_barrel, axis = 1)
    df.drop(['GYEAR', 'PIT_ID', 'T_ID', 'INN', 'HIT_VEL', 'HIT_ANG_VER', 'PIT_VEL', 'STADIUM'], axis=1, inplace=True)
    return df
    
hts_barrel_2018 = hts_barrel(hts_2018, list_2018)
hts_barrel_2019 = hts_barrel(hts_2019, list_2019)
hts_barrel_2020 = hts_barrel(hts_2020, list_2020)
hts_barrel_2021 = hts_barrel(hts_2021, list_2021)

hts_barrel_2018.to_csv('hts_barrel_2018.csv', index = False)
hts_barrel_2019.to_csv('hts_barrel_2019.csv', index = False)
hts_barrel_2020.to_csv('hts_barrel_2020.csv', index = False)
hts_barrel_2021.to_csv('hts_barrel_2021.csv', index = False)



# =============================================================================
# slg를 구하기 위함
# =============================================================================
hts_barrel_2018 = pd.read_csv("hts_barrel_2018.csv")
hts_barrel_2019 = pd.read_csv("hts_barrel_2019.csv")
hts_barrel_2020 = pd.read_csv("hts_barrel_2020.csv")
hts_barrel_2021 = pd.read_csv("hts_barrel_2021.csv")

daily_batter_2018 = pd.read_csv("daily_batter_2018.csv")
daily_batter_2019 = pd.read_csv("daily_batter_2019.csv")    
daily_batter_2020 = pd.read_csv("daily_batter_2020.csv")
daily_batter_2021 = pd.read_csv("daily_batter_2021.csv")

def cut_G_ID(x):
    return x[0:8]

def update_df(df):
    df['G_ID'] = df['G_ID'].apply(cut_G_ID)
    df['G_ID'] = df['G_ID'].apply(int)
    df = df.sort_values(by=['PCODE','G_ID'],ascending=[True,True])
    df.reset_index(drop = True, inplace=True)
    return df

hts_barrel_2018 = update_df(hts_barrel_2018)
hts_barrel_2019 = update_df(hts_barrel_2019)
hts_barrel_2020 = update_df(hts_barrel_2020)
hts_barrel_2021 = update_df(hts_barrel_2021)

# slg 계산
def make_slg(hts, daily):
    #barrel 누적값 계산
    hts['barrel_cnt'] = hts.groupby(['G_ID','PCODE'])['barrel'].cumsum()
    
    # 누적값 중 최댓값을 가져와서 배럴 개수 계산
    hts_barrel_cnt = hts.loc[hts.groupby(["PCODE", "G_ID"])["barrel_cnt"].idxmax()]
    
    # left join
    daily_barrel_cnt = pd.merge(daily, hts_barrel_cnt, on=['G_ID','PCODE'], how='left')
    
    # barrel 생산률 계산
    daily_barrel_cnt['barrel_per'] = daily_barrel_cnt['barrel_cnt'] / daily_barrel_cnt['PA']
    
    # 결측치 제거
    # daily_barrel_cnt['PA'] 가 0인 행이 null 값이 나옴
    # 그러한 행을 제거
    daily_barrel_cnt.dropna(axis=0, inplace=True)
    
    # 안타수 
    daily_barrel_cnt['1B'] = daily_barrel_cnt['H'] - daily_barrel_cnt['2B'] - daily_barrel_cnt['3B'] - daily_barrel_cnt['HR']
    
    # 장타율 계산
    daily_barrel_cnt['SLG'] = (daily_barrel_cnt['1B'] + daily_barrel_cnt['2B'] * 2 + daily_barrel_cnt['3B'] * 3 + daily_barrel_cnt['HR'] * 4) / daily_barrel_cnt['AB']
    
    # 결측치 0 처리
    daily_barrel_cnt['SLG'] = daily_barrel_cnt['SLG'].fillna(0)
    daily_barrel_cnt.drop(['SB','CS', 'BB','HBP', 'SO', 'GDP', 'AVG2', 'barrel_cnt'], axis=1, inplace = True)
    
    return daily_barrel_cnt

daily_slg_2018 = make_slg(hts_barrel_2018, daily_batter_2018)
daily_slg_2019 = make_slg(hts_barrel_2019, daily_batter_2019)
daily_slg_2020 = make_slg(hts_barrel_2020, daily_batter_2020)
daily_slg_2021 = make_slg(hts_barrel_2021, daily_batter_2021)

daily_slg_2018.to_csv("daily_slg_2018.csv",index=False)
daily_slg_2019.to_csv("daily_slg_2019.csv",index=False)
daily_slg_2020.to_csv("daily_slg_2020.csv",index=False)
daily_slg_2021.to_csv("daily_slg_2021.csv",index=False)

# =============================================================================
# obp를 구하기 위함
# =============================================================================

daily_batter_2018 = pd.read_csv("daily_batter_2018.csv")
daily_batter_2019 = pd.read_csv("daily_batter_2019.csv")
daily_batter_2020 = pd.read_csv("daily_batter_2020.csv")
daily_batter_2021 = pd.read_csv("daily_batter_2021.csv")

def select_df(df):
    df = df[df['HIT_RESULT'] == 17]
    df.drop(['barrel', 'barrel_cnt'], inplace = True, axis = 1)
    return df

hts_sf_2018 = select_df(hts_barrel_2018)
hts_sf_2019 = select_df(hts_barrel_2019)
hts_sf_2020 = select_df(hts_barrel_2020)
hts_sf_2021 = select_df(hts_barrel_2021)

def make_obp(hts_sf, daily):
    daily['SF'] = 0
    daily.drop(['AVG1', 'SO', 'R', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', '2B', '3B', 'HR', 'RBI', 'SB', 'CS','AVG2' , 'GDP'], axis=1, inplace = True)
    hts_sf.sort_values(by=['PCODE','G_ID'], inplace = True)
    hts_sf['SF'] = 1
    hts_sf['SF'] = hts_sf.groupby(['G_ID','PCODE'])['SF'].cumsum()
    hts_sf_count = hts_sf.loc[hts_sf.groupby(["PCODE", "G_ID"])["SF"].idxmax()]
    hts_sf_count.reset_index(drop=True, inplace=True)
    daily_sf = pd.merge(daily, hts_sf_count, on=['G_ID','PCODE'], how='left')
    daily_sf.rename(columns = {'GYEAR_x':'GYEAR','SF_y':'SF'}, inplace=True)
    daily_sf['SF'].fillna(0,inplace=True)
    daily_sf = daily_sf[['PA','AB','H','BB','HBP','SF','PCODE','G_ID']]
    daily_sf['OBP'] = (daily_sf['H'] + daily_sf['BB'] + daily_sf['HBP']) / (daily_sf['AB'] + daily_sf['BB'] + daily_sf['HBP'] + daily_sf['SF'])
    daily_sf['OBP'].fillna(0,inplace=True)   
    return daily_sf


daily_obp_2018 = make_obp(hts_sf_2018, daily_batter_2018)
daily_obp_2019 = make_obp(hts_sf_2019, daily_batter_2019)
daily_obp_2020 = make_obp(hts_sf_2020, daily_batter_2020)
daily_obp_2021 = make_obp(hts_sf_2021, daily_batter_2021)

daily_obp_2018.to_csv('daily_obp_2018.csv', index = False)
daily_obp_2019.to_csv('daily_obp_2019.csv', index = False)
daily_obp_2020.to_csv('daily_obp_2020.csv', index = False)
daily_obp_2021.to_csv('daily_obp_2021.csv', index = False)

