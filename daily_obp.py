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


os.chdir("C:/Users/sj545/OneDrive/바탕 화면/2021_BigContest")


daily_batter_2018 = pd.read_csv("daily_batter_2018.csv")
daily_batter_2019 = pd.read_csv("daily_batter_2019.csv")
daily_batter_2020 = pd.read_csv("daily_batter_2020.csv")
daily_batter_2021 = pd.read_csv("daily_batter_2021.csv")

    
daily_batter_2018.columns
daily_batter_2018.drop(['AVG1', 'SO', 'R', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', '2B', '3B', 'HR', 'RBI', 'SB', 'CS','AVG2' , 'GDP'], axis=1, inplace = True)
daily_batter_2019.drop(['AVG1', 'SO', 'R', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', '2B', '3B', 'HR', 'RBI', 'SB', 'CS','AVG2' , 'GDP'], axis=1, inplace = True)
daily_batter_2020.drop(['AVG1', 'SO', 'R', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', '2B', '3B', 'HR', 'RBI', 'SB', 'CS','AVG2' , 'GDP'], axis=1, inplace = True)
daily_batter_2021.drop(['AVG1', 'SO', 'R', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', '2B', '3B', 'HR', 'RBI', 'SB', 'CS','AVG2' , 'GDP'], axis=1, inplace = True)

daily_batter_2018['SF'] = 0
daily_batter_2019['SF'] = 0
daily_batter_2020['SF'] = 0
daily_batter_2021['SF'] = 0


#daily_batter_2018['OBP'] = (daily_batter_2018['H'] + daily_batter_2018['BB'] + daily_batter_2018['HBP']) / (daily_batter_2018['AB'] + daily_batter_2018['BB'] + daily_batter_2018['HBP'] + daily_batter_2018['SO']) 
#daily_batter_2018['OBP_cumsum'] = (daily_batter_2018['H'].cumsum() + daily_batter_2018['BB'].cumsum() + daily_batter_2018['HBP'].cumsum()) / (daily_batter_2018['AB'].cumsum() + daily_batter_2018['BB'].cumsum() + daily_batter_2018['HBP'].cumsum() + daily_batter_2018['SO'].cumsum()) 

hts_sf_2018 = pd.read_csv("hts_sf_2018.csv")
hts_sf_2019 = pd.read_csv("hts_sf_2019.csv")
hts_sf_2020 = pd.read_csv("hts_sf_2020.csv")
hts_sf_2021 = pd.read_csv("hts_sf_2021.csv")


hts_sf_2018.sort_values(by=['PCODE','G_ID'])
hts_sf_2018['SF'] = 1
hts_sf_2018['SF'] = hts_sf_2018.groupby(['G_ID','PCODE'])['SF'].cumsum()
hts_sf_count = hts_sf_2018.loc[hts_sf_2018.groupby(["PCODE", "G_ID"])["SF"].idxmax()]
hts_sf_count.reset_index(drop=True, inplace=True)
daily_sf_count = pd.merge(daily_batter_2018,hts_sf_count,on=['G_ID','PCODE'],how='left')
daily_batter_2018.dtypes
hts_sf_count.dtypes
daily_sf_count.columns
daily_sf_count.drop(['SF_x','GYEAR_y','HIT_RESULT'],axis=1, inplace=True)

daily_sf_count['SF_y'].fillna(0,inplace=True)

def make_sf(hts_sort, daily):
    hts_sort.sort_values(by=['PCODE','G_ID'])
    hts_sort['SF'] = 1
    hts_sort['SF'] = hts_sort.groupby(['G_ID','PCODE'])['SF'].cumsum()
    hts_sf_count = hts_sort.loc[hts_sort.groupby(["PCODE", "G_ID"])["SF"].idxmax()]
    hts_sf_count.reset_index(drop=True, inplace=True)
    daily_sf_count = pd.merge(daily,hts_sf_count,on=['G_ID','PCODE'],how='left')
    daily_sf_count.drop(['SF_x','GYEAR_y','HIT_RESULT'],axis=1, inplace=True)
    
    daily_sf_count['SF_y'].fillna(0,inplace=True)
    daily_sf_count.rename(columns = {'GYEAR_x':'GYEAR','SF_y':'SF'},inplace=True)
    daily_sf_count['SF'].fillna(0,inplace=True)
    daily_sf_count = daily_sf_count[['PA','AB','H','BB','HBP','SF','PCODE','GYEAR','G_ID']]
    return daily_sf_count

daily_sf_2018 = make_sf(hts_sf_2018,daily_batter_2018)
daily_sf_2019 = make_sf(hts_sf_2019,daily_batter_2019)
daily_sf_2020 = make_sf(hts_sf_2020,daily_batter_2020)
daily_sf_2021 = make_sf(hts_sf_2021,daily_batter_2021)

def make_obp(daily):
    daily['OBP'] = (daily['H'] + daily['BB'] + daily['HBP']) / (daily['AB'] + daily['BB'] + daily['HBP'] + daily['SF'])
    daily['OBP_cumsum'] = (daily.groupby(['PCODE'])['H'].cumsum() + daily.groupby(['PCODE'])['BB'].cumsum() + daily.groupby(['PCODE'])['HBP'].cumsum()) / (daily.groupby(['PCODE'])['AB'].cumsum() + daily.groupby(['PCODE'])['BB'].cumsum() + daily.groupby(['PCODE'])['HBP'].cumsum() + daily.groupby(['PCODE'])['SF'].cumsum())
    daily['OBP'].fillna(0,inplace=True)
    
    return daily
daily_obp_2018 = make_obp(daily_sf_2018)
daily_obp_2019 = make_obp(daily_sf_2019)
daily_obp_2020 = make_obp(daily_sf_2020)
daily_obp_2021 = make_obp(daily_sf_2021)    
    
    
daily_obp_2018.to_csv("daily_obp_2018.csv",index=False)
daily_obp_2019.to_csv("daily_obp_2019.csv",index=False)
daily_obp_2020.to_csv("daily_obp_2020.csv",index=False)
daily_obp_2021.to_csv("daily_obp_2021.csv",index=False)
    
    

    