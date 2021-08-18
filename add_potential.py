# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
os.chdir("C:/Users/sj545/OneDrive/바탕 화면/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_데이터_210803/01_제공데이터")


batter_2018 = pd.read_csv("C:/Users/sj545/OneDrive/바탕 화면/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_데이터_210803/01_제공데이터/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2018.csv",encoding="cp949")
batter_2019 = pd.read_csv("C:/Users/sj545/OneDrive/바탕 화면/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_데이터_210803/01_제공데이터/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2019.csv",encoding="cp949")
batter_2020 = pd.read_csv("C:/Users/sj545/OneDrive/바탕 화면/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_데이터_210803/01_제공데이터/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2020.csv",encoding="cp949")
batter_2021 = pd.read_csv("C:/Users/sj545/OneDrive/바탕 화면/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_데이터_210803/01_제공데이터/2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2021.csv",encoding="cp949")


batter = pd.concat([batter_2018,batter_2019,batter_2020,batter_2021],ignore_index = True)

batter_sort = batter.sort_values(by=['PCODE','GYEAR'],ascending=[True,True])


#batter_sort.to_csv('batter_sort.csv',index=False)


potential_2019 = pd.read_csv("potential_2019.csv")
potential_2020 = pd.read_csv("potential_2020.csv")
potential_2021 = pd.read_csv("potential_2021.csv")
potential_2019['year']=2019
potential_2020['year']=2020
potential_2021['year']=2021
potential = pd.concat([potential_2019,potential_2020,potential_2021],ignore_index=True)
potential_sort = potential.sort_values(by=['PCODE','year'],ascending=[True,True])
#potential_sort.to_csv('potential_sort.csv',index=False)




def make_new_df(df):
    batter_columns = list(batter_sort.columns)
    new_df = pd.DataFrame(index=range(df.shape[0]),columns = batter_columns)
    for idx in range(df.shape[0]):
        new_df.iloc[idx] = df.iloc[idx]
    return new_df


new_batter = make_new_df(batter_sort)

def add_potential(p,df):
    df['potential']=0
    for idx in range(df.shape[0]):
        for num in range(p.shape[0]):
            pot_pcode = p.PCODE[num]
            bat_pcode = df.PCODE[idx]
            pot_year = p.year[num]
            bat_year = df.GYEAR[idx]
            pot_potential = p.POTENTIAL[num]
            if pot_pcode == bat_pcode and pot_year == bat_year:
                df.loc[idx,['potential']] = pot_potential
                break
    
    return df
batter_final = add_potential(potential_sort,new_batter)
batter_final.to_csv('add_potential.csv',index=False)
