import pandas as pd
import os

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
print(os.listdir())
print(os.getcwd())
player_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2018.csv', encoding='CP949')
player_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2019.csv', encoding='CP949')
player_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2020.csv', encoding='CP949')
player_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2021.csv', encoding='CP949')

batter_2018 = player_2018[~(player_2018['POSITION'] == '투')]
batter_2019 = player_2019[~(player_2019['POSITION'] == '투')]
batter_2020 = player_2020[~(player_2020['POSITION'] == '투')]
batter_2021 = player_2021[~(player_2021['POSITION'] == '투')]

print(batter_2018.shape, batter_2019.shape, batter_2020.shape, batter_2021.shape)

batter_2018.reset_index(drop=True, inplace = True)
batter_2019.reset_index(drop=True, inplace = True)
batter_2020.reset_index(drop=True, inplace = True)
batter_2021.reset_index(drop=True, inplace = True)

def int_money(df):
    for idx in range(df.shape[0]):
        try:
            if '만원' in df.MONEY[idx]:
                df.MONEY[idx] = df.MONEY[idx].replace('만원', '')
            elif '달러' in df.MONEY[idx]:
                df.MONEY[idx] = df.MONEY[idx].replace('달러', '')
        except:
            df.MONEY[idx] = '0'

batter_list = [batter_2018, batter_2019, batter_2020, batter_2021]
for df in batter_list:
    int_money(df)

batter_2018 = batter_2018.astype({'MONEY':'int'})
batter_2019 = batter_2019.astype({'MONEY':'int'})
batter_2020 = batter_2020.astype({'MONEY':'int'})
batter_2021 = batter_2021.astype({'MONEY':'int'})

def potential(before, after):
    new_df = pd.DataFrame(index=range(after.shape[0]), columns = ['PCODE','POTENTIAL'])
    for idx in range(after.shape[0]):
        after_pcode = after.PCODE[idx]
        before_pcode = before[(before['PCODE'] == after_pcode)]
        if before_pcode.empty == False:
            after_money = after.MONEY[idx]
            before_money = before_pcode.MONEY.iloc[0]
            potential = round((after_money - before_money) / before_money, 3)
        else:   
            potential = 0.0
        new_df.PCODE[idx] = after_pcode
        new_df.POTENTIAL[idx] = potential
    return new_df

potential_2019 = potential(batter_2018, batter_2019)
potential_2020 = potential(batter_2019, batter_2020)
potential_2021 = potential(batter_2020, batter_2021)

potential_2019.to_csv('potential_2019.csv', index = False)
potential_2020.to_csv('potential_2020.csv', index = False)
potential_2021.to_csv('potential_2021.csv', index = False)