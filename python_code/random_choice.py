# =============================================================================
# PCODE별 random 추출
# train:validation:test = 6:2:2
# =============================================================================

import pandas as pd
import os
import random

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
os.listdir()
df = pd.read_csv('add_potential.csv')

pcode_df = df.groupby('PCODE')

pcode_list = []
for pcode in pcode_df.groups.keys():
    pcode_list.append(pcode)
    
random.shuffle(pcode_list)
len_pcode_list = len(pcode_list)//10
train_pcode = pcode_list[:len_pcode_list*6]
val_pcode = pcode_list[len_pcode_list*6:len_pcode_list*8]
test_pcode = pcode_list[len_pcode_list*8:]
print(len(train_pcode), len(val_pcode), len(test_pcode))

train_data = pd.DataFrame(columns = df.columns)
val_data = pd.DataFrame(columns = df.columns)
test_data = pd.DataFrame(columns = df.columns)

for idx in range(len(df)):
    if df.PCODE[idx] in train_pcode:
        train_data = train_data.append(df.iloc[idx], ignore_index=True)
    elif df.PCODE[idx] in val_pcode:
        val_data = val_data.append(df.iloc[idx], ignore_index=True)
    else:
        test_data = test_data.append(df.iloc[idx], ignore_index=True)

