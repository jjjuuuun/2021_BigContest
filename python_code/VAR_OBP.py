import pandas as pd
import numpy as np
import os
from datetime import datetime
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
import seaborn as sns
import math
#var modeling
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from IPython.display import display
import time

# =============================================================================
# user_path = 'C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/'
# =============================================================================
user_path = input()
os.chdir(user_path)

daily_obp_2018 = pd.read_csv('daily_obp_2018.csv')
daily_obp_2019 = pd.read_csv('daily_obp_2019.csv')
daily_obp_2020 = pd.read_csv('daily_obp_2020.csv')
daily_obp_2021 = pd.read_csv('daily_obp_2021.csv')

ten_player = [67341, 67872, 68050, 75847, 76232, 76290, 78224, 78513, 79192, 79215]

daily_obp_2018 = daily_obp_2018[daily_obp_2018['PCODE'].isin(ten_player)]
daily_obp_2019 = daily_obp_2019[daily_obp_2019['PCODE'].isin(ten_player)]
daily_obp_2020 = daily_obp_2020[daily_obp_2020['PCODE'].isin(ten_player)]
daily_obp_2021 = daily_obp_2021[daily_obp_2021['PCODE'].isin(ten_player)]

daily_obp_2018['OB'] = daily_obp_2018['AB'] + daily_obp_2018['BB'] + daily_obp_2018['HBP'] + daily_obp_2018['SF']
daily_obp_2019['OB'] = daily_obp_2019['AB'] + daily_obp_2019['BB'] + daily_obp_2019['HBP'] + daily_obp_2019['SF']
daily_obp_2020['OB'] = daily_obp_2020['AB'] + daily_obp_2020['BB'] + daily_obp_2020['HBP'] + daily_obp_2020['SF']
daily_obp_2021['OB'] = daily_obp_2021['AB'] + daily_obp_2021['BB'] + daily_obp_2021['HBP'] + daily_obp_2021['SF']
    

def make_obp_cumsum(df):
    data = pd.DataFrame()
    df = df[(df['OB'] > 0)]
    for pcode in ten_player:
        obp_df = df[df['PCODE'] == pcode]
        obp_df.sort_values(by = 'G_ID', ascending = True, inplace = True)
        obp_df.reset_index(drop = True, inplace=True)
        obp_df['OBP_CUMSUM'] = (obp_df['H'].cumsum() + obp_df['BB'].cumsum() + obp_df['HBP'].cumsum()) / (obp_df['AB'].cumsum() + obp_df['BB'].cumsum() + obp_df['HBP'].cumsum() + obp_df['SF'].cumsum())
        data = pd.concat([data,obp_df],axis=0)
    return data

daily_obp_2018 = make_obp_cumsum(daily_obp_2018)
daily_obp_2019 = make_obp_cumsum(daily_obp_2019)
daily_obp_2020 = make_obp_cumsum(daily_obp_2020)
daily_obp_2021 = make_obp_cumsum(daily_obp_2021)

data_for_var_obp = pd.concat([daily_obp_2018, daily_obp_2019, daily_obp_2020, daily_obp_2021], axis=0, ignore_index=True)
data_for_var_obp.reset_index(drop=True,inplace = True)
#============================================================
def eda_obp(player):
    df = data_for_var_obp[data_for_var_obp['PCODE'] == player]
    df.reset_index(drop = True, inplace = True)
    df['G_ID'] = df['G_ID'].astype('str')
    
    dt = []
    for k in df['G_ID']:
        dt_ = datetime.strptime(k, '%Y%m%d')
        dt.append(dt_)
    df['G_ID'] = pd.Series(dt)
    
    df = df.set_index('G_ID')
    df.rename(columns={'OBP':'OBP1','OBP_CUMSUM':'OBP2'},inplace=True)
    
    df['avg_OBP'] = df['OBP2'].mean()
    df['OBP_trend'] = (df['OBP2'] - df['avg_OBP']).astype(np.float16)
    df.drop(['avg_OBP'],axis=1,inplace=True)
    df = df[['PCODE', 'PA', 'AB', 'H', 'BB', 'HBP', 'SF','OB','OBP1', 'OBP_trend', 'OBP2']]
    df.drop('PCODE',axis=1,inplace=True)
    
    df_year_player_name = df
    
    return df_year_player_name     
#=================================================================
class Var_automation:
    def __init__(self, df, player, target, predict_range, mode, criterion):
        super(Var_automation, self).__init__()
        
        self.mode = mode
        self.criterion = criterion
        if self.mode=='train':
            self.df_set = df
            self.predict_range = predict_range
            self.player = player
            self.train = self.df_set.iloc[:-self.predict_range]
            self.test = self.df_set.iloc[-self.predict_range:]
            self.target = target
        
        else:
            zero_np = np.zeros((predict_range,len(df.columns)))
            df_np = df.to_numpy()
            df_concat = np.concatenate((df_np, zero_np))
            self.df_set = pd.DataFrame(df_concat, columns = df.columns)

            self.predict_range = predict_range
            self.player = player
            self.train = self.df_set.iloc[:-self.predict_range]
            self.test = self.df_set.iloc[-self.predict_range:]
            self.target = target

    #인과관계 검정
    def grangers_causation_matirx(self, test='ssr_chi2test', verbose=False): 
        variables = self.df_set.columns
        maxlag = 8
        
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns = variables, index = variables)
        for column in df.columns:
            for index in df.index:
                test_result = grangercausalitytests(self.df_set[[index,column]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[index, column] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        
        target_p_value = df.loc['{}_y'.format(self.target)]
        effective_var = []
        for i, v in enumerate(target_p_value):
            if v < 0.05:
                effective_var.append(df.index[i].split('_y')[0])
        
        return df, effective_var
    
    #공적분 검정
    def cointegration_test(self, alpha=0.05): 
        df_grangers, var_grangers = self.grangers_causation_matirx()
        var_coint = var_grangers
        var_coint.append(self.target) 
        df = self.df_set[var_coint] 
        df_shape = df.shape

        try:
            out = coint_johansen(df, -1, 5)
        except:
            #Singular Matrix Error Solution
            while True:
                try:
                    df = df + 0.00001*np.random.rand(df_shape[0],df_shape[1]) 
                    out = coint_johansen(df, -1, 5)
                    break
                except:
                    continue

        d = {'0.90': 0, '0.95': 1, '0.99': 2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1-alpha)]]
        def adjust(val, length= 6): return str(val).ljust(length)    
        
        dict = {'Name': [], 'Test Stat': [], '> C(95%)': [], 'Signif': []}
        effective_var = []

        for col, trace, cvt in zip(df.columns, traces, cvts):
            
            dict['Name'].append(adjust(col))
            dict['Test Stat'].append(adjust(round(trace,2), 9))
            dict['> C(95%)'].append(adjust(cvt, 8))
            dict['Signif'].append(trace > cvt)
            
            if trace>cvt:
                effective_var.append(col)
        
        df_coint = pd.DataFrame(dict)                
        
        return df_coint, effective_var
    
    #단위근 검정
    def adfuller_test(self, signif=0.05, verbose=False): 
        df_coint, effective_var = self.cointegration_test()
        effective_var.append(self.target)
        
        df_train = self.train[effective_var]
        df_test = self.test[effective_var]
        
        def adjust(val, length=6): return str(val).ljust(length)
        
        adfuller_dict = {}
        stationary_var = []
        
        for name, series in df_train.items():
            name = series.name
            
            dict = {'adfuller_test var': [], 'Significance Level': [], 'Test Statistic': [], 'No. Lags Chosen': []}
    
            r = adfuller(series, autolag = 'AIC')
            output = {'test_static': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
            p_value = output['pvalue']
            
            dict['adfuller_test var'].append(name)
            dict['Significance Level'].append(signif)
            dict['Test Statistic'].append(output['test_static'])
            dict['No. Lags Chosen'].append(output['n_lags'])
            
            for key, val in r[4].items():
                dict['Critical value {}'.format(adjust(key))] = round(val, 3)
                
            if p_value <= signif:
                dict['P-Value'] = '{} => Series is Stationary'.format(p_value)
                stationary_var.append(name)
            else:
                dict['P-Value'] = '{} => Series is Non-Stationary'.format(p_value)
            
            adfuller_dict[name] = dict
        if not self.target in stationary_var:
            stationary_var.append(self.target)

        return adfuller_dict, stationary_var
    
    def criterion_test(self, lag=10):
        dict_, effective_var = self.adfuller_test()
        df_train = self.train[effective_var]
        criterion = self.criterion

        model = sm.tsa.VAR(df_train)
        
        dict_criterion = {}
        if criterion == 'aic':
            for i in range(1, lag+1, 1):
                try:
                    result = model.fit(i)
                    dict_criterion[i] = result.aic
                except:
                    print()
                    print(f'AIC >> {self.player} >> {i} >> error')
                    print()
                    continue
        elif criterion == 'bic':
            for i in range(1, lag+1, 1):
                try:
                    result = model.fit(i)
                    dict_criterion[i] = result.bic
                except:
                    print()
                    print(f'BIC >> {self.player} >> {i} >> error')
                    print()
                    continue
        
            
        sorted_dict = sorted(dict_criterion.items(), key = lambda x: x[1])
        
        min_criterion = sorted_dict[0]
        
        min_lag = min_criterion[0]
        min_lag_result = min_criterion[1]
        
        return min_lag, min_lag_result, model, effective_var
    
    def var_fit(self, model_summary=False):
        min_lag, min_lag_result, model, effective_var = self.criterion_test()
        model_fitted = model.fit(min_lag)
        
        if model_summary:
            model_fitted.summary()
                        
        return model_fitted, effective_var
    
    def durbin_watson_test(self):
        model_fitted, effective_var = self.var_fit()
        df_train = self.train[effective_var]
        
        out = durbin_watson(model_fitted.resid)
        
        durbin_dict = {}
        for col, val in zip(df_train.columns, out_aic):
            durbin_dict[col] = round(val,2)
            
        return durbin_dict, model_fitted, effective_var
    
    def run(self, displaying = False): #nobs=20, 
        model_fitted, effective_var = self.var_fit()
        df_train = self.train[effective_var]
        df_test = self.test[effective_var]
        
        lag_order = model_fitted.k_ar
        forecast_input = df_train.values[-lag_order:]
        
        fc = model_fitted.forecast(y = forecast_input, steps=self.predict_range)
        df_forecast = pd.DataFrame(fc, index = df_test.index, columns = effective_var)
        
        #display
# =============================================================================
#         if displaying:
#             
#             display(df_forecast)
#         
#             fig,axes = plt.subplots(nrows=int(len(effective_var)/2), ncols=2, dpi=150, figsize=(10,10))
#             for i, (col,ax) in enumerate(zip(effective_var, axes.flatten())):
#                 df_test[col].plot(legend=True, ax=ax);
#                 df_forecast[col].plot(legend=True, ax=ax).autoscale(axis='x', tight=True)
# 
#                 ax.set_title(col + ": Forecast vs Actuals")
#                 ax.xaxis.set_ticks_position('none')
#                 ax.yaxis.set_ticks_position('none')
#                 ax.spines["top"].set_alpha(0)
#                 ax.tick_params(labelsize=6)
# 
#             plt.tight_layout()
#         true_mean = self.test[self.target].mean()
#         predict_mean = df_forecast[self.target].mean()
# =============================================================================
        
        return df_forecast[self.target].mean()
    
#=============================================================
var_obp=[]
non_var_obp=[]
for criterion in ['aic', 'bic']:
    for player in [67341, 67872, 68050, 75847, 76232, 76290, 78224, 78513, 79192, 79215]:
        try:
            var_model = Var_automation(eda_obp(player), player, 'OBP2', 21, mode='inference', criterion = criterion)
            OBP_pred = var_model.run()
            var_obp.append([criterion, player, OBP_pred])
        except:
            non_var_obp.append([criterion, player])

# Criterion 선택에 따른 OBP 결과를 데이터프레임으로
obp_df = pd.DataFrame()
obp_df['PCODE'] = np.nan
obp_df['AIC'] = np.nan  
obp_df['BIC'] = np.nan
aic_obp = var_obp[:10]  
bic_obp = var_obp[10:]
aic_df = pd.DataFrame(aic_obp, columns = ['AIC', 'PCODE', 'AIC_OBP'])
bic_df = pd.DataFrame(bic_obp, columns = ['BIC', 'PCODE', 'BIC_OBP'])
obp_df = pd.merge(aic_df, bic_df, how = 'right', left_on = 'PCODE', right_on = 'PCODE')
obp_df.drop(['AIC', 'BIC'], axis = 1, inplace = True)


        