# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import math

from covid19_analysis import __version__

__author__ = "J SAYRITUPAC"
__copyright__ = "J SAYRITUPAC"
__license__ = "mit"

# Calculate the population over time for a given double time magnitude
def doubling_time_fun(pop_init, num_days, grow_rate, t0=0):
    '''Doubling time calculation, generate an array with the double time value which follows: 
        P(t)=P0 * e^(t*ln(2)/T), with T=growing rate        
        pop_init:   <int> initial population
        num_days:   <int> number of days to plot
        grow_rate:  <int> growing ratio in days
        t0:         <int> time shift for growing calculation (NOT IMPLEMENTED YET)
        '''
    ndays =  np.array(range(0, num_days))
    new_pop = doubling_time_equation(pop_init, ndays,grow_rate)
    return new_pop

# Define doubling time equation
def doubling_time_equation(pop_init, num_day, grow_rate):
    '''Define the equation for doubling time calculation: 
        P(t)=P0 * e^(t*ln(2)/T), with T=growing rate        
        pop_init:   <int> initial population
        num_days:   <int> number of days past t0
        grow_rate:  <int> growing ratio in days
        
        Reference
            [1] https://en.wikipedia.org/wiki/Exponential_growth
            [2] https://mathinsight.org/doubling_time_half_life_discrete
            [3] http://sites.science.oregonstate.edu/~landaur/INSTANCES/WebModules/2_DecayGrowth/BiologicalGrowth/Pdfs/StudentReadings.pdf
        '''
    # doubling eq. as P(t)=P0 * e^(t*ln(2)/T), with T=growing rate
    if type(num_day) == int:
        new_pop = np.ceil(pop_init * math.e ** (num_day * np.log(2) / grow_rate))
    else:
        new_pop = np.ceil(pop_init * math.e ** (np.array(num_day) * np.log(2) / grow_rate))
    return new_pop

# Provide a timeseries for a define country from JHU dataset
def get_timeseries_from_JHU(df_jhu, country_name, mainland = True, verbose=True):
    '''Provide a timeseries for a define country from JHU dataset. 
        df_jhu:         <dataframe> Dataset read from JHU repository
        country_name:   <string> Name of the country within the JHU country list
        mainland:       <boolean> Allows to choose between have only mainland data or all places data, True by default
        verbose:        <boolean> Display message for the user from data extraction
        '''
    if country_name is 'all':
        # Calculate the sum of all cases
        temp_array = df_jhu.sum(axis=0, numeric_only=True)
        df_out = df_jhu.head(1).copy()
        for c in temp_array.index:
            if c != 'Lat' and c != 'Long':
                df_out[c] = temp_array[c]

    elif mainland:
        list_province = df_jhu['Province/State'].loc[df_jhu['Country/Region'] == country_name].unique()
        
        # check if exist more than one Province/Region
        if list_province.size > 1:
            if verbose: print('Warning: %s has several Province/State' %(country_name))
            if any(pd.isna(list_province)):
                if verbose: print('Warning: Only mainland was taken for %s' %(country_name))
                df_out = df_jhu.loc[(df_jhu['Country/Region'] == country_name) & (pd.isna(df_jhu['Province/State']))]
            
            else:
                if verbose: print('Warning: data for %s is the sum of all Provice/State' %(country_name))
                # calculate aggregate data
                df_tmp = df_jhu.loc[df_jhu['Country/Region'] == country_name]
                if country_name == 'US': # 'US' special case
                    just_states =  [re.search(', ', prov) == None for prov in list_province] 
                    df_tmp = df_tmp.loc[just_states]
                temp_array = df_tmp.sum(axis=0, numeric_only=True)
                df_out = df_tmp.head(1).copy()
                for c in temp_array.index:
                    if c != 'Lat' and c != 'Long':
                        df_out[c] = temp_array[c]
            
        else:
            df_out = df_jhu.loc[df_jhu['Country/Region'] == country_name]
    else:
        # calculate aggregate data
        df_tmp = df_jhu.loc[df_jhu['Country/Region'] == country_name]
        temp_array = df_tmp.sum(axis=0, numeric_only=True)
        df_out = df_tmp.head(1).copy()
        for c in temp_array.index:
            if c != 'Lat' and c != 'Long':
                df_out[c] = temp_array[c]

    # get timeseries
    ts_country = pd.Series(data=df_out.iloc[0][4:].fillna(0).values, index=pd.to_datetime(df_out.columns[4:]), dtype=int)
    return ts_country

# Allow to select one country from the JHU dataset (merger all regions or just mainland)
def select_country(df_all, country_name, just_mainland = True):
    '''Provide a data-frame with the data from the selected country. Note: variable  'just_mainland' equal false,  will sum all Province/States'''
    if just_mainland:
        # check if exist more than one Province/Region
        if df_all['Province/State'].loc[df_all['Country/Region'] == country_name].size > 1:
            print('Warning: %s has more than one Province/State, only mainland was took on the output dataframe' %(country_name))
            df_out = df_all.loc[(df_all['Country/Region'] == country_name) & (df_all['Province/State'] == country_name)]
        else:
            df_out = df_all.loc[df_all['Country/Region'] == country_name]
        return df_out
    else:
        df_tmp = df_all.loc[df_all['Country/Region'] == country_name]
        temp_array = df_tmp.sum(axis=0, numeric_only=True)
        df_out = df_tmp.head(1).copy()
        for c in temp_array.index:
            if c != 'Lat' and c != 'Long':
                df_out[c] = temp_array[c]
        df_out['Province/State'] = country_name
        return df_out

# Define a division for two vectors (array dim 1) when the divisor has zero
def safe_div(x,y):
    ''' Calculate a division between two vector on which the divisor have a zero value. The final result will have zero as well:
        z = x / y
        '''
    isZero = (y == 0)
    y2 = np.array(y)
    y2[isZero] = 1
    res = x / y2
    res[isZero] = 0
    return res

# define moving mean (rolling average) based on a convolution function
def mov_avg(data_set, periods=3, conv_mode = 'full'):
    ''' Moving average / rolling mean, based on a convolution function
        data_set : data to treat
        periods : points to consider within the rolling window
        conv_mode : select among 'valid', 'same', 'full'
    '''
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode=conv_mode)

# Ancient function. Define a new dataframe from JHU dataframe by reshaping columns by rows and excluding some variables (lat & long)
def recreate_df(raw_df):
    '''OLD FUNCTION: Create a dataframe based on the DF provide by the JHU repository'''
    # identify columns and datetime data
    col_names = raw_df.columns
    date_data = pd.to_datetime(raw_df.columns[4:])
    
    # build columns header as country - province (if not empty)
    region_col = pd.Series(data=raw_df['Province/State'], dtype='str')
    country_col = pd.Series(data=raw_df['Country/Region'], dtype='str')
    col_headers = country_col.str.cat(region_col, sep=(' - '))
    col_headers = col_headers.str.rstrip(' nan').str.rstrip(' -')
    
    # Build dataframe without coordinates and with time as row + countries as columns
    new_df = pd.DataFrame(data=date_data, columns=['Date'])
    for cidx, c in enumerate(col_headers):
        data_tmp = np.array(raw_df.iloc[cidx][4:], dtype=int)
        new_df[c] = data_tmp
    return new_df
