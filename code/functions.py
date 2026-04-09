# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:10:35 2025

@author: haley.synan
"""

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import pandas as pd
import xarray as xr 
import os
import ctd
from seabird.cnv import fCNV
from datetime import datetime
import gsw
from scipy.stats import kruskal
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython.display import display, HTML
from itertools import combinations
from scipy.signal import filtfilt
import math

def bland_altman_plot(up_merged, var='temp'):
    sub = up_merged[[col for col in up_merged.columns if var in col]]
    res = []
    differences = {}
    mean = {}
    for (i, j) in combinations(range(len(sub.columns)), 2):
        x, y = sub.iloc[:,i], sub.iloc[:,j]
        differences[f'{sub.columns[i]} vs {sub.columns[j]}'] = x-y
        mean[f'{sub.columns[i]} vs {sub.columns[j]}'] = (x+y)/2
    if var=='temp':
        fig, axes = plt.subplots(2,3, figsize=(14, 10)) # Adjust figsize as needed
        axes = axes.flatten() # Flatten the axes array for easier iteration
        for x in range(len(list(mean))):
            axes[x].scatter(mean[list(mean)[x]], differences[list(mean)[x]],s=5)
            axes[x].axhline(np.mean(differences[list(mean)[x]]), color='red', linestyle='--', label='Mean Diff')
            axes[x].axhline(np.mean(differences[list(mean)[x]]) + 1.96*np.std(differences[list(mean)[x]]), color='gray', linestyle='--', label='Limits of agreement')
            axes[x].axhline(np.mean(differences[list(mean)[x]]) - 1.96*np.std(differences[list(mean)[x]]), color='gray', linestyle='--')
            axes[x].set_title(list(mean)[x].replace(f'{var}_', ""))
            fig.supxlabel('Average Temperature (C)')
            fig.supylabel('Difference')
            fig.suptitle(f'Bland-Altman Plots \n All paired observations')
    elif var=='sal':
        x=0
        fig, axes = plt.subplots(1,1, figsize=(14, 10)) # Adjust figsize as needed
        axes.scatter(mean[list(mean)[x]], differences[list(mean)[x]],s=5)
        axes.axhline(np.mean(differences[list(mean)[x]]), color='red', linestyle='--', label='Mean Diff')
        axes.axhline(np.mean(differences[list(mean)[x]]) + 1.96*np.std(differences[list(mean)[x]]), color='gray', linestyle='--', label='Limits of agreement')
        axes.axhline(np.mean(differences[list(mean)[x]]) - 1.96*np.std(differences[list(mean)[x]]), color='gray', linestyle='--')
        axes.set_title(list(mean)[x].replace(f'{var}_', ""))
        fig.supxlabel('Average Salinity (PSU)')
        fig.supylabel('Difference')
        fig.suptitle(f'Bland-Altman Plots \n All paired observations')
    
        
def calculate_mbe(y_true, y_pred):
    """
    Calculates the Mean Bias Error (MBE).

    Args:
        y_true (np.array): Array of observed (true) values.
        y_pred (np.array): Array of predicted values.

    Returns:
        float: The Mean Bias Error.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    
    mbe = np.mean(y_pred- y_true)
    return mbe

def merge(sbe_up, ecage_up, edoors_up, rbr_up,var='temp'):
    merged = pd.merge(sbe_up, ecage_up, on='binned_pres', how='inner')
    merged = merged.rename(columns={var+'_x':var+'_sbe',var+'_y':var+'_ecage','sal00':'sal_sbe','datetime_x':'datetime_sbe','datetime_y':'datetime_ecage'})[[var+'_sbe',var+'_ecage','binned_pres','sal_sbe','datetime_sbe','datetime_ecage']]
    merged = pd.merge(merged, edoors_up, on='binned_pres', how='inner').rename(columns={var:var+'_edoors','datetime':'datetime_edoors'})[['binned_pres',var+'_sbe',var+'_ecage',var+'_edoors','sal_sbe','datetime_sbe','datetime_ecage','datetime_edoors']]
    merged = pd.merge(merged, rbr_up, on='binned_pres', how='inner').rename(columns={var:var+'_rbr','Salinity':'sal_rbr','datetime':'datetime_rbr',})[['binned_pres',var+'_sbe',var+'_ecage',var+'_edoors',var+'_rbr','sal_sbe','sal_rbr','datetime_sbe','datetime_ecage','datetime_edoors','datetime_rbr','station','cruise_ID']]
    return merged


def timelag_correction(temp, datetime, tau):
    datetime = pd.to_datetime(datetime)
    time_seconds = (datetime - datetime.iloc[0]).dt.total_seconds()
    # Compute dT/dt using np.gradient
    dTdt = np.gradient(temp, time_seconds)
    # Apply lag correction
    corrected_temp = temp + tau * dTdt
    return corrected_temp
    
    #dTdt = np.gradient(np.diff(temp),np.diff(datetime))
    #corrected_temp = temp[1:] + tau * dTdt
    #cor = pd.concat([pd.Series(temp[0]),corrected_temp])
    #cor = cor.replace([np.inf, -np.inf], np.nan)
    #if len(temp)!= len(cor):
    #    print('Lengths not properly calculated. Retry')
    #    return
    #else:
    #    return cor

def load_data(fnames, cruise_ID):
    sbe = []
    rbr=[]
    ecage=[]
    edoors=[]
    subset = [fname for fname in fnames if cruise_ID in fname]
    if fnames[0].__contains__('SBE'):
        for file in subset:
            df=ctd.from_cnv(file)
            df['Time'] = fCNV(file).attributes['datetime'] #add STATIC start time
            df = df.reset_index()
            if 'timeM' in df.columns:
                df['datetime'] = [df['Time'][x] + pd.Timedelta(seconds=df.timeM[x]) for x in df.index] #get dynamic datetime (add seconds since cast start to static datetime)
            elif 'timeS' in df.columns:
                df['datetime'] = [df['Time'][x] + pd.Timedelta(seconds=df.timeS[x]) for x in df.index] #get dynamic datetime (add seconds since cast start to static datetime)
            df['cruise_ID'] = [file.split('\\')[8]]*len(df) #add cruise ID based on file name: either IFS_2024, IFS_2025, or SUP_2025
            df['station'] = int(file.split('\\')[9].split('.')[0].split('e')[1]) #add station ID from filename
            sbe.append(df.reset_index()) 
        data = pd.concat(sbe).reset_index() #get all stations from all cruises loaded into 1 dataframe
    elif fnames[0].__contains__('RBR'):
        for file in subset:
            df = pd.read_csv(file)
            try:
                df=df.drop('PressureBins',axis=1)
            except:
                pass
            if file.__contains__('2024'):
                df['station'] = [int(row.split('-')[0])-1 for row in df.tow] #get station names for 2024 formatting
            elif file.__contains__('DyrstenCruise2025'):
                df['station'] = [df.tow[x].split('-')[0] for x in range(len(df))] #get station name for 2025
            elif file.__contains__('Supplemental'):
                df['station'] = df.profile
            df['cruise_ID'] = [file.split('\\')[8]]*len(df) #add cruise ID based on file name: either IFS_2024, IFS_2025, or SUP_2025
            rbr.append(df)
        rbr = pd.concat(rbr).reset_index()
        rbr['Time'] = format_dates(rbr)
        data=rbr.rename(columns={'Pressure':'air_pres'})
    elif fnames[0].__contains__('Doors'):
        counter = 1 
        for file in subset:
            df=pd.read_csv(file)
            if file.__contains__('IFS_2025'):
                if counter < 8: 
                    df['station'] = counter
                elif counter == 8: 
                    df['station'] = np.where(df.index < 1005, 8, 9) #2 casts in 1 csv.. manually split into correct stationnumbers 
                else:
                    df['station'] = counter + 1
            if file.__contains__('SUP_2025'):
                df['station'] = counter  #ifs 2025 only has 15 profiles... subtract 14 to start from 1 again for the supplemental data
            df['cruise_ID'] = [file.split('\\')[9]]*len(df) #add cruise ID based on file name: either IFS_2024, IFS_2025, or SUP_2025
            counter = counter + 1
    
            edoors.append(df) 
        data = pd.concat(edoors).reset_index()
    elif fnames[0].__contains__('Cage'):
        counter =1 
        for file in subset:
            df=pd.read_csv(file)
            df['station'] = counter
            df['cruise_ID'] = [file.split('\\')[9]]*len(df) #add cruise ID based on file name: either IFS_2024, IFS_2025, or SUP_2025
            if file.__contains__('SUP_2025'):
                df['station'] = counter #- 16 #MISSING 1 SUPPLEMENTARY PROFILE????
            counter = counter + 1
            ecage.append(df) 
        data = pd.concat(ecage).reset_index()
    data = format_raw(data)
    data = data[data.pres>0]
    return data


def bin_merge_2(sbe,rbr):
    data=[]
    for xx in range(sbe.station.min(),sbe.station.max()):
        #split by station
        sbe_sub = sbe[sbe.station==xx]
        rbr_sub = rbr[rbr.station==str(xx)]
        if len(rbr_sub)==0:
            rbr_sub = rbr[rbr.station==xx]
        sbe_down, sbe_up = pres_bin(split_cast(sbe_sub)[0]),pres_bin(split_cast(sbe_sub)[1])
        try:
            rbr_down, rbr_up = pres_bin(rbr_sub[rbr_sub['tow'].str.contains('-A')]), pres_bin(rbr_sub[rbr_sub['tow'].str.contains('-C')])
        except KeyError:
            rbr_down, rbr_up = pres_bin(split_cast(rbr_sub)[0]),pres_bin(split_cast(rbr_sub)[1]) #supplemental 2025 data was taken as casts only, no tows.
        if 'Practical_Salinity' in rbr_down.columns:
            rbr_down = rbr_down.rename(columns={'Practical_Salinity':'Salinity'})
        d=pd.merge(sbe_up, rbr_down, on='binned_pres',how='inner').rename(columns={'temp_x':'temp_sbe','temp_y':'temp_rbr','datetime_x':'datetime_sbe',
                                                                             'datetime_y':'datetime_rbr','sal00':'sal_sbe','Salinity':'sal_rbr',
                                                                             'station_x':'station','cruise_ID_x':'cruise_ID',})[['binned_pres','temp_sbe','temp_rbr','datetime_sbe',
                                                                             'datetime_rbr','sal_sbe','sal_rbr','station','cruise_ID']]
        d['temp_ecage'] = [float('nan')]*len(d)
        d['temp_edoors'] = [float('nan')]*len(d)
        d['datetime_ecage'] = [float('nan')]*len(d)
        d['datetime_edoors'] = [float('nan')]*len(d)
        d[['binned_pres', 'temp_sbe','temp_ecage','temp_edoors', 'temp_rbr','sal_sbe','sal_rbr','datetime_sbe','datetime_ecage','datetime_edoors','datetime_rbr','station','cruise_ID',]]
        data.append(d)
    dat = pd.concat(data)
    return dat

def bin_merge_4(sbe, rbr, edoors, ecage):
    data=[]
    for xx in range(sbe.station.min(),sbe.station.max()):
        #split by station
        sbe_sub = sbe[sbe.station==xx]
        rbr_sub = rbr[rbr.station==str(xx)]
        edoors_sub = edoors[edoors.station==xx]
        edoors_sub = subset_tows(edoors_sub)
        ecage_sub = ecage[ecage.station==xx]
        #put into 1 m bins
        sbe_down, sbe_up = pres_bin(split_cast(sbe_sub)[0]),pres_bin(split_cast(sbe_sub)[1])
        ecage_down, ecage_up = pres_bin(split_cast(ecage_sub)[0]),pres_bin(split_cast(ecage_sub)[1])
        rbr_down, rbr_up = pres_bin(rbr_sub[rbr_sub['tow'].str.contains('-A')]), pres_bin(rbr_sub[rbr_sub['tow'].str.contains('-C')])
        edoors_down, edoors_up = pres_bin(edoors_sub[edoors_sub.tow=='A']), pres_bin(edoors_sub[edoors_sub.tow=='C'])
        #merge into 1 dataframe
        data.append(merge(sbe_up, ecage_up, edoors_down, rbr_down).reset_index())
    
    dat = pd.concat(data)
    return dat

#def merge(sbe, ecage, edoors, rbr,var='temp'):
#    merged = pd.merge(sbe, ecage, on='binned_pres', how='inner')
#    merged = merged.rename(columns={var+'_x':var+'_sbe',var+'_y':var+'_ecage','sal00':'sal_sbe',
#                                   'datetime_x':'datetime_sbe','datetime_y':'datetime_ecage','station_x':'station','cruise_ID_x':'cruise_ID'})[[var+'_sbe',var+'_ecage','binned_pres','sal_sbe','datetime_sbe','datetime_ecage']]
#    merged = pd.merge(merged, edoors, on='binned_pres', how='inner').rename(columns={var:var+'_edoors','datetime':'datetime_edoors'})[[var+'_sbe',var+'_ecage','binned_pres','sal_sbe','datetime_sbe','datetime_ecage','temp_edoors','datetime_edoors']]
#    merged= pd.merge(merged, rbr, on='binned_pres', how='inner').rename(columns={var:var+'_rbr','datetime':'datetime_rbr','Salinity':'sal_rbr'})[['binned_pres', var+'_sbe',var+'_ecage','temp_edoors', 'temp_rbr','sal_sbe','sal_rbr','datetime_sbe','datetime_ecage','datetime_edoors','datetime_rbr','station','cruise_ID',]]
#    return merged
    

def process_rbr(df):
    up_cast = []
    down_cast= []
    df['tow_og'] = df.tow
    df = subset_tows(df)
    grouped = df.groupby(['station','cruise_ID'])
    for group_name, group_df in grouped:
        if group_name[1]=='SUP_2025':
            df_down, df_up = pres_bin(split_cast(group_df)[0]),pres_bin(split_cast(group_df)[1])
            up_cast.append(df_up)
            down_cast.append(df_down)
        elif group_name[1] == 'IFS_2024':
            df_down =group_df[group_df['tow_og'].str.contains('-A')]
            df_up =group_df[group_df['tow_og'].str.contains('-C')]
            df_up, df_down = pres_bin(df_up),pres_bin(df_down)
            up_cast.append(df_up)
            down_cast.append(df_down)
        else: 
            df_up =group_df[group_df.tow=='C']
            df_down = group_df[group_df.tow=='A']
            if len(df_down) or len(df_up) ==0:
                df_down, df_up = pres_bin(split_cast(group_df)[0]),pres_bin(split_cast(group_df)[1])
                up_cast.append(df_up)
                down_cast.append(df_down)
            else: 
                df_up, df_down = pres_bin(df_up),pres_bin(df_down)
                up_cast.append(df_up)
                down_cast.append(df_down)
    return pd.concat(up_cast),pd.concat(down_cast)

def process_casts(df):
    up_cast = []
    down_cast= []
    grouped=df.groupby(['station','cruise_ID'])
    for group_name, group_df in grouped:
        cast_down, cast_up = pres_bin(split_cast(group_df)[0]),pres_bin(split_cast(group_df)[1])
        up_cast.append(cast_up)
        down_cast.append(cast_down)
    return pd.concat(up_cast), pd.concat(down_cast)

def format_dates(df):
    #rbr times are not consistently formatted.. fix 
    formatted=[]
    for date_str in df.Time:
        try:
            date_str.split('.')[1]
            formatted.append(date_str)
        except:
            try: 
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                formatted.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
            except:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                formatted.append(dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return formatted

def get_fnames(base_dir):
    file_paths = []
    for dirpath, dirnames, fnames in os.walk(base_dir):
        for fname in fnames:
            full_path = os.path.join(dirpath, fname)
            file_paths.append(full_path)
    return file_paths

def format_raw(df): 
    df['Time'] = df[df.columns[[e.__contains__('Time') for e in df.columns].index(True)]] #check for time column names --> rename

    if type(df['Time'][0]) == pd._libs.tslibs.timestamps.Timestamp:
        df['datetime'] = df.Time
    else: 
        try:
            df['datetime'] = [datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ') for x in df['Time']] #convert to datetime 
        except: 
            try:
                try:
                    df['datetime'] = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f") for x in df['Time']] #convert to datetime 
                except:
                    df['datetime'] = [datetime.strptime(x,"%Y-%m-%d %H:%M:%S") for x in df['Time']] 
            except TypeError:
                pass
    try: 
        df = df.rename(columns={[df.columns[[e.__contains__('Temperature') for e in df.columns].index(True)]][0]:'temp'}) #rename to temp
    except: 
        df = df.rename(columns={[df.columns[[e.__contains__('tv290C') for e in df.columns].index(True)]][0]:'temp'})
    df = df.rename(columns={[df.columns[[e.__contains__('ressure') for e in df.columns].index(True)]][0]:'pres'}) #rename to pres
    df['timestamp'] = [e.timestamp() for e in df.datetime] #convert to numeric
    return df

from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
def check_datetimes(dates):
    # Generate all unique name pairs and compute differences
    differences = {
        (name1, name2): abs(dates[name1] - dates[name2])
        for name1, name2 in combinations(dates, 2)}
    
    # Print results using names
    nm1 = []
    nm2 = [] 
    dif = []
    for (name1, name2), diff in differences.items():
        if diff.seconds/60 > 40: 
            nm1.append(name1)
            nm2.append(name2)
            dif.append(diff)
            #print(f"Difference between '{name1}' and '{name2}': {diff}")
        else: 
            pass 
        return 1 
    if nm1 == []:
        print('Times are all matched (within 40 minutes). Pre-processing can begin')
    elif Counter(nm1).most_common()[0][1] == 1 and Counter(nm2).most_common()[0][1] == 1: 
        print('There is a problem with multiple of the sensors datetimes. Check manually that stations match.')
        return None
    elif Counter(nm1).most_common()[0][1] > 1 or Counter(nm2).most_common()[0][1] > 1:
        if Counter(nm1).most_common()[0][1] > 1: 
            print(f"{Counter(nm1).most_common()[0][0]} is more than 40 minutes apart from the other sensors.  Check that the station matches." )
            [print(f'Difference between {nm1[x]} and {nm2[x]} is {dif[x]}') for x in range(len(nm1))][0]
        elif Counter(nm2).most_common()[0][1] > 1:
            print(f"{Counter(nm2).most_common()[0][0]} is more than 40 minutes apart from the other sensors. Check that the station matches." )
            [print(f'Difference between {nm1[x]} and {nm2[x]} is {dif[x]}') for x in range(len(nm1))][0]
        return None

def split_cast(df):
    max_pres = df['pres'].argmax()
    down = df.iloc[:max_pres + 1]
    up = df.iloc[max_pres + 1:][::-1] #
    return down, up 


def subset_tows(df,bottom_thresh=70):
    is_bottom = (np.abs(df['pres'] - df.pres.values.max()) < bottom_thresh) #first subset by a bottom threshold (make sure data is within x meters of bottom)
    sub = df[is_bottom==True]
    down_break_point = np.where((np.diff(sub.pres.values.round(2))<0.2))[0][0]
    up_break_point = np.where((np.diff(sub.pres.values.round(2))>0.2))[0][-1]
    down_break_val = sub.pres.values[down_break_point]
    up_break_val = sub.pres.values[up_break_point]
    down_idx =df[df.pres==down_break_val].index[0]
    up_idx = df[df.pres==up_break_val].index[0]
    downcast = df[:down_idx]
    tow = df[down_idx:up_idx]
    upcast = df[up_idx:]
    df['tow'] = [np.nan] * len(df)
    df.loc[:down_idx, 'tow'] = 'A'
    df.loc[down_idx:up_idx, 'tow'] = 'B'
    df.loc[up_idx:, 'tow'] = 'C'
    #return downcast,tow,upcast
    return df

def pres_bin(df, bin_size=1):
    #from google AI
    bin_edges = np.arange(np.floor(df.pres.min()), np.ceil(df.pres.max()) + bin_size, bin_size)
    df['pres_bin'] = pd.cut(df['pres'], bins=bin_edges, include_lowest=True) #add column defining bin 

    # Now you can group by depth_bin, for example, averaging temperature in each:
    binned_avg = df.groupby('pres_bin').mean(numeric_only=True) #group by bin
    #binned_avg['binned_pres'] = binned_avg.pres.round().values
    binned_avg['binned_pres'] = np.floor(binned_avg.pres.values)
    binned_avg = binned_avg.reset_index()
    binned_avg['datetime'] = df[['pres_bin','datetime']].groupby('pres_bin').mean().reset_index().datetime
    #binned_avg['datetime'] = [pd.to_datetime(datetime.fromtimestamp(t)) if not math.isnan(t) else float('nan') for t in binned_avg.timestamp]
    binned_avg['cruise_ID'] = df.cruise_ID.iloc[0]
    binned_avg['station'] = df.station.iloc[0]
    #binned_avg['datetime'] = [pd.to_datetime(datetime.fromtimestamp(t)) for t in binned_avg.timestamp if not math.isnan(t)]
    return binned_avg

#def merge(sbe_up, ecage_up, edoors_up, rbr_up,var='temp'):
#    merged = pd.merge(sbe_up, ecage_up, on='binned_pres', how='inner')
#    merged = merged.rename(columns={var+'_x':var+'_sbe',var+'_y':var+'_ecage'})[[var+'_sbe',var+'_ecage','binned_pres']]
#    merged = pd.merge(merged, edoors_up, on='binned_pres', how='inner').rename(columns={var:var+'_edoors'})
#    merged = pd.merge(merged, rbr_up, on='binned_pres', how='inner').rename(columns={var:var+'_rbr'})[[var+'_sbe',var+'_ecage',var+'_edoors',var+'_rbr','binned_pres']]
#    return merged

def get_descent_rate(df,sensor, dr={}):
    df=df[df.pres>1]
    if sensor=='sbe':
        for x in df.station.unique(): 
            sub=df[df.station==x] #split into single station
            sub=split_cast(sub)[1] #get upcast only
            if 'timeM' in sub.columns:
                sub['datetime'] = [sub['datetime'][x] + pd.Timedelta(minutes=sub.timeM[x]) for x in sub.index] #get dynamic datetime (add seconds since cast start to static datetime)
            elif 'timeS' in sub.columns:
                sub['datetime'] = [sub['datetime'][x] + pd.Timedelta(seconds=sub.timeS[x]) for x in sub.index] #get
            dr.update({f'{df.cruise_ID.iloc[0]}_{x}':(sub.pres.max()-sub.pres.min())/(sub.datetime.max()-sub.datetime.min()).total_seconds()})
    elif sensor=='rbr':
        try:
            for x in df.station.unique(): 
                sub=df[df.station==x] #split into single station
                sub =sub[sub['tow'].str.contains('-A')] #get downcast
                dr.update({f'{df.cruise_ID.iloc[0]}_{x}':(sub.pres.max()-sub.pres.min())/(sub.datetime.max()-sub.datetime.min()).total_seconds()})
        except:
            for x in df.station.unique(): 
                try:
                    sub=df[df.station==x] #split into single station
                    sub =split_cast(sub)[0] #get downcast
                    dr.update({f'{df.cruise_ID.iloc[0]}_{x}':(sub.pres.max()-sub.pres.min())/(sub.datetime.max()-sub.datetime.min()).total_seconds()})
                except ValueError:
                    print('')
    elif sensor=='ecage':
        for x in df.station.unique():
            sub=df[df.station==x] #split into single station
            sub=split_cast(sub)[1] #get upcast only
            dr.update({f'{df.cruise_ID.iloc[0]}_{x}':(sub.pres.max()-sub.pres.min())/(sub.datetime.max()-sub.datetime.min()).total_seconds()})
    elif sensor=='edoors':
        for x in df.station.unique():
            sub=df[df.station==x] #split into single station
            sub=split_cast(sub)[1] #get upcast only
            dr.update({f'{df.cruise_ID.iloc[0]}_{x}':(sub.pres.max()-sub.pres.min())/(sub.datetime.max()-sub.datetime.min()).total_seconds()})
    else:
        raise ExceptionName("Sensor name not recognized. Valid options include 'sbe','rbr','ecage','edoors'")
        
    return dr
