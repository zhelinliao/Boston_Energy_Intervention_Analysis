# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
import random

abslote_path = os.path.abspath('.')

dataset1_file = abslote_path + "/../data/6ddcd912-32a0-43df-9908-63574f8c7e77.csv"
dataset2_file = abslote_path + "/../data/fy19fullpropassess.csv"

headname_interv =['permitnumber','comments','applicant','issued_date',
           'property_id','parcel_id','lat','long','zip','city']
headname_house =['PID','GIS_ID','AV_LAND','AV_BLDG','AV_TOTAL','GROSS_AREA']
keyword_interv = 'solar|hot water|photovoltaic|ground source|gshp|air source|heat pump|insulation|gas stove|hvac|boiler|wiring|furnace|weather stripping'
choose_color = ['purple', 'cadetblue', 'orange', 'green', 
                    'yellow', 'pink','blue']

def process_permit_data_noninter_interv(dataset_file):
    '''
    Differentiate the invervention related data and  non-invervention related data
    from dataset
    '''
    raw_data = pd.read_csv(dataset_file)[headname_interv]
    raw_data = raw_data.drop_duplicates(keep='first')
    raw_data['comments'] = raw_data['comments'].str.lower()
    # drop null
    raw_data = raw_data.dropna(subset=['lat','long'])
    raw_data['comments']=raw_data['comments'].replace({np.nan:' '})
    #select intervention related sample 
    interv = raw_data[raw_data['comments'].str.contains(keyword_interv)]
    non_interv = raw_data[~raw_data['permitnumber'].isin(interv['permitnumber'])]
    return interv, non_interv
    
def plot_noninter_interv(interv, non_interv):
    '''
    Plot the all permits including intervention related permits and non-intervention  
    related permits
    '''
    a = interv.shape[0]
    b = non_interv.shape[0]
    plt.figure(figsize=(30,30))
    plt.scatter(
                non_interv['long'], non_interv['lat'],
                c='black', marker='o',s=1,
                label='Non-intervention Permit'
                )
    plt.scatter(
                interv['long'], interv['lat'],
                c='red', marker='o', s=1,
                label='Intervention Permit'
                )
    plt.text(-71.175,42.380,'Number of non-intervention related permits:'+str(b), size = 25)
    plt.text(-71.175,42.375,'Number of intervention related permits:'+str(a), size = 25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title('Building permits for construction project',fontsize=25)
    plt.xlabel('Longtitude',fontsize=25)
    plt.ylabel('Latitude',fontsize=25)
    plt.grid()
    plt.legend(fontsize=50,markerscale=30)

def judging_solar_df(dataset_file):
    # Count solar related energy intervention
    solar_df = dataset_file[['comments']]
    solar_df = dataset_file[dataset_file['comments'].str.contains('hot water|photovoltaic|solar')]
    return solar_df

def judging_heatpump_df(dataset_file):
    heatpump_df = dataset_file[['comments']]
    heatpump_df = dataset_file[dataset_file['comments'].str.contains('heat pump|ground source|gshp|air source')]
    return heatpump_df

def judging_furnace_df(dataset_file):
    furnace_df = dataset_file[['comments']]
    furnace_df = dataset_file[dataset_file['comments'].str.contains('furnace|boiler|gas stove')]
    return furnace_df

def judging_HVAC_df(dataset_file):
    HVAC_df = dataset_file[['comments']]
    HVAC_df = dataset_file[dataset_file['comments'].str.contains('hvac')]
    return HVAC_df

def judging_wiring_df(dataset_file):
    wiring_df = dataset_file[['comments']]
    wiring_df = dataset_file[dataset_file['comments'].str.contains('wiring')]
    return wiring_df

def judging_insulation_df(dataset_file):
    insulation_df = dataset_file[['comments']]
    insulation_df = dataset_file[dataset_file['comments'].str.contains('insulation|weather stripping')]
    return insulation_df

def count_energy_intervention_by_type(data):
    solar_df = judging_solar_df(data)
    heatpump_df = judging_heatpump_df(data)
    HVAC_df = judging_HVAC_df(data)
    wiring_df = judging_wiring_df(data)
    insulation_df = judging_insulation_df(data)
    furnace_df = judging_furnace_df(data)
    size_list = [len(solar_df),
             len(heatpump_df),
             len(HVAC_df),
             len(wiring_df),
             len(insulation_df),
             len(furnace_df)]
    plt.figure(figsize=(10,10))
    namelist = ['Solar', 'Heatpump', 'HVAC', 'Wiring', 'Insulation','Furnace']
    plt.bar(namelist, size_list)
    for x, y in enumerate(size_list):
        plt.text(x-0.2, y+400, "%s" %y,fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Energy intervention type',fontsize=20)
    plt.ylabel('Number',fontsize=20)
    plt.title('Different energy interventions in construction projects within Boston',fontsize=25)
    plt.show()

interv, non_interv = process_permit_data_noninter_interv(dataset1_file)
plot_noninter_interv(interv, non_interv)
count_energy_intervention_by_type(interv)

