# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:46:49 2019

@author: seanl
"""
import predictor
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import wealthiness_analysis as wa
abslote_path = os.path.abspath('.')
dataset1_file = abslote_path + "/../data/6ddcd912-32a0-43df-9908-63574f8c7e77.csv"
dataset2_file = abslote_path + "/../data/fy19fullpropassess.csv"

def race_analysis(combined_data):
    cls = predictor.EthnicityPredictor()
    cls.readProbabilityFromPkl('train_result.pkl')
    combined_data.info()
    combined_data = combined_data.dropna(subset=['applicant'])
    keyword = ' llc|company|llc '
    combined_data['applicant_category'] = combined_data['applicant'].str.contains(keyword)
    applicant_category = []
    for index, row in combined_data.iterrows():
        if row['applicant_category']:
            applicant_category.append("Co.")
        else:
            temp = row['applicant'].strip()
            temp = temp.replace("  "," ")
            split = temp.split(" ")
            if len(split) > 1:
                applicant_category.append(cls.classify(split[1]))
            else:
                applicant_category.append(cls.classify(split[0]))
    combined_data['applicant_category'] = applicant_category            
    #plot
    category = ['white','am.ind.', 'asian', 'black', 'hispanic','Co.']
    color = ['black','purple', 'orange', 'blue', 'lightgreen', 'red']
    size = [1,4,4,4,2,4]
    num_group = len(category)
    plt.figure(figsize=(15,15))
    for i in range(num_group):
        temp_df = combined_data[combined_data['applicant_category'] == category[i]]
        num = len(temp_df)
        plt.scatter(
                temp_df['long'], temp_df['lat'],
                c=color[i], marker='o',
                s=size[i],
                label=category[i]+ "   Num:"+str(num)
                )
    plt.title('The race distribution of applicants whom apply for energy intervention permit',fontsize=25 )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Longitute',fontsize=25 )
    plt.ylabel('Latitude',fontsize=25 )
    plt.grid()
    leg = plt.legend(fontsize=20,markerscale=10)
    for lh in leg.legendHandles: 
        lh._sizes = [250]
    for i in range(num_group):
        plot_each_race(combined_data,category[i],color[i])

def plot_each_race(combined_data,race,color):
    plt.figure(figsize=(15,15))
    temp_df1 = combined_data[combined_data['applicant_category'] != race]
    num1 = len(temp_df1)
    plt.scatter(
                temp_df1['long'], temp_df1['lat'],
                c='lightgrey', marker='o',
                s=4,
                label="non-"+race+ "   Num:"+str(num1)
                )
    temp_df2 = combined_data[combined_data['applicant_category'] == race]
    num2 = len(temp_df2)
    plt.scatter(
                temp_df2['long'], temp_df2['lat'],
                c=color, marker='o',
                s=10,
                label=race+ "   Num:"+str(num2)
                )
    plt.title(race+' applicants whom apply for energy intervention permit',fontsize=25 )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Longitute',fontsize=25 )
    plt.ylabel('Latitude',fontsize=25 )
    plt.grid()
    leg = plt.legend(fontsize=20,markerscale=10)
    for lh in leg.legendHandles: 
        lh._sizes = [250]
    
        
def calculate_declared_valuation(total_data_set):
    # extract OWN_OCC and declared_valuation to analysis data
    pre_group = total_data_set[['OWN_OCC', 'declared_valuation']]
    # calculate the mean of declared_valuation for each year
    group = pre_group.groupby(['OWN_OCC']).mean()
    group.reset_index(level=0, inplace=True)
    print(group)
    plt.bar(range(len(group['declared_valuation'])), group['declared_valuation'], tick_label=group['OWN_OCC'])
    plt.show()

def calculate_total_fees(total_data_set):
    # extract OWN_OCC and total_fees to analysis data
    pre_group = total_data_set[['OWN_OCC', 'total_fees']]
    # calculate the mean of total_fees for each year
    group = pre_group.groupby(['OWN_OCC']).mean()
    group.reset_index(level=0, inplace=True)
    print(group)
    plt.bar(range(len(group['total_fees'])), group['total_fees'], tick_label=group['OWN_OCC'])
    plt.show()
    
def calculate_number_of_building(house_data):
    # calculate the number of intervention for each year
    house_data = house_data.dropna(subset=['OWN_OCC'])
    group_data = house_data.groupby(['OWN_OCC']).count()
    group_data = group_data.reset_index(level=0, inplace=False)
    plt.bar(range(len(group_data['parcel_id'])), group_data['parcel_id'], tick_label=group_data['OWN_OCC'],width = 0.35)
    plt.title('Building occupies in engergy intervention permits')
    plt.xlabel('If occupied by owner')
    plt.ylabel('Permit number')
    plt.show()
    
df1 = wa.process_intervention_data_from_file(dataset1_file)
df2 = wa.process_house_data_from_file(dataset2_file)  
df3 = wa.connect_tables(df1,df2)  
calculate_number_of_building(df3)
race_analysis(df3)
