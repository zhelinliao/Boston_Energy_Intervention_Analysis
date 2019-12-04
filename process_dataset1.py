# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:32:16 2019

@author: seanl
"""

import os
import predictor
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
import folium
from folium.plugins import HeatMap
abslote_path = os.path.abspath('..')

dataset1_file = abslote_path + "/6ddcd912-32a0-43df-9908-63574f8c7e77.csv"
dataset2_file = abslote_path + "/fy19fullpropassess.csv"

headname_interv =['permitnumber','comments','applicant','issued_date',
           'property_id','parcel_id','lat','long','zip','city']
headname_house =['PID','GIS_ID','AV_LAND','AV_BLDG','AV_TOTAL']
keyword_test = 'solar|hot water|photovoltaic|ground source|gshp|air source|heat pump|insulation|gas stove|hvac|boiler|wiring|furnace|weather stripping'

choose_color = ['purple', 'cadetblue', 'orange', 'green', 
                    'yellow', 'pink']


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
    interv = raw_data[raw_data['comments'].str.contains(keyword_test)]
    non_interv = raw_data[~raw_data['permitnumber'].isin(interv['permitnumber'])]
    print(interv.head())
    print(non_interv.info())
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
                non_interv['lat'], non_interv['long'],
                c='black', marker='o',s=1,
                label='Non-intervention Permit'
                )
    plt.scatter(
                interv['lat'], interv['long'],
                c='red', marker='o', s=1,
                label='Intervention Permit'
                )
    plt.text(42.225,-71.025,'Number of non-intervention related permits:'+str(b), size = 25)
    plt.text(42.225,-71.035,'Number of intervention related permits:'+str(a), size = 25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.legend(fontsize=50,markerscale=30)
       
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
    print(combined_data['applicant_category'].value_counts())
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
                temp_df['lat'], temp_df['long'],
                c=color[i], marker='o',
                s=size[i],
                label=category[i]+ "   Num:"+str(num)
                )
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.legend(fontsize=20,markerscale=10)
    
def area_group_by_zipcode_analysis(data):
    data.info()
    #print(data['city'].value_counts())
    
    tmp = data['GIS_ID'].value_counts().keys()
    tmp1 = data['GIS_ID'].value_counts().values
    df1 = pd.DataFrame(tmp,columns=['GIS_ID'])
    df2 = pd.DataFrame(tmp1,columns=['freq'])
    dd = pd.concat([df1,df2],axis=1)
    data_tmp = data[['lat','long','GIS_ID']] 
    data_tmp = data_tmp.drop_duplicates(['GIS_ID'],keep='first')
    data_done = pd.merge(data_tmp,dd,on='GIS_ID')
    create_heatmap_permit_freq(data_done)
    
    
def create_heatmap_value(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat','long','AV_COMBINE']], radius=9, max_zoom
        =50,name="the heapmap of the property values").add_to(base_map)
    base_map.save('index_heatmap_value.html')
    
def create_heatmap_permit_freq(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat','long','freq']], radius=9, max_zoom
        =13,name="the heapmap of the permit frequency").add_to(base_map)
    base_map.save('index_heatmap_permit_freq.html')
    
def create_heatmap_permit_intensity(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat','long']], radius=9, max_val=0.5,max_zoom
        =20).add_to(base_map)
    base_map.save('index_heatmap_permit_intensity.html')    
    
def generateBaseMap(defaultlocation =[42.358380,-71.053950]):
    basemap = folium.Map(location=defaultlocation)
    return basemap

def mark_on_real_map(dataset,kpp):
    dataset_df = pd.DataFrame(dataset,columns=['lat','long','AV_COMBINE'])
    res = pd.DataFrame(kpp.labels_,columns=['cluster'])
    temp = pd.concat([dataset_df,res],axis=1)
    

    basemap = generateBaseMap()
    for index, row in temp.iterrows():
        la = row['lat']
        lo = row['long']
        cluster = int(row['cluster'])
        folium.Circle(location=[la,lo],
                      radius=1,
                      color=choose_color[cluster],
                      fill=True,
                      fill_color=choose_color[cluster]).add_to(basemap)
    basemap.save('index_map.html')
        
def process_intervention_data_from_file(dataset_file):
    raw_data = pd.read_csv(dataset_file)[headname_interv]
    # drop null
    raw_data['parcel_id']=raw_data['parcel_id'].replace({' ':np.nan})
    raw_data = raw_data.dropna(subset=['parcel_id','comments','lat','long'])
    raw_data['comments'] = raw_data['comments'].str.lower()
    raw_data['applicant'] = raw_data['applicant'].str.lower()
    raw_data['parcel_id'] = raw_data['parcel_id'].astype('int64')
    #select intervention related sample 
    raw_data = raw_data[raw_data['comments'].str.contains(keyword_test)]
    return raw_data

def process_house_data_from_file(dataset_file):
    raw_data = pd.read_csv(dataset_file)[headname_house]
    #fill null
    raw_data = raw_data.rename(columns={'PID':'parcel_id'})
    # drop duplicate samples
    raw_data = raw_data.drop_duplicates(keep='first')
    # take the max value from 'AV_LAND','AV_BLDG','AV_TOTAL' as the value of the parcel
    raw_data['AV_COMBINE']=raw_data[['AV_LAND','AV_BLDG','AV_TOTAL']].max(axis=1)    
    # some value of parcle is 0, replace with median number
    raw_data['AV_COMBINE']=raw_data['AV_COMBINE'].replace({0:raw_data['AV_COMBINE'].median()})
    return raw_data

def connect_tables(tableA,tableB):
    return pd.merge(tableA,tableB,on='parcel_id')

def KPP_clustering(dataset,k):
    kpp = KMeans(
    n_clusters=k, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
    )
    kpp.fit(dataset)
    return kpp

def KPP_elbow_plot(dataset,k):
    """
    Draw the elbow plot for kpp
    """
    distortions = []
    for i in range(1, k):
        kpp = KPP_clustering(dataset,i)
        distortions.append(kpp.inertia_)
    plt.plot(range(1, k), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    
def kpp_plot(kpp,dataset,dataset_origin):
    dataset_df = pd.DataFrame(dataset_origin,columns=['lat','long','AV_COMBINE'])
    res = pd.DataFrame(kpp.labels_,columns=['cluster'])
    dd = pd.concat([dataset_df,res],axis=1)
    
    plt.figure(figsize=(15,15))
    num_centroids = len(kpp.cluster_centers_)
    for i in range(num_centroids):
        cnt = np.sum(kpp.labels_==i)
        meanv = int(dd[dd['cluster']==i]['AV_COMBINE'].mean())
        i_label='cluster %d' % (i+1)
        plt.scatter(
                dataset[kpp.labels_ == i, 0], dataset[kpp.labels_ == i, 1],
                c=choose_color[i], marker='o',
                s=20,
                label=i_label + ": Num="+str(cnt)+"  Mean value="+str(meanv)
                )
        
    plt.scatter(
                kpp.cluster_centers_[:, 0], kpp.cluster_centers_[:, 1],
                c='red', marker='*',
                s=200,
                label='centroid'
                )
    
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(scatterpoints=1,fontsize=15,markerscale=2)
    plt.grid()
    plt.show()
   # plt.legend(fontsize=50,markerscale=2)

    
df1 = process_intervention_data_from_file(dataset1_file)
df2 = process_house_data_from_file(dataset2_file)
df3 = connect_tables(df1,df2)
#create_heatmap_permit_intensity(df3)
#race_analysis(df3)
area_group_by_zipcode_analysis(df3)
'''
stan = preprocessing.StandardScaler().fit(df3[['lat','long']])
dataset_s_loc = stan.transform(df3[['lat','long']])   
minm = preprocessing.MinMaxScaler().fit(df3[['AV_COMBINE']])
dataset_s_p = minm.transform(df3[['AV_COMBINE']])

dataset_scaled= np.concatenate((dataset_s_loc,dataset_s_p),axis=1)
print(dataset_scaled)
#KPP_elbow_plot(df3[['lat','long','AV_COMBINE']],12)

kpp_clusters = 6
kpp = KPP_clustering(dataset_scaled,kpp_clusters)
dataset_s_loc_inversed = stan.inverse_transform(dataset_s_loc)   
dataset_s_p_inversed = minm.inverse_transform(dataset_s_p)
dataset_inversed= np.concatenate((dataset_s_loc_inversed,dataset_s_p_inversed),axis=1)
print(dataset_inversed)
kpp_plot(kpp,dataset_scaled,dataset_inversed)

#mark_on_real_map(dataset_inversed,kpp)
'''