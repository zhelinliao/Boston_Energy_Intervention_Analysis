# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:32:16 2019

@author: seanl
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

abslote_path = os.path.abspath('.')
dataset1_file = abslote_path + "/../data/6ddcd912-32a0-43df-9908-63574f8c7e77.csv"
dataset2_file = abslote_path + "/../data/fy19fullpropassess.csv"
url = abslote_path + '/bzp.geojson'
bzp = f'{url}'
headname_interv =['permitnumber','comments','applicant','issued_date',
           'property_id','parcel_id','lat','long','zip','city']
headname_house =['PID','GIS_ID','AV_LAND','AV_BLDG','AV_TOTAL','GROSS_AREA','OWN_OCC']
keyword_interv = 'solar|hot water|photovoltaic|ground source|gshp|air source|heat pump|insulation|gas stove|hvac|boiler|wiring|furnace|weather stripping'
choose_color = ['purple', 'cadetblue', 'orange', 'green', 
                    'yellow', 'pink','blue']

    
def create_heatmap_value(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat','long','AV_COMBINE']], radius=9, max_zoom
        =50,name="the heapmap of the property values").add_to(base_map)
    base_map.save(abslote_path+'/../plots/heatmap_parcel_value.html')
    
def create_heatmap_permit_intensity(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat','long']], radius=9, max_val=0.5,max_zoom
        =20).add_to(base_map)
    base_map.save(abslote_path+'/../plots/heatmap_permit_intensity.html')    
    
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
    basemap.save(abslote_path+'/../plots/Map_KPP_cluster_wealthiness.html')       

def process_intervention_data_from_file(dataset_file):
    '''
    Pre-process data from file, filter out energy intervetion related samples 
    from all the building permits for construction projects within Boston
    '''
    raw_data = pd.read_csv(dataset_file)[headname_interv]
    # drop null
    raw_data['parcel_id']=raw_data['parcel_id'].replace({' ':np.nan})
    raw_data = raw_data.dropna(subset=['parcel_id','comments','lat','long'])
    raw_data['comments'] = raw_data['comments'].str.lower()
    raw_data['applicant'] = raw_data['applicant'].str.lower()
    raw_data['parcel_id'] = raw_data['parcel_id'].astype('int64')
    #select intervention related sample 
    raw_data = raw_data[raw_data['comments'].str.contains(keyword_interv)]
    return raw_data

def process_intervention_data_from_file_zipcode(dataset_file):

    raw_data = pd.read_csv(dataset_file)[headname_interv]
    # drop null
    raw_data['parcel_id']=raw_data['parcel_id'].replace({' ':np.nan})
    raw_data['zip']=raw_data['zip'].replace({' ':np.nan})
    raw_data = raw_data.dropna(subset=['parcel_id','comments','lat','long','zip'])
    raw_data['comments'] = raw_data['comments'].str.lower()
    raw_data['applicant'] = raw_data['applicant'].str.lower()
    raw_data['parcel_id'] = raw_data['parcel_id'].astype('int64')
    #select intervention related sample 
    raw_data = raw_data[raw_data['comments'].str.contains(keyword_interv)]
    raw_data['zip'] = raw_data['zip'].astype('int64')
    raw_data = raw_data[(raw_data['zip'] < 2171) & (raw_data['zip'] > 2108)]
    raw_data['zip'] = raw_data['zip'].apply(lambda x: str(x))
    return raw_data

def process_house_data_from_file(dataset_file):
    '''
    Pre process data from file, clean the data samples of parcel/building in Boston
    '''
    raw_data = pd.read_csv(dataset_file)[headname_house]
    #fill null
    raw_data = raw_data.rename(columns={'PID':'parcel_id'})
    # Drop duplicate samples
    raw_data = raw_data.drop_duplicates(['parcel_id','GIS_ID'],keep='first')  
    raw_data['AV_COMBINE'] = raw_data['AV_TOTAL']   
    return raw_data

def connect_tables(tableA,tableB):
    '''
    link two dataseta using primary key 'parcel_id'
    '''
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
    plt.title('KPP elbow plot',fontsize=15)
    plt.show()
    
def kpp_plot(kpp,dataset,dataset_origin, text):
    dataset_df = pd.DataFrame(dataset_origin,columns=['lat','long','value'])
    res = pd.DataFrame(kpp.labels_,columns=['cluster'])
    dd = pd.concat([dataset_df,res],axis=1)
    
    plt.figure(figsize=(15,15))
    num_centroids = len(kpp.cluster_centers_)
    for i in range(num_centroids):
        cnt = np.sum(kpp.labels_==i)
        meanv = int(dd[dd['cluster']==i]['value'].mean())
        i_label='cluster %d' % (i+1)
        plt.scatter(
                dataset[kpp.labels_ == i, 1], dataset[kpp.labels_ == i, 0],
                c=choose_color[i], marker='o',
                s=20,
                label=i_label + ": Num="+str(cnt)+"   Mean="+str(meanv)
                )
        
    plt.scatter(
                kpp.cluster_centers_[:, 1], kpp.cluster_centers_[:, 0],
                c='red', marker='*',
                s=200,
                label='centroid'
                )
    plt.title('KPP clustring base on location and '+text,fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Scaled Longitude',fontsize=15)
    plt.ylabel('Scaled Latitude',fontsize=15)
    leg = plt.legend(scatterpoints=1,fontsize=15)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [100]
    plt.grid()
    plt.show()

def wealth_neighborhood_analysis_KPP_clustering(data): 
    '''
    Using KPP clustering method to group energy intervetion permit, to analyze if
    energy inervention happen more often in more expensive building or wealthier 
    neighborbhood
    We use two measurement to evalute the wealthiness, one is the total value of 
    the property, another is the unit area price of the property, which we obtain 
    by (total value)/(gross floor area).
    '''
    # Omit the samples which don't have the value of the property;
    # Omit the samples which don't show the living area of the property, because 
    # we need to use value/area to evaluate the unit price of the property
    data['GROSS_AREA']=data['GROSS_AREA'].replace({0:np.nan})
    data['AV_COMBINE']=data['AV_COMBINE'].replace({0:np.nan})
    data = data.dropna(subset=['AV_COMBINE','GROSS_AREA'])
    data['UNIT_PRICE']=data['AV_COMBINE']/data['GROSS_AREA']
    # Scale latitude and longtitude
    scaler = preprocessing.StandardScaler().fit(data[['lat','long']])
    dataset_s_loc = scaler.transform(data[['lat','long']])   
    # Scale total price of property
    scaler_total_p = preprocessing.MinMaxScaler().fit(data[['AV_COMBINE']])
    dataset_s_tp = scaler_total_p.transform(data[['AV_COMBINE']])
    # Scale unit price of property
    scaler_unit_p = preprocessing.MinMaxScaler().fit(data[['UNIT_PRICE']])
    dataset_s_up = scaler_unit_p.transform(data[['UNIT_PRICE']])
    # concatenate table then do clustering separately
    dataset_scaled_total = np.concatenate((dataset_s_loc,dataset_s_tp),axis=1)
    dataset_scaled_unit = np.concatenate((dataset_s_loc,dataset_s_up),axis=1)
    # Plot KPP elbow
    KPP_elbow_plot(dataset_scaled_total,12)
    kpp_clusters = 6
    # KPP clustring using total price of property and location, then plot the clusters
    kpp_total = KPP_clustering(dataset_scaled_total,kpp_clusters)
    dataset_s_loc_inversed = scaler.inverse_transform(dataset_s_loc)   
    dataset_s_tp_inversed = scaler_total_p.inverse_transform(dataset_s_tp)
    dataset_inversed_total= np.concatenate((dataset_s_loc_inversed,dataset_s_tp_inversed),axis=1)
    kpp_plot(kpp_total,dataset_scaled_total,dataset_inversed_total,'total price of the property')
    # KPP clustring using unit price of property and location, then plot the clusters
    kpp_unit = KPP_clustering(dataset_scaled_unit,kpp_clusters)   
    dataset_s_up_inversed = scaler_unit_p.inverse_transform(dataset_s_up)
    dataset_inversed_unit= np.concatenate((dataset_s_loc_inversed,dataset_s_up_inversed),axis=1)
    kpp_plot(kpp_unit,dataset_scaled_unit,dataset_inversed_unit,'unit area price in GFA(Gross Floor Area)')
    # Mark the result of clustering in real map
    mark_on_real_map(dataset_inversed_total,kpp_total)
    create_heatmap_permit_intensity(data)
    create_heatmap_value(data)

# Calculate the average AV_TOTAL of an area based on zip code
def calculate_average(dataset):
    group = dataset.groupby(['zip']).mean()
    group.reset_index(level=0, inplace=True)
    group = group.rename(columns={'AV_COMBINE': 'average_price'})
    return group[['zip', 'average_price']]


# Generate a heat map based on the latitude and longitude, the result is not
# clear, so try to use choropleth map
def draw_heat_map(dataset):
    base_map = generateBaseMap()
    HeatMap(data=dataset[['lat', 'long', 'AV_TOTAL']].
    groupby(['lat', 'long']).mean().
        reset_index().values.tolist(), radius = 8, max_zoom
    = 13).add_to(base_map)
    base_map.save('heatmap.html')

# Draw a scatter picture on a map to figure out whether the area is strictly cut
# based on zip code
def draw_scatter_on_map(data_set):
    color = ['red', 'yellow', 'blue', 'green', 'skyblue', 'purple']
    incidents = folium.map.FeatureGroup()
    for lat, lng, zipcode in zip(data_set['lat'], data_set['long'], data_set['zip']):
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=2,  # define how big you want the circle markers to be
                fill=True,
                fill_color=color[int(zipcode) % 5],
                color=color[int(zipcode) % 5],
                fill_opacity=0.4
            )
        )
    B_map = generateBaseMap()
    B_map.add_child(incidents)
    B_map.save('scatter.html')


def draw_choropleth_map(dataset):
    # Draw the choropleth map according to the average price of AV_TOTAL based on zip code
    m = folium.Map(location=[42.37909, -71.03215], zoom_start=12)
    bins = list(dataset['average_price'].quantile([0, 0.25, 0.5, 0.75, 1]))
    print(dataset.info())
    dataset = dataset.dropna()
    folium.Choropleth(
        geo_data=bzp,
        data=dataset,
        columns=['zipcode', 'average_price'],
        key_on='feature.properties.ZCTA5CE10',
        # fill_color='red',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        highlight=True,
        bins = bins,
        legend_name='Boston Average House Price'
    ).add_to(m)
    m.save(abslote_path+'/../plots/choropleth.html')
    
def wealth_neighborhood_analysis_by_zip():
    df1 = process_intervention_data_from_file_zipcode(dataset1_file)
    df2 = process_house_data_from_file(dataset2_file)
    df3 = connect_tables(df1,df2)
    df3['AV_COMBINE']=df3['AV_COMBINE'].replace({0:np.nan})
    df3 = df3.dropna(subset=['AV_COMBINE'])
    average_price = calculate_average(df3)
    zipcode = average_price['zip'].apply(lambda x: '0' + x)
    average_price['zipcode'] = zipcode
    average_price['average_price'] = average_price['average_price'].apply(lambda x:int(x))
    draw_choropleth_map(average_price)    
    
df1 = process_intervention_data_from_file(dataset1_file)
df2 = process_house_data_from_file(dataset2_file)
df3 = connect_tables(df1,df2)

wealth_neighborhood_analysis_by_zip()
wealth_neighborhood_analysis_KPP_clustering(df3)



