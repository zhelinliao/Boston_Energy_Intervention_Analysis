# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

abslote_path = os.path.abspath('.')
dataset1_file = abslote_path + "/../data/6ddcd912-32a0-43df-9908-63574f8c7e77.csv"
dataset2_file = abslote_path + "/../data/fy19fullpropassess.csv"

headname_interv =['parcel_id','declared_valuation', 'total_fees','comments','issued_date']
headname_house =['PID','GIS_ID', 'YR_BUILT', 'YR_REMOD']
keyword_interv = 'solar|hot water|photovoltaic|ground source|gshp|air source|heat pump|insulation|gas stove|hvac|boiler|wiring|furnace|weather stripping'
choose_color = ['purple', 'cadetblue', 'orange', 'green', 
                    'yellow', 'pink','blue']
pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)



def process_intervention_data_from_file(dataset_file):
    '''
    Pre-process data from file, filter out energy intervetion related samples 
    from all the building permits for construction projects within Boston
    '''
    raw_data = pd.read_csv(dataset_file)[headname_interv]
    # drop null and clean data
    raw_data['parcel_id']=raw_data['parcel_id'].replace({' ':np.nan})
    raw_data[['total_fees', 'declared_valuation']] = raw_data[['total_fees', 'declared_valuation']].replace(0, np.nan)
    raw_data = raw_data.dropna(subset=['parcel_id','total_fees','declared_valuation','comments'])
    raw_data['comments'] = raw_data['comments'].str.lower()
    # put string type to int type
    raw_data['declared_valuation'] = raw_data['declared_valuation'].astype('int32')
    raw_data['total_fees'] = raw_data['total_fees'].astype('int32')
    # there are some negative number, put them to positive number
    raw_data['declared_valuation'] = raw_data['declared_valuation'].abs()
    raw_data['total_fees'] = raw_data['total_fees'].abs()
    raw_data['parcel_id'] = raw_data['parcel_id'].astype('int64')
    #select intervention related sample 
    raw_data = raw_data[raw_data['comments'].str.contains(keyword_interv)]
    return raw_data

def process_house_data_from_file(dataset_file):
    '''
    Pre process data from file, clean the data samples of parcel/building in Boston
    '''
    raw_data = pd.read_csv(dataset_file)[headname_house]
    raw_data = raw_data.rename(columns={'PID':'parcel_id'})
    # Drop duplicate samples
    raw_data = raw_data.drop_duplicates(['parcel_id','GIS_ID'],keep='first')
    # drop null
    raw_data['YR_BUILT'].fillna(0, inplace=True)
    raw_data['YR_BUILT'] = raw_data['YR_BUILT'].replace(0, np.nan)
    raw_data = raw_data.dropna(subset=['parcel_id','GIS_ID','YR_BUILT'])
    # inplace the nan value in YR_REMOD as YR_BUILT
    raw_data['YR_REMOD'].fillna(0, inplace=True)
    raw_data['YR_REMOD'] = raw_data[['YR_REMOD','YR_BUILT']].max(axis=1)
    return raw_data

def connect_tables(tableA,tableB):
    '''
    link two dataseta using primary key 'parcel_id'
    '''
    return pd.merge(tableA,tableB,on='parcel_id')

def calculate_number_of_building_by_built_year(house_data):
    group = house_data.groupby(['YR_BUILT']).count()
    group.reset_index(level=0, inplace=True)
    plt.plot(group['parcel_id'], group['YR_BUILT'])
    plt.title('The number of energy intervention permit and built year of its corresponding building')
    plt.xlabel('Number of permits')
    plt.ylabel('The built year of building')
    plt.show()

def calculate_number_of_building_by_remodeled_year(house_data):
    group = house_data.groupby(['YR_REMOD']).count()
    group.reset_index(level=0, inplace=True)
    plt.plot(group['parcel_id'], group['YR_REMOD'])
    plt.title('The number of energy intervention permit and remodeled year of its corresponding building')
    plt.xlabel('Number of permits')
    plt.ylabel('The remodeled year of building')
    plt.show()
    
def calculate_declared_valuation(total_data_set):
    # extract YR_BUILT and declared_valuation to analysis data
    pre_group = total_data_set[['YR_BUILT', 'declared_valuation']]
    # calculate the mean of declared_valuation for each year
    group = pre_group.groupby(['YR_BUILT']).mean()
    group.reset_index(level=0, inplace=True)
    print(group)
    plt.plot(group['declared_valuation'], group['YR_BUILT'])
    plt.show()

def calculate_total_fees(total_data_set):
    # extract YR_BUILT and total_fees to analysis data
    pre_group = total_data_set[['YR_BUILT', 'total_fees']]
    # calculate the mean of total_fees for each year
    group = pre_group.groupby(['YR_BUILT']).mean()
    group.reset_index(level=0, inplace=True)
    print(group)
    plt.plot(group['total_fees'], group['YR_BUILT'])
    plt.show()

def calculate_number_of_permit_by_issue_year(data):
    year = []
    for index, row in data.iterrows():
        time = row['issued_date'].split(" ")
        date = time[0].split("-")
        year.append(date[0])
    data['permit_year'] = year
    group = data.groupby(['permit_year']).count()
    group.reset_index(level=0, inplace=True)
    plt.plot(group['permit_year'],group['parcel_id'])
    plt.title('The number of energy intervention permit in recent years')
    plt.xlabel('Year')
    plt.ylabel('The number of energy intervention permit')
    plt.show()
    
def GMM_clustering(dataset,k):
    gmm = GaussianMixture(n_components = k)
    label = gmm.fit_predict(dataset)
    return label 

def KPP_clustering(dataset,k):
    kpp = KMeans(
    n_clusters=k, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
    )
    label = kpp.fit_predict(dataset)
    return label

def clustering_plot(label_, dataset_origin, num_clusters, clusterType):
    num_centroids = num_clusters
    plt.figure(figsize=(10,10))
    for i in range(num_centroids):
        cnt = np.sum(label_==i)
        i_label='cluster %d' % (i+1)
        plt.scatter(
                dataset_origin[label_ == i, 0], dataset_origin[label_ == i, 1],
                c=choose_color[i], marker='o',
                s=2,
                label=i_label+ ": Num="+str(cnt)
                )
    plt.title('Energy Intervention related buildings clustered by '+ clusterType,fontsize=25)
    plt.xlabel('Built year',fontsize=25)
    plt.ylabel('Remodeled year',fontsize=25 )
    leg = plt.legend(scatterpoints=1,fontsize=25)
    for lh in leg.legendHandles: 
        lh._sizes = [100]
    plt.grid()
    
def clustring_energy_interv_builing_by_built_and_remodel_year(data):
    scaler = preprocessing.StandardScaler().fit(data[['YR_BUILT','YR_REMOD']])
    dataset_s_loc = scaler.transform(data[['YR_BUILT','YR_REMOD']]) 
    dataset_i_loc = scaler.inverse_transform(dataset_s_loc)
    cluster_num = 3
    #GMM clustering
    label_gmm = GMM_clustering(dataset_s_loc,cluster_num)
    clustering_plot(label_gmm,dataset_i_loc,cluster_num,'GMM')
    #KPP clustering
    label_kpp = KPP_clustering(dataset_s_loc,cluster_num)
    clustering_plot(label_kpp,dataset_i_loc,cluster_num,'KPP')
    
df1 = process_intervention_data_from_file(dataset1_file)
df2 = house_data = process_house_data_from_file(dataset2_file)
df3 = connect_tables(df1,df2)
calculate_number_of_permit_by_issue_year(df3)
calculate_number_of_building_by_built_year(df3)
calculate_number_of_building_by_remodeled_year(df3)
clustring_energy_interv_builing_by_built_and_remodel_year(df3)

