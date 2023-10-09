
import pandas as pd
import numpy as np
import datetime
import math
from collections import Counter
from scipy.stats import skew
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate,train_test_split,StratifiedKFold,KFold
import pickle

# Extract data from CSVs
cgmData = pd.read_csv('CGMData.csv', sep=',', low_memory = False)
cgmData['dateTime'] = pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])
cgmData = cgmData.sort_values(by='dateTime',ascending=True)

insulinData = pd.read_csv('InsulinData.csv', sep=',', low_memory = False)
insulinData['dateTime'] = pd.to_datetime(insulinData['Date'] + ' ' + insulinData['Time'])
insulinData = insulinData.sort_values(by='dateTime',ascending=True)

# Compare the dateTime to identify how long have one eaten the previous meal
insulinData['New Index'] = range(0, 0+len(insulinData))
mealTimes = insulinData.loc[insulinData['BWZ Carb Input (grams)'] > 0][['New Index', 'Date', 'Time', 'BWZ Carb Input (grams)', 'dateTime']]
mealTimes['diff'] = mealTimes['dateTime'].diff(periods=1)
mealTimes['shiftUp'] = mealTimes['diff'].shift(-1)
mealTimes

# Using the previous meal time, filter out any meals eaten before the threshold (2 hours)
mealTimes = mealTimes.loc[(mealTimes['shiftUp'] > datetime.timedelta (minutes = 120)) | (pd.isnull(mealTimes['shiftUp']))]
mealTimes


# Create a new dataframe. Using the meal time data from insulindata file and filter out the relevant time. Add those rows into the new dataframe
cgmdata_withMeal = pd.DataFrame()
cgmdata_withMeal['New Index'] = ""
for i in range(len(mealTimes)) : 
    preMealTime = mealTimes['dateTime'].iloc[i] - datetime.timedelta(minutes = 30)
    endMealTime = mealTimes['dateTime'].iloc[i] + datetime.timedelta(minutes = 120)
    filteredcgmdata = cgmData.loc[(cgmData['dateTime'] >= preMealTime) & (cgmData['dateTime'] < endMealTime )]
    arr = []
    index_label = 0
    index_label = mealTimes['New Index'].iloc[i]
    for j in range(len(filteredcgmdata)) :
        arr.append(filteredcgmdata['Sensor Glucose (mg/dL)'].iloc[j])
    cgmdata_withMeal = cgmdata_withMeal.append(pd.Series(arr), ignore_index=True)
    cgmdata_withMeal.iloc[i, cgmdata_withMeal.columns.get_loc('New Index')] = index_label
cgmdata_withMeal['New Index'] = cgmdata_withMeal['New Index'].astype(int)
cgmdata_withMeal

cgmdata_withMeal_index = pd.DataFrame()
cgmdata_withMeal_index['New Index'] = cgmdata_withMeal['New Index']
cgmdata_withMeal = cgmdata_withMeal.drop(columns='New Index')

# Apply threshold for missing data and interpolation
no_of_rows= cgmdata_withMeal.shape[0]
no_of_columns = cgmdata_withMeal.shape[1]
cgmdata_withMeal.dropna(axis=0, how='all', subset=None, inplace=True)
cgmdata_withMeal.dropna(axis=0, thresh=no_of_columns/4, subset=None, inplace=True)
cgmdata_withMeal.dropna(axis=1, how='all', subset=None, inplace=True)
cgmdata_withMeal.dropna(axis=1, thresh=no_of_rows/4, subset=None, inplace=True)
cgmdata_withMeal.interpolate(axis=0, method ='linear', limit_direction ='forward', inplace=True)
cgmdata_withMeal.bfill(axis=1,inplace=True)
cgmdata_withMeal
cgmdata_withMeal_without_index = cgmdata_withMeal.copy()
mean_cgm_meal = cgmdata_withMeal.copy()

cgmdata_withMeal = pd.merge(cgmdata_withMeal, cgmdata_withMeal_index, left_index=True, right_index=True)
cgmdata_withMeal['mean CGM data'] = cgmdata_withMeal_without_index.mean(axis=1)
cgmdata_withMeal['max-start_over_start'] = cgmdata_withMeal_without_index.max(axis = 1)/cgmdata_withMeal_without_index[0]

# Extract the meal amounts from insulinData
mealAmount = mealTimes[['BWZ Carb Input (grams)', 'New Index']]
mealAmount = mealAmount.rename(columns={'BWZ Carb Input (grams)': 'Meal Amount'})
max_mealAmount = mealAmount['Meal Amount'].max()
min_mealAmount = mealAmount['Meal Amount'].min()

# Extracting Ground Truth from meal amounts
Meal_Amount_bin_label = pd.DataFrame()

def bin_label(x):
    if (x <= 23):
        return np.floor(0);
    elif (x <= 43):
        return np.floor(1);
    elif (x <= 63):
        return np.floor(2);
    elif (x <= 83):
        return np.floor(3);
    elif (x <= 103):
        return np.floor(4);
    else:
        return np.floor(5);

Meal_Amount_bin_label['Bin Label'] = mealAmount.apply(lambda row: bin_label(row['Meal Amount']).astype(np.int64), axis=1)
Meal_Amount_bin_label['New Index'] = mealAmount['New Index']
# Join Meal Data and Meal Amount
Meal_Data_and_Amount = cgmdata_withMeal.merge(Meal_Amount_bin_label, how='inner', on=['New Index'])
Meal_Data_and_Amount

mealTimesCarbInput = pd.DataFrame()
mealTimesCarbInput = mealTimes[['BWZ Carb Input (grams)', 'New Index']]
Meal_Data_and_Amount = Meal_Data_and_Amount.merge(mealTimesCarbInput, how='inner', on=['New Index'])
Meal_Data_and_Amount = Meal_Data_and_Amount.drop(columns='New Index')

New_feature_extraction = pd.DataFrame()
New_feature_extraction = Meal_Data_and_Amount[['BWZ Carb Input (grams)', 'mean CGM data']]
New_feature_extraction

# scatter plot to find if we can find any pattern of how many cluster

# Normalize DBScan data
kmeans_data = New_feature_extraction.copy()
kmeans_data = kmeans_data.values.astype('float32', copy=False)
kmeans_data_scaler = StandardScaler().fit(kmeans_data)
Feature_extraction_scaler = kmeans_data_scaler.transform(kmeans_data)

# Find the SSE at each cluster level
k_rng = range(1, 16)
sse = []
for k in k_rng:
    km_test = KMeans(n_clusters=k)
    km_test.fit(Feature_extraction_scaler)
    sse.append(km_test.inertia_)
sse

km = KMeans(n_clusters=10)
km

y_predicted = km.fit_predict(Feature_extraction_scaler)
y_predicted

KMeans_sse = km.inertia_

New_feature_extraction['cluster'] = y_predicted
New_feature_extraction.head()


km.cluster_centers_

# Ground true array
ground_true_arr = Meal_Data_and_Amount["Bin Label"].tolist()

bins_clusters_df = pd.DataFrame({'ground_true_arr': ground_true_arr, 'kmeans_labels': list(y_predicted)}, columns=['ground_true_arr', 'kmeans_labels'])

confusion_matrix = pd.pivot_table(bins_clusters_df, index='kmeans_labels', columns='ground_true_arr', aggfunc=len)
confusion_matrix.fillna(value=0,inplace=True)

confusion_matrix = confusion_matrix.reset_index()
confusion_matrix = confusion_matrix.drop(columns=['kmeans_labels'])

# KMeans Entropy
confusion_matrix_copy = confusion_matrix.copy()

def row_entropy(row):
    total = 0
    entropy = 0
    for i in range(len(confusion_matrix.columns)):
        total = total + row[i];
    for j in range(len(confusion_matrix.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j]/total*math.log2(row[j]/total)
    return -entropy
        
confusion_matrix_copy['Total'] = confusion_matrix.sum(axis=1)
confusion_matrix_copy['Row_entropy'] = confusion_matrix.apply(lambda row: row_entropy(row), axis = 1)
total_total = confusion_matrix_copy['Total'].sum()
confusion_matrix_copy['entropy_prob'] = confusion_matrix_copy['Total']/total_total*confusion_matrix_copy['Row_entropy']
KMeans_entropy = confusion_matrix_copy['entropy_prob'].sum()
# KMeans Purity

confusion_matrix_copy['Max_val'] = confusion_matrix.max(axis=1)
KMeans_purity = confusion_matrix_copy['Max_val'].sum()/total_total;

# DBScan
dbscan_data_feature = New_feature_extraction.copy()[['BWZ Carb Input (grams)', 'mean CGM data']]
dbscan_data_feature_arr = dbscan_data_feature.values.astype('float32', copy=False)
dbscan_data_feature_arr

# Normalize DBScan data with StandardScaler
dbscan_data_scaler = StandardScaler().fit(dbscan_data_feature_arr)
dbscan_data_feature_arr = dbscan_data_scaler.transform(dbscan_data_feature_arr)
dbscan_data_feature_arr

# Construct model - requires a minimum 8 data points in a neighborhood; eps in radius 0.2
# model = DBSCAN(eps = 0.2, min_samples = 8, metric = 'euclidean').fit(dbscan_data)
model = DBSCAN(eps = 0.19, min_samples = 5).fit(dbscan_data_feature_arr)
model

# Separate outliers from clustered data
outliers_df = dbscan_data_feature[model.labels_ == -1]
clusters_df = dbscan_data_feature[model.labels_ != -1]

# display('run', model.labels_)
New_feature_extraction['cluster'] = model.labels_
# display(New_feature_extraction)
colors = model.labels_
colors_clusters = colors[colors != -1]
color_outliers = 'black'

# Get info about the clusters
clusters = Counter(model.labels_)


dbscana = dbscan_data_feature.values.astype('float32', copy = False)
bins_clusters_df_dbscan = pd.DataFrame({'ground_true_arr': ground_true_arr, 'dbscan_labels': list(model.labels_)}, columns=['ground_true_arr', 'dbscan_labels'])
confusion_matrix_dbscan = pd.pivot_table(bins_clusters_df_dbscan, index='ground_true_arr', columns='dbscan_labels', aggfunc=len)
confusion_matrix_dbscan.fillna(value=0,inplace=True)
confusion_matrix_dbscan = confusion_matrix_dbscan.reset_index()
confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=['ground_true_arr'])
confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=[-1])

# DBSCANS Entropy
confusion_matrix_dbscan_copy = confusion_matrix_dbscan.copy()

def row_entropy_dbscan(row):
    total = 0
    entropy = 0
    for i in range(len(confusion_matrix_dbscan.columns)):
        total = total + row[i];
    
    for j in range(len(confusion_matrix_dbscan.columns)):
        if (row[j] == 0):
            continue;
        entropy = entropy + row[j]/total*math.log2(row[j]/total)
    return -entropy
        
confusion_matrix_dbscan_copy['Total'] = confusion_matrix_dbscan.sum(axis=1)
confusion_matrix_dbscan_copy['Row_entropy'] = confusion_matrix_dbscan.apply(lambda row: row_entropy_dbscan(row), axis = 1)
total_total = confusion_matrix_dbscan_copy['Total'].sum()
confusion_matrix_dbscan_copy['entropy_prob'] = confusion_matrix_dbscan_copy['Total']/total_total*confusion_matrix_dbscan_copy['Row_entropy']
DBScan_entropy = confusion_matrix_dbscan_copy['entropy_prob'].sum()
confusion_matrix_dbscan_copy['Max_val'] = confusion_matrix_dbscan.max(axis=1)
DBSCAN_purity = confusion_matrix_dbscan_copy['Max_val'].sum()/total_total;
New_feature_extraction = New_feature_extraction.loc[New_feature_extraction['cluster'] != -1]
dbscan_feature_extraction_centroid = New_feature_extraction.copy()
centroid_carb_input_obj = {}
centroid_cgm_mean_obj = {}
squared_error = {}
DBSCAN_SSE = 0
for i in range(len(confusion_matrix_dbscan.columns)):
    cluster_group = New_feature_extraction.loc[New_feature_extraction['cluster'] == i]
    centroid_carb_input = cluster_group['BWZ Carb Input (grams)'].mean()
    centroid_cgm_mean = cluster_group['mean CGM data'].mean()
    centroid_carb_input_obj[i] = centroid_carb_input
    centroid_cgm_mean_obj[i] = centroid_cgm_mean

def centroid_carb_input_calc(row):
    return centroid_carb_input_obj[row['cluster']]
def centroid_cgm_mean_calc(row):
    return centroid_cgm_mean_obj[row['cluster']]
dbscan_feature_extraction_centroid['centroid_carb_input'] = New_feature_extraction.apply(lambda row: centroid_carb_input_calc(row), axis=1)
dbscan_feature_extraction_centroid['centroid_cgm_mean'] = New_feature_extraction.apply(lambda row: centroid_cgm_mean_calc(row), axis=1)
dbscan_feature_extraction_centroid['centroid_difference'] = 0
for i in range(len(dbscan_feature_extraction_centroid)):
    dbscan_feature_extraction_centroid['centroid_difference'].iloc[i] = math.pow(dbscan_feature_extraction_centroid['BWZ Carb Input (grams)'].iloc[i] - dbscan_feature_extraction_centroid['centroid_carb_input'].iloc[i], 2) + math.pow(dbscan_feature_extraction_centroid['mean CGM data'].iloc[i] - dbscan_feature_extraction_centroid['centroid_cgm_mean'].iloc[i], 2)
for i in range(len(confusion_matrix_dbscan.columns)):
    squared_error[i] = dbscan_feature_extraction_centroid.loc[dbscan_feature_extraction_centroid['cluster'] == i]['centroid_difference'].sum()
for i in squared_error:
    DBSCAN_SSE = DBSCAN_SSE + squared_error[i];


KMeans_DBSCAN = [KMeans_sse, DBSCAN_SSE, KMeans_entropy, DBScan_entropy, KMeans_purity, DBSCAN_purity]
print_df = pd.DataFrame(KMeans_DBSCAN).T
print_df


print_df.to_csv('Result.csv', header=False, index=False)


