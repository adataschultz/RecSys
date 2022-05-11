# -*- coding: utf-8 -*-
"""
@author: aschu
"""

import os
import random
import numpy as np
import sys
import pandas as pd
import time
import json
from surprise import Dataset, Reader
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import KNNBaseline, KNNWithMeans, KNNBasic, KNNWithZScore
from surprise import SVD, SVDpp
from surprise import NMF
from surprise import CoClustering
from surprise import dump
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection import GridSearchCV
from scipy.sparse.linalg import svds

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set seed 
seed_value = 42
os.environ['Recommender'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Set path
path = r'D:\AmazonReviews\Data'
os.chdir(path)

# Read data
df = pd.read_csv('Clothing_Shoes_and_Jewelry.csv', header=None, skiprows=[0],
                 low_memory=False)
df = df.drop_duplicates()

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('AmazonReviews_RecSys_EDA.txt', 'w')

print('\nAmazon Reviews EDA: Clothing_Shoes_and_Jewelry') 
print('======================================================================')

df.columns = ['reviewerID','item','rating','timestamp']

def data_summary(df):
    print('Number of Rows: {}, Columns: {}'.format(df.shape[0] ,df.shape[1]))
    a = pd.DataFrame()
    a['Number of Unique Values'] = df.nunique()
    a['Number of Missing Values'] = df.isnull().sum()
    a['Data type of variable'] = df.dtypes
    print(a)

print('\nInitial Data Summary')     
print(data_summary(df))
print('======================================================================')

# Remove timestamp column
df = df.drop(['timestamp'], axis = 1) 

# Number of unique reviewer id and product id in the data
print('Number of unique reviewers in initial set ', df['reviewerID'].nunique())
print('Number of unique items in initial set ', df['item'].nunique())
print('======================================================================')

# Examine reviewers
reviewers_top20 = df.groupby('reviewerID').size().sort_values(ascending=False)[:20]
print('Reviewers with highest number of ratings in initial set:')
print(reviewers_top20)
print('======================================================================')

del reviewers_top20

# Examine items
items_top20 = df.groupby('item').size().sort_values(ascending=False)[:20]
print('Items with highest number of ratings in initial set:')
print(items_top20)
print('======================================================================')

del items_top20

###############################################################################
# Create new integer id for item due to sparsity
value_counts = df['item'].value_counts(dropna=True, sort=True)
df1 = pd.DataFrame(value_counts)
df1 = df1.reset_index()
df1.columns = ['item_unique', 'counts'] # change column 
df1 = df1.reset_index()
df1.rename(columns={'index': 'item_id'}, inplace=True)

df1 = df1.drop(['counts'], axis=1)

df = pd.merge(df, df1, how='left', left_on=['item'], 
               right_on=['item_unique'])
df = df.drop_duplicates()

del value_counts, df1

df = df.drop(['item_unique'], axis=1)

###############################################################################
# Create new integer id for reviewerID due to sparsity
value_counts = df['reviewerID'].value_counts(dropna=True, sort=True)
df1 = pd.DataFrame(value_counts)
df1 = df1.reset_index()
df1.columns = ['id_unique', 'counts'] 
df1 = df1.reset_index()
df1.rename(columns={'index': 'reviewer_id'}, inplace=True)

df1 = df1.drop(['counts'], axis=1)

df = pd.merge(df, df1, how='left', left_on=['reviewerID'], 
               right_on=['id_unique'])
df = df.drop_duplicates()

del value_counts, df1

df = df.drop(['id_unique'], axis=1)

# Create key for merging new integer id variables for later join
df1 = df[['item', 'item_id', 'reviewerID','reviewer_id']]
df1.to_csv('Clothing_Shoes_and_Jewelry_idMatch.csv', index = False)

del df1

# Drop unnecessary keys
df = df.drop(['item', 'reviewerID'], axis=1)

###############################################################################
# Filter to greater than or equal to 1500 due to sparsity
reviewer_count = df.reviewer_id.value_counts()
df = df[df.reviewer_id.isin(reviewer_count[reviewer_count >= 1500].index)]
df = df.drop_duplicates()

del reviewer_count

print('Number of reviewers with 1500 or more ratings: ', len(df))
print('Number of unique reviewers: ', df['reviewer_id'].nunique())
print('Number of unique items: ', df['item_id'].nunique())
print('\n')

# Count reviewers based on rating
for i in range(1,6):
  print('Number of reviewers who rated {0} rating = {1}'.format(i,
                                                               df[df['rating'] == i].shape[0]))
print('======================================================================')

# Examine reviewers
reviewers_top20 = df.groupby('reviewer_id').size().sort_values(ascending=False)[:20]
print('Reviewers with highest number of ratings in filtered set:')
print(reviewers_top20)
print('======================================================================')

del reviewers_top20

# Examine items
items_top20 = df.groupby('item_id').size().sort_values(ascending=False)[:20]
print('Items with highest number of ratings filtered set:')
print(items_top20)
print('======================================================================')

del items_top20

# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin

###############################################################################
# RecSys Methods
# Set path for results
path = r'D:\AmazonReviews\Models'
os.chdir(path)

# Write results to log file
stdoutOrigin=sys.stdout 
sys.stdout = open('AmazonReviews_RecSys_Methods.txt', 'w')

print('\nAmazon Reviews: Recommender Systems Methods') 
print('======================================================================')

print('\nCreate Recommendation Systems using Surprise:')
print('\n')

# Load data using reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewer_id', 'item_id', 'rating']], reader)

# Iterate over all algorithms
print('Time for iterating through different algorithms..')
search_time_start = time.time()
benchmark = []
for algorithm in [BaselineOnly(), KNNBaseline(), KNNBasic(), KNNWithMeans(), 
                  KNNWithZScore(), CoClustering(), SVD(), SVDpp(), NMF(), 
                  NormalPredictor()]:
    # Cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, 
                             verbose=False, n_jobs=-1)
    
    # Model results
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],
                               index=['Algorithm']))
    benchmark.append(tmp)
print('Finished iterating through different algorithms:',
      time.time() - search_time_start)
print('\n')

# Create df with results   
surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    

print('Results from testing different algorithms:')
print(surprise_results)
print('======================================================================')

# Save df
surprise_results.to_csv('results_algorithms.csv')

del surprise_results

###############################################################################
# Use SVDpp with lowest rmse to train/predict 
print('Train/predict using SVDpp default parameters for 3 epochs:')
print('\n')

print('Time for iterating through SVDpp default parameters..')
search_time_start = time.time()
algo = SVDpp(n_epochs=3, random_state=seed_value)
cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False, 
                    n_jobs=-1)
print('Finished iterating through SVDpp default parameters:',
      time.time() - search_time_start)
print('\n')

print('Cross validation results:')
# Iterate over key/value pairs in cv results dict 
for key, value in cv.items():
    print(key, ' : ', value)
print('\n')

# Partition data for train/test sets
train, test = train_test_split(data, test_size=0.2)

# Fit and predict on best SVD model
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./SVDpp_3epochs_DefaultParamModel_file', predictions, algo)
# predictions, algo = dump.load('./SVDpp_3epochs_DefaultParamModel_file')

# Examine results from predictions
def get_Ir(reviewerID):
    '''Determine the number of items rated by given reviewer
    Args: 
      reviewerID: the id of the reviewer
    Returns: 
      Number of items rated by the reviewer
    '''
    try:
        return len(train.ur[train.to_inner_uid(reviewerID)])
    except ValueError: # If the reviewer was not in train set
        return 0
    
def get_Ri(itemID):
    ''' Determine number of reviewers that rated given item
    Args:
      itemID: the id of the item
    Returns:
     Number of reviewers that have rated the item
    '''
    try: 
        return len(train.ir[train.to_inner_iid(itemID)])
    except ValueError:
        return 0

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])

# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_SVDpp_DefaultParamModel.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del predictions, df1, best_predictions, worst_predictions

###############################################################################
# HPO using grid search
print('SVDpp HPO using Grid Search:')
print('\n')

# Define parameters for grid search       
param_grid = {'n_epochs': [5, 10], 
              'n_factors': [20, 25, 30, 35, 40 ,45 , 50], 
              'lr_all': [0.002, 0.003, 0.004, 0.005, 0.006, 0.007], 
              'reg_all': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04],
              'random_state': [seed_value]
              }

# Print nested dictionary line by line
print('Grid search parameters:')
print(json.dumps(param_grid, indent=4))
print('\n')
 
# Run grid search
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3,
                  n_jobs=-1)
print('Time for iterating grid search parameters..')
search_time_start = time.time()
gs.fit(data)
print('Finished iterating grid search parameters:',
      time.time() - search_time_start)
print('\n')

# Lowest RMSE score
print('Lowest RMSE from Grid Search:')
print(gs.best_score['rmse'])
print('\n')
# Parameters with the lowest RMSE 
print('Parameters of Model with lowest RMSE from Grid Search:')
print(gs.best_params['rmse'])
print('\n')

# Save results to df
results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df = results_df.sort_values('mean_test_rmse', ascending=True)
print('SVDpp GridSearch HPO Cross Validation Results:')
print(results_df.head())
print('\n')
results_df.to_csv('SVDpp_gridSearch_cvResults.csv', index=False)

del results_df

# Fit model with the lowest rmse
algo = gs.best_estimator['rmse']

# Fit and predict on best SVD model
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./SVDpp_bestGrid_Model_file', predictions, algo)
# predictions, algo = dump.load('./SVDpp_bestGrid_Model_file')

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])

# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_SVDpp_gridSearch.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del predictions, df1, best_predictions, worst_predictions

###############################################################################
# Use KNNBaseline with second lowest rmse to train/predict use Alternating Least Squares
print('Train/predict using KNNBaseline default parameters with Alternating Least Squares for 3 epochs:')
print('\n')

bsl_options = {'method': 'als',
               'n_epochs': 3
               }

# Print nested dictionary line by line
print('Baselines estimates configuration:')
print(json.dumps(bsl_options, indent=4))
print('\n')

print('Model parameters:')
print('KNNBaseline(k=40, min_k=1, bsl_options=bsl_options, verbose=False)') 
print('\n')

# Iterate over all algorithms
print('Time for iterating through KNNBaseline default parameters epochs=3 using ALS..')
search_time_start = time.time()
algo = KNNBaseline(bsl_options=bsl_options)
cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False, 
                    n_jobs=-1)
print('Finished iterating through KNNBaseline default parameters epochs=3 using ALS:',
      time.time() - search_time_start)
print('\n')

print('Cross validation results:')
# Iterate over key/value pairs in cv results dict 
for key, value in cv.items():
    print(key, ' : ', value)
print('\n')

# Partition data for train/test sets
train, test = train_test_split(data, test_size=0.2, random_state=seed_value)

# Fit and predict on best model with the lowest rmse
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./KNNBaseline_3epochs_DefaultParamModel_file', predictions, algo)
# predictions, algo = dump.load('./KNNBaseline_3epochs_DefaultParamModel_file')

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])
# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_KNNBaseline_DefaultParamModel.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del predictions, df1, best_predictions, worst_predictions

###############################################################################
# HPO using grid search
print('KNNBaseline HPO using Grid Search:')
print('\n')

# Define parameters for grid search
param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                              'n_epochs': [30, 35, 40, 45, 50, 55, 60, 65, 70]}, 
              'k': [30, 35, 40, 45, 50, 55, 60], 
              'min_k': [1, 5, 10, 15, 20, 25],
              'random_state': [seed_value],
              'sim_options': {'name': ['pearson_baseline'],
                              'min_support': [1, 5, 10],
                              'shrinkage': [0, 100]}
              }

# Print nested dictionary line by line
print('Grid search parameters:')
print(json.dumps(param_grid, indent=4))
print('\n')
                
# Run grid search
gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3,
                  joblib_verbose=0, n_jobs=-1)
print('Time for iterating grid search parameters..')
search_time_start = time.time()
gs.fit(data)
print('Finished iterating grid search parameters:',
      time.time() - search_time_start)

print('Model with lowest RMSE:')
print(gs.best_score['rmse'])

# Parameters with the lowest RMSE 
print('Parameters with the lowest RMSE:')
print(gs.best_params['rmse'])
print('======================================================================')

# Fit and predict on best model with the lowest rmse
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save results to df
results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df = results_df.sort_values('mean_test_rmse', ascending=True)
print('KNNBaseline GridSearch HPO Cross Validation Results:')
print(results_df)
results_df.csv('KNNBaseline_gridSearch_cvResults.csv', index=False)

del results_df

# Fit model with the lowest rmse
algo = gs.best_estimator['rmse']

# Fit and predict on best SVD model
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./KNNBaseline_bestGrid_Model_file', predictions, algo)
# predictions, algo = dump.load('./KNNBaseline_bestGrid_Model_file')

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])

# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_KNNBaseline_gridSearch.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del predictions, df1, best_predictions, worst_predictions

###############################################################################
# Use SVD to train/predict 
print('Train/predict using SVD default parameters for 3 epochs:')
print('\n')

print('Time for iterating through SVD default parameters..')
search_time_start = time.time()
algo = SVD(n_epochs=3, random_state=seed_value)
cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False, 
                    n_jobs=-1)
print('Finished iterating through SVD default parameters:',
      time.time() - search_time_start)
print('\n')

print('Cross validation results:')
# Iterate over key/value pairs in cv results dict 
for key, value in cv.items():
    print(key, ' : ', value)
print('\n')

# Fit and predict on best SVD model
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./SVD_3epochs_DefaultParamModel_file', predictions, algo)
# predictions, algo = dump.load('./SVD_3epochs_DefaultParamModel_file')

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])

# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_SVD_DefaultParamModel.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del predictions, df1, best_predictions, worst_predictions

###############################################################################
# HPO using grid search
print('SVD HPO using Grid Search:')
print('\n')

# Define parameters for grid search       
param_grid = {'n_epochs': [30, 35, 40, 45, 50, 55, 60, 65, 70], 
              'n_factors': [20, 25, 30, 35, 40 ,45 , 50], 
              'lr_all': [0.002, 0.003, 0.004, 0.005, 0.006, 0.007], 
              'reg_all': [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04],
              'random_state': [seed_value]
              }

# Print nested dictionary line by line
print('Grid search parameters:')
print(json.dumps(param_grid, indent=4))
print('\n')
 
# Run grid search
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3,
                  n_jobs=-1)
print('Time for iterating grid search parameters..')
search_time_start = time.time()
gs.fit(data)
print('Finished iterating grid search parameters:',
      time.time() - search_time_start)
print('\n')

# Lowest RMSE score
print('Lowest RMSE from Grid Search:')
print(gs.best_score['rmse'])
print('\n')
# Parameters with the lowest RMSE 
print('Parameters of Model with lowest RMSE from Grid Search:')
print(gs.best_params['rmse'])
print('\n')

# Save results to df
results_df = pd.DataFrame.from_dict(gs.cv_results)
results_df = results_df.sort_values('mean_test_rmse', ascending=True)
print('SVD GridSearch HPO Cross Validation Results:')
print(results_df.head())
print('\n')
results_df.to_csv('SVD_gridSearch_cvResults.csv', index=False)

del results_df

# Fit model with the lowest rmse
algo = gs.best_estimator['rmse']

# Fit and predict on best SVD model
predictions = algo.fit(train).test(test)
print('RMSE from fit best parameters on train predict on test:')
print(accuracy.rmse(predictions))
print('\n')

# Save predictions and algorithm
# dump.dump('./SVD_bestGrid_Model_file', predictions, algo)
# predictions, algo = dump.load('./SVD_bestGrid_Model_file')

# Make df of prediction results    
df1 = pd.DataFrame(predictions, columns=['reviewerID', 'itemID', 'rui', 'est',
                                         'details'])

# Apply functions
df1['Iu'] = df1.reviewerID.apply(get_Ir)
df1['Ui'] = df1.itemID.apply(get_Ri)
df1['err'] = abs(df1.est - df1.rui)

# Save prediction results    
df1.to_csv('predictions_SVD_gridSearch.csv')

# Find best/worst predictions
best_predictions = df1.sort_values(by='err')[:10]
worst_predictions = df1.sort_values(by='err')[-10:]

print('Best 10 predictions:')
print(best_predictions)
print('\n')

print('Worst 10 predictions:')
print(worst_predictions)
print('======================================================================')

del data, train, test, predictions, df1, best_predictions, worst_predictions

###############################################################################
print('\nCreate the rating matrix with items and reviewers for other RecSys Methods:')
print('\n')
ratingsMat = pd.pivot_table(df, index=['reviewer_id'], columns = 'item_id', 
                            values = 'rating').fillna(0)

print('Ratings matrix information: ', ratingsMat.info())
print('\n')
print('Dimensions of rating matrix: ', ratingsMat.shape)
print('\n')

rating_nonZero = np.count_nonzero(ratingsMat)
print('Number of non zero ratings: ', rating_nonZero)

rating_possible = ratingsMat.shape[0] * ratingsMat.shape[1]
print('Number of possible ratings: ', rating_possible)
print('\n')

density = (rating_nonZero/rating_possible) *100
print ('Density of rating matrix: {:4.3f}%'.format(density))
print('======================================================================')

###############################################################################
# Use train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Create train/test sets for modeling
train, test = train_test_split(df, test_size = 0.2, random_state=seed_value)

print('Dimensions of train set: ', train.shape)
print('Dimensions of test set: ', test.shape)
print('\n')
print('Count of different ratings in train set:')
print(train[['rating']].value_counts())
print('\n')
print('Count of different ratings in test set:')
print(test[['rating']].value_counts())
print('======================================================================')

###############################################################################
print('\nCreate SVD Based Recommendation System using SciPy:')
print('\n')

# Define reviewer index 
ratingsMat['reviewer_index'] = np.arange(0, ratingsMat.shape[0], 1)
ratingsMat.set_index(['reviewer_index'], inplace=True)

# Define parameters
U, sigma, Vt = svds(ratingsMat, k = 6)

# Construct a diagonal matrix in SVD
sigma = np.diag(sigma)

# Predicted rating
reviewers_predRating = np.dot(np.dot(U, sigma), Vt) 

ratingPred = pd.DataFrame(reviewers_predRating, columns = ratingsMat.columns)

# Recommend the items with the highest predicted rating
def recommend_items(reviewerID, ratingsMat, ratingPred, num_recommendations):
    reviewer_idx = reviewerID-1
    
    # Get and sort the reviewer's rating
    sorted_reviewer_rating = ratingsMat.iloc[reviewer_idx].sort_values(ascending=False)
    sorted_reviewer_predictions = ratingPred.iloc[reviewer_idx].sort_values(ascending=False)

    # Concatenate rating with predicted rating
    tmp = pd.concat([sorted_reviewer_rating, sorted_reviewer_predictions], 
                     axis=1)
    tmp.index.name = 'Recommended Items'
    tmp.columns = ['reviewer_rating', 'reviewer_predictions']
    tmp = tmp.loc[tmp.reviewer_rating == 0]
    tmp = tmp.sort_values('reviewer_predictions', ascending=False)
    
    print('\nBelow are the recommended items for reviewer(reviewer_id = {}):\n'.format(reviewerID))
    print(tmp.head(num_recommendations))

reviewerID = 1
num_recommendations = 10
recommend_items(reviewerID, ratingsMat, ratingPred, num_recommendations)

reviewerID = 2
num_recommendations = 10
recommend_items(reviewerID, ratingsMat, ratingPred, num_recommendations)

reviewerID = 3
num_recommendations = 10
recommend_items(reviewerID, ratingsMat, ratingPred, num_recommendations)
print('======================================================================')

print('\nEvaluate the SciPy SVD Collaborative recommender model:')
rmse_df = pd.concat([ratingsMat.mean(), ratingPred.mean()], axis=1)
rmse_df.columns = ['Avg_actual_rating', 'Avg_predicted_rating']
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)

RMSE = round((((rmse_df.Avg_actual_rating - rmse_df.Avg_predicted_rating) ** 2).mean() ** 0.5), 10)
print('\nRMSE of SciPy SVD Model = {} \n'.format(RMSE))
print('======================================================================')

###############################################################################    
print('\nCreate Popularity Recommender Model:')
print('\n')

# Count of each reviewer for each unique item as recommendation score 
train_grouped = train.groupby('item_id').agg({'reviewer_id': 'count'}).reset_index()
train_grouped.rename(columns = {'reviewer_id': 'rating'}, inplace=True)

# Sort the products on recommendation score 
train_sort = train_grouped.sort_values(['rating', 'item_id'], 
                                       ascending = [0,1]) 
      
# Generate a recommendation rank based by scoring 
train_sort['rank'] = train_sort['rating'].rank(ascending=0, method='first') 
 
# Get the top 5 recommendations 
popularity_recommendations = train_sort.head() 
print('\nTop 5 recommendations ')
print(popularity_recommendations)
print('\n')

# Use popularity based recommender model to make predictions
def recommend(reviewer_id):     
    reviewer_recommendations = popularity_recommendations 
          
    # Add reviewer_id column for which the recommendations are being generated 
    reviewer_recommendations['reviewer_id'] = reviewer_id 
      
    # Bring reviewer_id column to the first column
    cols = reviewer_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    reviewer_recommendations = reviewer_recommendations[cols] 
          
    return reviewer_recommendations 

find_recom = [1,100,200]   
for i in find_recom:
    print('The list of recommendations for the reviewer_id: %d\n' %(i))
    print(recommend(i))    
print('======================================================================')

###############################################################################    
# Close to create log file
sys.stdout.close()
sys.stdout=stdoutOrigin