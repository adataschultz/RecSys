# Amazon Review Recommendation System

## Summary
In this project, the user ratings from a subset of Amazon Reviews are used train a collaborative-filtering recommendation system by evaluating various algorithms with hyperparameter optimization. The models return a list of recommended items based on the reviewer's previous actions. 

## Data
The `Clothing_Shoes_and_Jewelry` ratings data was retrieved from https://jmcauley.ucsd.edu/data/amazon/. This includes (item, user, rating, timestamp) tuples. A subset of the data was used in the analysis due to constraints of sparcity for computation.

## Modelling

### Model Training Process
Models were trained using:
- `surprise` library
- `scipy` library
- A created popularity recommender model 

For the initial training of the models using `surprise`:
- The default parameters of `SVD`, `SVDpp` `BaselineOnly`, `KNNBaseline`, `KNNBasic`, `KNNWithMeans`, `KNNWithZScore`, `CoClustering`, `SVD`, `SVDpp`, `NMF`, `NormalPredictor` were evaluated using the `cross_validate` method to determine which algorithm yielded the lowest RMSE error. Cross validation using a model with arbitrary parameters was performed and predictions made.
- Then hyperparameter optimization was performed to find the best parameters for this model type.

For the training of the SVD based models using `SciPy`:
- A rating matrix with items and reviewers was constructed.
- The parameters for the model were `U, sigma, Vt = svds(ratingsMat, k = 6)`.
- A diagonal matrix was constructed in SVD.
- Then the ratings were predicted and RMSE was calculated. 

For the construction of the popularity recommender model:
- A recommendation score was created by counting each reviewer for each unique item.
- This score was sorted and a recommendation rank was created based on scoring.
- Predictions were then calculated for various reviewers.