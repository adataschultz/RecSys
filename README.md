# `Amazon Reviews Recommendation System`

## `Summary`
In this project, the user ratings from a subset of Amazon Reviews are used to train various recommendation systems by evaluating various algorithms with hyperparameter optimization. The models return a list of recommended items based on the reviewer's previous actions. 

## `Data`
The `Movies_and_TV` ratings data was retrieved from [here](https://jmcauley.ucsd.edu/data/amazon/). This includes (`item`, `user`, `rating`, `timestamp`) tuples. A subset of the data was used in the analysis due to computation constraints due to sparcity.

## `Modeling`

### `Model Training Process`
Models were trained using:
- `surprise` library
- A ranking model using `tensorflow-recommenders`
- Collaborative filtering models using `tensorflow`
- `scipy` library
- A created popularity recommender model

For the initial training of the models using `surprise`:
- The default parameters of `BaselineOnly`, `KNNBaseline`, `KNNBasic`, `KNNWithMeans`, `KNNWithZScore`, `CoClustering`, `SVD`, `SVDpp`, `NMF` and `NormalPredictor` were evaluated using the `cross_validate` method using 3-fold cross validation to determine which algorithm yielded the lowest `RMSE` errors.
- Then hyperparameter optimization was performed to find the best parameters for the four model types with the lowest `RMSE` errors (`SVDpp`, `SVD`, `BaselineOnly` and `KNNBaseline`).

For the construction of the ranking recommender model:
- The data was loaded in a `tensorflow` dataset and partitioned into train/test sets.
- A vocabulary was generated to map the feature values to embedding vectors.
- The model used an `embedding_dimension=64` and contained multiple stacked dense layers with `activation='relu`.
- Raw features were used as input to `compute_loss` and `MeanSquaredError` was used for the loss metric.
- The model was fit for 20 epochs and evaluated.
- The ranking model was tested by computing predictions for items and ranking by the predictions made.
- The model was saved and exported for serving purpose in `TensorFLow Lite`.

For the training of the `SVD` based models using `SciPy`:
- A rating matrix with items and reviewers was constructed.
- The parameters for the model were `U, sigma, Vt` were tested in `randomized_svd` and `TruncatedSVDTruncatedSVD`.
- A diagonal matrix was constructed in `SVD`.
- Then the ratings were predicted and RMSE was calculated. 

For the construction of the popularity recommender model:
- A recommendation score was created by counting each reviewer for each unique item.
- This score was sorted and a recommendation rank was created based on scoring.
- Predictions were then calculated for various reviewers.
