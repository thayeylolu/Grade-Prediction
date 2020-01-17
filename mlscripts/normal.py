import numpy as np
import knn 
def Norm(dataset_features):
    '''
        Normailization makes vector have equal weight ( or importance) 
        when vectors are passed into the machine learning model

        the formula :=
         prev_value - min / range
         that is: div(prev_value - min , range)
         where range = max - min

    It returns a new dataset which is normalized
    '''

    # get the minimum and maximum values of all the columns (features) in the dataset
    norm_data  = np.zeros((dataset_features.shape))
    max_in_cols = np.max(dataset_features, axis = 1)
    min_in_cols = np.min(dataset_features, axis = 1)

    # range
    the_range = max_in_cols - min_in_cols

    # subtract the dataset from the min
    sub = dataset_features - min_in_cols[:, None]
    norm_data = sub/the_range[:, None]

    return (norm_data , the_range, min_in_cols)

