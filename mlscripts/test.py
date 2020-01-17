import pandas as pd
import numpy as np
import normal as nm
import knn 

def GradeTest():

    #increase rate
    rate = 0.30

    #read the file and get the values
    df = pd.read_csv('studentsgrade.csv')
    numdf = df.values

    #get the last col, get the number of rows and columns in the dataframe 
    

    row_num = numdf.shape[0] 
    col_num = numdf.shape[1]
    grade = numdf[:, col_num - 1]
    features = numdf[:, :col_num - 1]

    #number of test data, initialize the error counter
    test_data = int(row_num * rate)
    error_count = 0.0

    #normalization the features
    norm_features , therange, mincols = nm.Norm(features)

    df_test  = norm_features[test_data:row_num,:]
    label = grade[test_data:row_num]

    

    for i in range (test_data):
        classified_result = knn.KNN(df_test,label,norm_features[i, :],3)
        print('The classifiier returned {}. The real answer is: {}'.format(classified_result , grade[i]))
        
        if (classified_result != grade[i]):
            error_count += 1.0
    print( 'Total Error rate is: {}. '.format(error_count/float(test_data)))

GradeTest()