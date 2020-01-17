import pandas as pd
import numpy as np
import normal as nm
import numpy as np
import knn 

def ClassifyGrade(inArr):

    #read the file and get the values
    df = pd.read_csv('mlscripts/studentsgrade.csv')
    numdf = df.values

    #get the last col, get the number of rows and columns in the dataframe 

    row_num = numdf.shape[0] 
    col_num = numdf.shape[1]
    grade = numdf[: , col_num - 1]
    features = numdf[: , :col_num - 1]
    
    #normalization the features
    norm_features , therange, mincols = nm.Norm(features)

    sub = inArr - mincols[:, None]
    inputs = sub / therange[:, None]

    classified_result = knn.KNN(features , grade , inArr, 3)
    #predicted_grade = grade[classified_result - 1]
    return (classified_result)


#asn = ClassifyGrade([3,5,60, 9, 3])
#print(asn)