import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#Commented by: James Xi Zheng 12/Aug/2019
#please create functions to do the following jobs
#1. load dataset ->  sample code availalable in the workshops
#2. visualize data -> sample code given
#3. remove signal noises -> sample code given
#4. extract features -> sample code given
#5. prepare training set -> sample code given 
#6. training the given models -> sample code given
#7. test the given models -> sample code given
#8. print out the evaluation results -> sample code given

#as I said in the lecture, the sample code is completed in a un-professional software engineering style
#software refactoring is required
#please manage the project using SCRUM sprints and manage the source code using Github
#document your progress and think critically what are missing from such IoT application and what are missing to move such IoT application from PoC (proof of concept) to solve real-world life
#think with which components added, what kind of real-world problems can be solved by it -> this shall be discussed in the conclusion part in the document

'''
At first, we should explore the raw time-series sensor data. We could draw line plot of sensor signals.
In this example code, the wrist sensor accelerometer data dataset_1 sitting activity is visualized.   
'''
def data_visulization(file,pos,firstAc, lastAc, firstGy, lastGy):
    # read dataset file
    df =  reader(file)
    df_sitting = df[df[24] == pos].values
    graph,axes = plt.subplots(2, 2,figsize=(20,15))
    axes[0][0].plot(shower(df_sitting,file,pos,500,2500,firstAc,lastAc,'Raw Acceleration',axes[0][0]))
    axes[0][1].plot(shower(df_sitting,file,pos,500,2500,firstGy,lastGy,'Raw Gyroscopic',axes[0][1]))
    if(firstAc<=firstGy):
        df_filtered = noise_removing(df,pos,firstAc,lastGy)
    else:
        df_filtered = noise_removing(df,pos,firstGy,lastAc)
        
    axes[1][0].plot(shower(df_filtered,file,pos,500,2500,firstAc,lastAc,'Filtered Acceleration',axes[1][0]) )
    axes[1][1].plot(shower(df_filtered,file,pos,500,2500,firstGy,lastGy,'Filtered Gyroscopic',axes[1][1])   )
    graph.show()

'''
For raw sensor data, it usually contains noise that arises from different sources, such as sensor mis-
calibration, sensor errors, errors in sensor placement, or noisy environments. We could apply filter to remove noise of sensor data
to smooth data. In this example code, Butterworth low-pass filter is applied. 
'''
def noise_removing(df,label,sectStart,sectEnd):
    # Butterworth low-pass filter. You could try different parameters and other filters. 
    b, a = signal.butter(4, 0.04, 'low', analog=False)
    df_filtered = df[df[24] == label].values
    for i in range(sectStart,sectEnd):
        df_filtered[:,i] = signal.lfilter(b, a, df_filtered[:, i])
    return df_filtered
    


'''
To build a human activity recognition system, we need to extract features from raw data and create feature dataset for training 
machine learning models.

Please create new functions to implement your own feature engineering. The function should output training and testing dataset.
'''
def feature_engineering_example(features):
    training = np.empty(shape=(0, features*3 +1))
    testing = np.empty(shape=(0, features *3 +1))
    # deal with each dataset file
    for i in range(19):
        df = reader(i+1)
        print('deal with dataset ' + str(i + 1))
        for c in range(1, 14):
            activity_data = df[df[24] == c].values
            activity_data = noise_removing(df,c,6,12)
            datat_len = len(activity_data)
            training_len = math.floor(datat_len * 0.8)
            training_data = activity_data[:training_len, :]
            testing_data = activity_data[training_len:, :]

            # data segementation: for time series data, we need to segment the whole time series, and then extract features from each period of time
            # to represent the raw data. In this example code, we define each period of time contains 1000 data points. Each period of time contains 
            # different data points. You may consider overlap segmentation, which means consecutive two segmentation share a part of data points, to 
            # get more feature samples.
            training_sample_number = training_len // 1000 + 1
            testing_sample_number = (datat_len - training_len) // 1000 + 1
            
            #training = dataCreater(training_sample_number,training_data,training_sample_number,training)
            #testing = dataCreater(testing_sample_number,testing_data,training_sample_number,testing)
                          
            for s in range(training_sample_number):   
                training = np.concatenate((training, dataCreater(training_data,training_sample_number,s,6,9,features)), axis=0)
            
            for s in range(testing_sample_number): 
                testing = np.concatenate((testing, dataCreater(testing_data,training_sample_number,s,6,9,features)), axis=0)

    df_training = pd.DataFrame(training)
    df_testing = pd.DataFrame(testing)
    df_training.to_csv('training_data.csv', index=None, header=None)
    df_testing.to_csv('testing_data.csv', index=None, header=None)
    model_training_and_evaluation_example(3,features)
    #model_training_and_evaluation_example(4,features)
    
def dataCreater(data, trainingNumber,s,rangeStart, rangeEnd,features):
                if s < trainingNumber - 1:
                    sample_data = data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = data[1000*s:, :]

                feature_sample = []
                for i in range(rangeStart,rangeEnd):
                    feature_sample.append(np.min(sample_data[:, i]))
                    feature_sample.append(np.max(sample_data[:, i]))
                    if features >= 3:
                        feature_sample.append(np.mean(sample_data[:, i]))                    
                    if features >= 4:
                        feature_sample.append(np.average(sample_data[:, i]))
                    if features >= 5:
                        feature_sample.append(np.std(sample_data[:,i]))
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])  
                return feature_sample

'''
When we have training and testing feature set, we could build machine learning models to recognize human activities.

Please create new functions to fit your features and try other models.
'''
def model_training_and_evaluation_example(neighbours,features):
    df_training = pd.read_csv('training_data.csv', header=None)
    df_testing = pd.read_csv('testing_data.csv', header=None)
    
    X_test = df_testing.drop([features*3], axis=1).values
    X_train = df_training.drop([features*3], axis=1).values
    Y_train = df_training[features*3].values -1
    Y_test = df_testing[features*3].values -1
    
    # Feature normalization for improving the performance of machine learning models. In this example code, 
    # StandardScaler is used to scale original feature to be centered around zero. You could try other normalization methods.
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Build KNN classifier, in this example code
    KNNBuilderTester(X_train,Y_train,X_test,Y_test,neighbours)
    
    
    # Another machine learning model: svm. In this example code, we use gridsearch to find the optimial classifier
    # It will take a long time to find the optimal classifier.
    # the accuracy for SVM classifier with default parameters is 0.71, 
    # which is worse than KNN. The reason may be parameters of svm classifier are not optimal.  
    # Another reason may be we only use 9 features and they are not enough to build a good svm classifier. \
    SVCBuilderTester(X_train,Y_train,X_test,Y_test)
    
def KNNBuilderTester(X_train,Y_train,X_test,Y_test,neighbours):
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(X_train, Y_train)

    # Evaluation. when we train a machine learning model on training set, we should evaluate its performance on testing set.
    # We could evaluate the model by different metrics. Firstly, we could calculate the classification accuracy. In this example
    # code, when n_neighbors is set to 4, the accuracy achieves 0.757.
    y_pred = knn.predict(X_test)
    print('Accuracy: ', accuracy_score(Y_test, y_pred))
    # We could use confusion matrix to view the classification for each activity.
    print(confusion_matrix(Y_test, y_pred))
    
def SVCBuilderTester(X_train,Y_train,X_test,Y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2, 1e-3, 1e-4],
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}]
    acc_scorer = make_scorer(accuracy_score)
    grid_obj  = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=acc_scorer)
    grid_obj  = grid_obj .fit(X_train, Y_train)
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))


def reader(i):
    return pd.read_csv('dataset_' + str(i) + '.txt', sep=',', header=None)

def shower(data,person, pos, tStart, tEnd, sectStart, sectEnd,dataType,plot):    
    activity =  {
        1: "Sitting",
        2: "Lying", 
        3: "Standing", 
        4: "Washing Dishes", 
        5: "Vacuuming", 
        6: "Sweeping", 
        7: "Walking", 
        8: "Ascending Stairs",
        9: "Descending stairs", 
        10: "Running on a Treadmill", 
        11: "Riding on a 50W Ergometer", 
        12: "Riding on a 100W Ergometer", 
        13: "Jumping Rope",    
    }.get(pos) 
    plot.set_title('%s Data of person %ds Chest while %s from time %d-%d:'%(dataType,person,activity,tStart,tEnd))
    return data[tStart:tEnd , sectStart:sectEnd]
    



if __name__ == '__main__':
    
    #data_visulization(1,1,6,9,9,12)
    #data_visulization(2,2,6,9,9,12)
    
    feature_engineering_example(4)
    #model_training_and_evaluation_example(4,5)
    #model_training_and_evaluation_example(4)