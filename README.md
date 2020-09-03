# Optimal-K-Nearest-Neighbor


Introduction to Statistical Learning

1) I will Fit KNN with K = 1,2,...,100. In order to find the optimal K value. 

2) From figure(1) it is evident to see that the testing error is lower than the training error. This is not what we would expect. Therefore, exploration of the training and testing datasets was necessary. From the two scatter plots shown under figure(1) we can see two things. The data in the training set are scattered around without dense clusters*. However, the test data seems to have 4 dense clusters and the data is not too scattered. This means the model had harder cases for the training set and had easier cases for testing set. Thus, causing a lower testing error than training error. 

*In other words the quality of the trainig set is not good.

![TrainTestError](https://github.com/JaimeGoB/Optimal-K-Nearest-Neighbor/blob/master/other/testin-vs-error-rate.png)


3) After running K value from 1-100. The optimal K value for nearest neighbors is 26. When K is 26, the testing error is the lowest with 0.02825 error. On the other hand, the training error is 0.03025.

4) The Decision Boundary for the optimal K shows two important things. The first is the decision boundary that is shown in the black line, it partitions the space into two sets or regions for each class(yes or no). Depending on the x.1 value and x.2 value a given observation it will place it on its respective class(yes or no) and color. The decision boundary cuts through the middle; however, it does not work perfectly because it is possible to see red points in the green boundary and green points in red boundary. This represents the minimal error associated with the K value. Despite the minimal error, the predictions for the model when K is 26 are reasonable.

![DecisionBoundary1](https://github.com/JaimeGoB/Optimal-K-Nearest-Neighbor/blob/master/other/Decision-Boundary-Optimal-K.png)

