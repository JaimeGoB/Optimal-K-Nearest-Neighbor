#Jose Jaime Goicoechea
# K-Nearest Neighbors (K-NN)

#Libraries needed to run knn
library(class)

#Reading csv file into trainingData variable
training_set <- read.csv("1-training_data(1).csv")
y_train = training_set[['y']]
x_train <- cbind(training_set[['x.1']],training_set[['x.2']])

#reading csv file for test data
test_set <- read.csv("1-test_data(1).csv")
y_test = test_set[['y']]
x_test <- cbind(test_set[['x.1']],test_set[['x.2']])

#setting up vectors to hold errors for each k
train_error_vector <- c(1:100)*0
test_error_vector <- c(1:100)*0

#This function will calculate missclassifcation error from
#predicted class label from knn and actual class label
missclasification_error = function(label, predictedLabel){
  #Add all errors and divided by # of errors
  mean(label != predictedLabel)
}

#################################################
#(a) Fit KNN with K = 1,2,...,100.
#################################################
for(k_index in 1:100){
  set.seed(1)
  #using training data to train knn model
  y_pred_train = knn(train = x_train,
               test = x_train,
               cl = y_train,
               k = k_index, 
               )
  #calculate the training error for this k neigheres neighbors
  train_error_vector[k_index] = missclasification_error(y_train ,y_pred_train)

  set.seed(1)
  #using test data on model
  y_pred_test = knn(train = x_train,
                    test = x_test,
                    cl = y_train,
                    k = k_index,
                    )
  #calculate the training error for this k neigheres neighbors
  test_error_vector[k_index] = missclasification_error(y_test ,y_pred_test)
}

#################################################
#(b) Plot training and test error rates against K.
#################################################
kNeighbors <- c(seq(1, 100, by = 1))

plot(kNeighbors, train_error_vector, xlab = "Number of nearest neighbors", ylab = "Error rate", 
     main = "Figure (1) Error rate on Traing and Testing datasets", type = "b", ylim = range(c(train_error_vector, test_error_vector)), col = "blue", pch = 20)
lines(kNeighbors, test_error_vector, type="b", col="red", pch = 20)
abline(h = min(test_error_vector), col = "green", lty = 2)
legend("bottomright", lty = 1, col = c("blue", "red", "green"), legend = c("training", "test", "min value test error"))

#plotting the training set and testing set to see differences in datasets
par(mfrow=c(1,2))
plot(training_set[["x.1"]], training_set[["x.2"]], xlab = "x.1", ylab = "x.2", main = "Training Dataset ")
plot(test_set[["x.1"]], test_set[["x.2"]], xlab = "x.1", ylab = "x.2", main = "Test Dataset ")
par(mfrow = c(1, 1))

#################################################
#(c) What is the optimal value of K? -> optimal_K = 26
#################################################
#getting errors and knn index in a vector
min_error_vector <- data.frame(kNeighbors, train_error_vector, test_error_vector)
#getting the row with lowest test error
optimal_K_errors <- min_error_vector[test_error_vector == min(min_error_vector$test_error_vector), ]
optimal_K_errors
#storing optimal k from vector
optimal_K_number <- optimal_K_errors[["kNeighbors"]]

#################################################
#(d) Make a plot of the training data that also shows the decision boundary for the optimal K
#################################################
grid_length <- 100
grid_x1 <- seq(f = min(x_train[, 1]), t = max(x_train[, 1]), l = grid_length)
grid_x2 <- seq(f = min(x_train[, 2]), t = max(x_train[, 2]), l = grid_length)
grid <- expand.grid(grid_x1, grid_x2)

set.seed(1)
optimal_knn_model <- knn(x_train, grid, y_train, k = optimal_K_number, prob = T)
prob <- attr(optimal_knn_model, "prob")
prob <- ifelse(optimal_knn_model == "yes", prob, 1 - prob)
prob <- matrix(prob, grid_length, grid_length)

#Make a plot of the training data that also shows the decision boundary for the optimal K ####
plot(x_train, col = ifelse(y_train == "yes", "green", "red"), xlab = "x.1", ylab = "x.2", main = "Decision Boundary for 26-Nearest Neighbors" )
contour(grid_x1, grid_x2, prob, levels = 0.5, labels = "", xlab = "", ylab = "", 
        main = "", add = TRUE)
points(grid, pch = '.', col = ifelse(optimal_knn_model == "yes", 'springgreen3', 'tomato'))


