# Applying the kNN (k-Nearest Neighbour) algorithm on Pima Indians Diabetes dataset

#Load the packages
library(mlbench)
library(class)
library(caTools)
library(caret)


#Load the data
data("PimaIndiansDiabetes2")
df <- PimaIndiansDiabetes2


# Inspect the data --------------------------------------------------------

head(df)
summary(df)


#Remove all NAs and keep all 9 features/variables
#we sacrifice  half of our dataset 

df_2 <- na.omit(df)


#Alternatively we could remove the two variables with the highest frequency of NAs
#remove inuslin and triceps and do a separate analyses.
#We would be left with a lot more data
#We will proceed with the second option

df_3 <- df
df_3$triceps <- NULL
df_3$insulin <- NULL
df_3 <- na.omit(df_3)


# kNN Algorithm requires all variables besides the dependent variable to be numeric.

str(df_3)

#We now need to normalise  the data so kNN isn’t improperly influenced by different measurments
#First create a function

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#normalize all the numeric features in the data frame

df_3[1:6] <- as.data.frame(lapply(df_3[1:6], normalize))
summary(df_3)


#Split data into training and testing sets

splitcc <- sample.split(df_3, SplitRatio = 0.7)
train <- subset(df_3, splitcc == "TRUE")
test <- subset(df_3, splitcc == "FALSE")



# Testing Models k-NN  -------------------------------------------------------

#Minimum accuracy to beat
table(df_3$diabetes)
minimum <- 475/724


#Create Labels
trainlabels <- train[, "diabetes"]
testlabels <- test[, "diabetes"]


#Implement the kNN model with k = 1
knn_1 <- knn(train[1:6], test[1:6], cl = trainlabels, k = 1)

#Compare with actual data
#Accuracy of % 71.9
confusionMatrix(knn_1, testlabels)


#Alter k to see if the model can be improved at diagnosing patients

#k = 15
knn_2 <- knn(train[1:6], test[1:6], cl = trainlabels, k = 15)
#Accuracy % 75.8
confusionMatrix(knn_2,testlabels)

#It’s common practice to make the k value equal to the square root of the sum of the
#observations in the training set - the square root of 414 is approximately 20

#k =20
knn_3 <- knn(train[1:6], test[1:6], cl = trainlabels, k = 20)
#Accuracy % 75.8
confusionMatrix(knn_3, testlabels)

#Of the models built knn_2 & knn_3 are the most acurate
#However it is still not accurate enough to diagnose diabetes
#Try different algorithms as kNN in this example is not the
#ideal machine learning algorithm


