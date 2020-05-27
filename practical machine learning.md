---
title: "practical machine learning"
author: "Ravi Bhushan Bhardwaj"
date: "May 27, 2020"
output: html_document
---
```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(RGtk2)
library(rattle)
library(randomForest)
library(gbm)
```

##Here are the datasets, loaded directly from web and then downloaded.
```{r}
training.data<- 'pml-training.csv'
testing.data <- 'pml-testing.csv'
training.data<- read.csv('pml-training.csv')
testing.data <-read.csv('pml-testing.csv')
```
```{r}
dim(training.data)
```
```
[1] 19622   160
```
```{r}
dim(testing.data)
```
```
[1]  20 160
```
#Data Cleansing
#A. Removing Variables which are having nearly zero variance.
```{r}
non_zero_var <- nearZeroVar(training.data)
training.df <- training.data[,-non_zero_var]
testing.df <- testing.data[,-non_zero_var]
```
```{r}
dim(training.df)
```
```
[1] 19622   100
```
```{r}
dim(testing.df)
```
```
[1]  20 100
```
#B. Removing Variables which are having NA values. Our threshhold is 95%.
```{r}
na_val_col <- sapply(training.df, function(x) mean(is.na(x))) > 0.95
training.df <- training.df[,na_val_col == FALSE]
testing.df <- testing.df[,na_val_col == FALSE]
```
```{r}
dim(training.df)
```
[1] 19622    59
```{r}
dim(testing.df)
```
```
[1] 20 59
```
#C. Removing variables which are non-numeric and hence will not contribute into our model. The very first 7 variables are of that kind only. Hence those needs to be removed from the datasets.
```{r}
training.df <- training.df[,8:59]
testing.df <- testing.df[,8:59]
```
```{r}
dim(training.df)
```
```
[1] 19622    52
```
```{r}
dim(testing.df)
```
```
[1] 20 52
```
```{r}
colnames(training.df)
```
```
[1] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
 [4] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
 [7] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
[10] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
[13] "roll_arm"             "pitch_arm"            "yaw_arm"             
[16] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
[19] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
[22] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
[25] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
[28] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
[31] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
[34] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
[37] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
[40] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
[43] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
[46] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
[49] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
[52] "classe"
```
```{r}
colnames(testing.df)
```
```
[1] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
 [4] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
 [7] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
[10] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
[13] "roll_arm"             "pitch_arm"            "yaw_arm"             
[16] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
[19] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
[22] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
[25] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
[28] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
[31] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
[34] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
[37] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
[40] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
[43] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
[46] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
[49] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
[52] "problem_id"
```

#Partition the training data into a training set and a testing/validation set
```{r}
inTrain <- createDataPartition(training.df$classe, p=0.6, list=FALSE)
training <- training.df[inTrain,]
testing <- training.df[-inTrain,]
dim(training)
```
```
[1] 11776    52
```
```{r}
dim(testing)
```
```
[1] 7846   52
```
```{r}
DT_modfit <- train(classe ~ ., data = training, method="rf")
DT_prediction <- predict(DT_modfit, testing)
confusionMatrix(DT_prediction,DT_prediction)
```
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 4203    0    0    0    0
         B    0 1122    0    0    0
         C    0    0 1273    0    0
         D    0    0    0  824    0
         E    0    0    0    0  424

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9995, 1)
    No Information Rate : 0.5357     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
                                     
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000    1.000   1.0000    1.000  1.00000
Specificity            1.0000    1.000   1.0000    1.000  1.00000
Pos Pred Value         1.0000    1.000   1.0000    1.000  1.00000
Neg Pred Value         1.0000    1.000   1.0000    1.000  1.00000
Prevalence             0.5357    0.143   0.1622    0.105  0.05404
Detection Rate         0.5357    0.143   0.1622    0.105  0.05404
Detection Prevalence   0.5357    0.143   0.1622    0.105  0.05404
Balanced Accuracy      1.0000    1.000   1.0000    1.000  1.00000


```
```{r}
rpart.plot(DT_modfit$finalModel, roundint=FALSE)
```
<img src=C:\Users\Admin\Documents\Rplottree1.png>

#Random Forest Model
```{r}
RF_modfit <- train(classe ~ ., data = training, method = "rf", ntree = 100)
```
###Prediction in terms of Random Forest Model
```{r}
RF_prediction <- predict(RF_modfit, testing)
RF_pred_conf <- confusionMatrix(RF_prediction, testing$classe)
```
```
Confusion Matrix and Statistics
Confusion Matrix and Statistics

          Reference
 Prediction    A    B    C    D    E
          A 2230   14    0    0    0
          B    1 1495   10    0    0
          C    1    7 1353   17    1
          D    0    0    5 1268    5
          E    0    2    0    1 1436
 
 Overall Statistics
                                           
                Accuracy : 0.9918          
                  95% CI : (0.9896, 0.9937)
     No Information Rate : 0.2845          
     P-Value [Acc > NIR] : < 2.2e-16       
                                           
                  Kappa : 0.9897          
  Mcnemar's Test P-Value : NA              
 
 Statistics by Class:
 
                      Class: A Class: B Class: C Class: D Class: E
 Sensitivity            0.9991   0.9848   0.9890   0.9860   0.9958
 Specificity            0.9975   0.9983   0.9960   0.9985   0.9995
 Pos Pred Value         0.9938   0.9927   0.9811   0.9922   0.9979
 Neg Pred Value         0.9996   0.9964   0.9977   0.9973   0.9991
 Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
 Detection Rate         0.2842   0.1905   0.1724   0.1616   0.1830
 Detection Prevalence   0.2860   0.1919   0.1758   0.1629   0.1834
 Balanced Accuracy      0.9983   0.9916   0.9925   0.9922   0.9977
```
#Gradient Boosting Model 
```{r}
 GBM_modfit <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE)
```
```{r}
GBM_prediction <- predict(GBM_modfit, testing)
```
```{r}
GBM_pred_conf <- confusionMatrix(GBM_prediction, testing$classe)
```
```
Confusion Matrix and Statistics
           Reference
 Prediction    A    B    C    D    E
          A 2191   61    0    2    1
          B   26 1414   40    7   11
          C  8   36 1313   40   16
          D    6    1   12 1220   18
          E    1    6    3   17 1396
 
 Overall Statistics
                                           
                Accuracy : 0.9602          
                  95% CI : (0.9557, 0.9645)
     No Information Rate : 0.2845          
     P-Value [Acc > NIR] : < 2.2e-16       
                                           
                   Kappa : 0.9497          
  Mcnemar's Test P-Value : 4.337e-08       
 
 Statistics by Class:
 
                      Class: A Class: B Class: C Class: D Class: E
 Sensitivity            0.9816   0.9315   0.9598   0.9487   0.9681
 Specificity            0.9886   0.9867   0.9846   0.9944   0.9958
 Pos Pred Value         0.9716   0.9439   0.9292   0.9706   0.9810
 Neg Pred Value         0.9927   0.9836   0.9915   0.9900   0.9928
 Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
 Detection Rate         0.2793   0.1802   0.1673   0.1555   0.1779
 Detection Prevalence   0.2874   0.1909   0.1801   0.1602   0.1814
 Balanced Accuracy      0.9851   0.9591   0.9722   0.9715   0.9819
```
##** Now we need to see how each model has predicted the validation dataset across the classifications. ** We are not considering Decision Tree model as it didn??????t reach the satisfactory prediction accuracy level. SO only Random Forest and Gradient Boosting methods are being compared.
```{r}
RF_pred_conf$overall
```
```
 Accuracy       Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
0.9918430      0.9896806      0.9895954      0.9937126      0.2844762 
 AccuracyPValue  McnemarPValue 
      0.0000000            NaN
```
```{r}
GBM_pred_conf$overall
```
```
Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
9.602345e-01   9.496836e-01   9.556727e-01   9.644503e-01   2.844762e-01 
 AccuracyPValue  McnemarPValue 
 0.000000e+00   4.336741e-08
```
#Conclusion 
###After checking the Overall Statistics data, the Random Forest model has definitely more accuracy than GBM. Hence we will be selecting Random Forest model for final prediction from testing.df
#Final Prediction- Applying selected model on the Test Data
```{r}
Final_RF_prediction <- predict(RF_modfit, testing.df)
```
```
[1] B A B A A E D B A A B C B A E E A B B B
 Levels: A B C D E
```
