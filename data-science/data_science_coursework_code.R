library(ggplot2)
library(corrplot)
library(dplyr)
#Import the dataset
data <- read.csv("Metro.csv", header=TRUE)

#transfer the format of time
data$timestamp <- as.POSIXct(data$timestamp, format = "%d/%m/%Y %H:%M")
#Add dataframe containing failure period
failures <- data.frame(
  start = as.POSIXct(c("18/04/2020 00:00", "29/05/2020 23:30", "05/06/2020 10:00"),
                     format = "%d/%m/%Y %H:%M"),
  end   = as.POSIXct(c("18/04/2020 23:59", "30/05/2020 06:00", "07/06/2020 14:30"),
                     format = "%d/%m/%Y %H:%M")
)
data$failure <- 0
#Add correct value to the column failure.
for (i in 1:nrow(failures)) {
  data$failure[data$timestamp >= failures$start[i] & data$timestamp <= failures$end[i]] <- 1
}



#Plot correlation matrix given the dataset
data_modi <- data %>%
  select_if(is.numeric) %>%
  select(-X, -failure)

cor_matrix <- cor(data_modi, use = "complete.obs")
corrplot(cor_matrix, method = "circle", 
         order = "hclust", tl.col = "blue", tl.cex = 0.8, tl.srt = 45,
         main = "Variables Correlation Matrix ")

par(mfrow = c(3, 2), mar = c(4, 5, 3, 1))
#Display boxplots
boxplot(TP3 ~ failure, data = data, main = "TP3 vs Failure", col = c("lightblue", "red"), outline = FALSE)
boxplot(H1 ~ failure, data = data, main = "H1 vs Failure", col = c("lightblue", "red"), outline = FALSE)
boxplot(Motor_current ~ failure, data = data, main = "Motor_Current vs Failure", col = c("lightblue", "red"), outline = FALSE)
boxplot(TP2 ~ failure, data = data, main = "TP2 vs Failure", col = c("lightblue", "red"), outline = FALSE)
boxplot(Oil_temperature ~ failure, data = data, main = "Oil_Temperature vs Failure", col = c("lightblue", "red"), outline = FALSE)
boxplot(DV_pressure ~ failure, data = data, main = "DV_Pressure vs Failure", col = c("lightblue", "red"), outline = FALSE)

#Display proportion corresponding tables
oil_prop <- prop.table(table(data$failure, data$Oil_level), margin = 1)
oil_df <- as.data.frame.matrix(oil_prop)
names(oil_df) <- c("Oil_Level = 0", "Oil_Level = 1")
rownames(oil_df) <- c("Failure = 0", "Failure = 1")

cat("Proportion of Oil Level States with Different Failure Situations:\n")
print(oil_df)

ps_prop <- prop.table(table(data$failure, data$Pressure_switch), margin = 1)
ps_df <- as.data.frame.matrix(ps_prop)
names(ps_df) <- c("Pressure_Switch = 0", "Pressure_Switch = 1")
rownames(ps_df) <- c("Failure = 0", "Failure = 1")

cat("\nProportion of Pressure Switch States with Different Failure Situations:\n")
print(ps_df)

model1 <- glm(failure ~ as.factor(Oil_level), data = data, family = "binomial")
model2 <- glm(failure ~ as.factor(Pressure_switch), data = data, family = "binomial")
data.frame(
  Model = c("Oil_level", "Pressure_switch"),
  AIC = c(AIC(model1), AIC(model2)),
  NullDeviance = c(model1$null.deviance, model2$null.deviance),
  ResidualDeviance = c(model1$deviance, model2$deviance)
)
#split data into testing and training
split_ind <- floor(0.7 * nrow(data))
split_time <- data$timestamp[split_ind]
train <- data[data$timestamp <= split_time,]
test <- data[data$timestamp > split_time,]
#Remove some variables
train <- train[, !names(train) %in% c("X", "timestamp", "Reservoirs", "Pressure_switch")]
test  <- test[, !names(test) %in% c("X", "timestamp", "Reservoirs", "Pressure_switch")]
#Modelling for Random Forest
#Transfer failure as factor and remove missing value
train_clean <- na.omit(train)
test_clean <- na.omit(test)
train_clean$failure <- as.factor(train_clean$failure)
test_clean$failure <- as.factor(test_clean$failure)
train_clean$COMP <- as.factor(train_clean$COMP)
test_clean$COMP <- as.factor(test_clean$COMP)
train_clean$DV_eletric <- as.factor(train_clean$DV_eletric)
test_clean$DV_eletric <- as.factor(test_clean$DV_eletric)
train_clean$Towers <- as.factor(train_clean$Towers)
test_clean$Towers <- as.factor(test_clean$Towers)
train_clean$MPG <- as.factor(train_clean$MPG)
test_clean$MPG <- as.factor(test_clean$MPG)
train_clean$LPS <- as.factor(train_clean$LPS)
test_clean$LPS <- as.factor(test_clean$LPS)
train_clean$Oil_level <- as.factor(train_clean$Oil_level)
test_clean$Oil_level <- as.factor(test_clean$Oil_level)
train_clean$Caudal_impulses <- as.factor(train_clean$Caudal_impulses)
test_clean$Caudal_impulses <- as.factor(test_clean$Caudal_impulses)


rt <- trainControl(method = "cv", number = 5)
#train the model
model_rf <- train(failure ~ ., data = train_clean,
                  method = "rf",
                  trControl = rt,
                  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8)),
                  ntree = 100,
                  metric = "Accuracy")


#prediction
predict_rf <- predict(model_rf, newdata = test_clean)
predict_prob <- predict(model_rf, newdata = test_clean, type = "prob")[, "1"]
conf_mat <- confusionMatrix(predict_rf, test_clean$failure, positive = "1")
print(conf_mat$table)                    
cat("\nAccuracy:", round(conf_mat$overall["Accuracy"], 3), "\n")
cat("\nSensitivity:", round(conf_mat$byClass["Sensitivity"], 3), "\n")
cat("\nPrecision:", round(conf_mat$byClass["Precision"], 3), "\n")

roc_acc <- roc(test_clean$failure, predict_prob)
plot(roc_acc, main = "ROC Curve For Random Forest(Accuracy-Tuned )")
#return corresponding auc value
auc_value <- auc(roc_acc)
cat("AUC:", round(auc_value, 3), "\n")

#plot feature Importance
fea_importance <- varImp(model_rf)
plot(fea_importance, main="Feature Importance for Accuracy Metric Model")

#ROC tuning case
rt2 <- trainControl(method = "cv", number = 5,
                    classProbs = TRUE,
                    summaryFunction = twoClassSummary)

#transfer variable can be used for ROC
levels(train_clean$failure) <- c("No", "Yes")
levels(test_clean$failure)  <- c("No", "Yes")

model_rf2 <- train(failure ~ ., data = train_clean,
                   method = "rf",
                   trControl = rt2,
                   tuneGrid = expand.grid(mtry = c(2, 4, 6, 8)),
                   ntree = 100,
                   metric = "ROC")

#predict corresponding probability
predict_rf2 <- predict(model_rf2, newdata = test_clean, type = "prob")[,"Yes"]
predict_class2 <- predict(model_rf2, newdata = test_clean)
#ROC curve, accuracy and sensitivity
conf_mat2 <- confusionMatrix(predict_class2, test_clean$failure, positive = "Yes")
print(conf_mat2$table)                    
cat("\nAccuracy:", round(conf_mat2$overall["Accuracy"], 3), "\n")
cat("\nSensitivity:", round(conf_mat2$byClass["Sensitivity"], 3), "\n")
cat("\nPrecision:", round(conf_mat2$byClass["Precision"], 3), "\n")
roc_obj <- roc(test_clean$failure, predict_rf2)
plot(roc_obj, main = "ROC Curve For Random Forest (ROC-Tuned)")
#return corresponding auc value
auc2 <- auc(roc_obj)
cat("AUC:", round(auc2, 3), "\n")

#plot feature importance
fea_importance2 <- varImp(model_rf2)
plot(fea_importance2, main="Feature Importance for Random Forest Model(ROC Metric)")

#modelling logistic regression
#transfer back the failure value
levels(train_clean$failure) <- c("0", "1")
levels(test_clean$failure)  <- c("0", "1")
model_logit <- glm(failure ~., data = train_clean, family = "binomial"(link = "logit"))
estimates <- coef(model_logit)
estimates_df <- data.frame(estimate = estimates)
print(estimates_df)

#modelling lrm with reduced feature
train_new <- train_clean[,-c(7:13)]
test_new <- test_clean[,-c(7:13)]
model_logit2 <- glm(failure ~., data = train_new, family = "binomial"(link = "logit"))
estimates2 <- coef(model_logit2)
estimates_df2 <- data.frame(estimate = estimates2)
print(estimates_df2)

#function for evaluating
model_evalu <- function(pred_class, prob, testing){
  cm <- confusionMatrix(pred_class, testing, positive = "1")
  accuracy <- cm$overall["Accuracy"]
  sensitivity <- cm$byClass["Sensitivity"]
  precision <- cm$byClass["Precision"]
  table_cm <- cm$table
  list(table = table_cm, accuracy = accuracy, sensitivity = sensitivity, precision = precision)
  
}

#For Logistic Regression
pred_prob_glm <- predict(model_logit2, newdata = test_new, type = "response")
pred_class_glm <- factor(ifelse(pred_prob_glm > 0.5, "1", "0"), levels = c("0", "1"))

#evaluation
evalu_glm <- model_evalu(pred_class_glm, pred_prob_glm, test_new$failure)
print(evalu_glm$table)
cat("Accuracy for GLM:", round(evalu_glm$accuracy, 3),"\n")
cat("Sensitivity for GLM:", round(evalu_glm$sensitivity, 3),"\n")
cat("Precision for GLM:", round(evalu_glm$precision, 3),"\n")