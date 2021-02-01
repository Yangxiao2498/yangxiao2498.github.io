rm(list = ls())    #delete objects
cat("\014")        #clear console
library(keras)  #https://keras.rstudio.com/
library(MESS) # calculate auc
library(glmnet)
library(alocvBeta)
library(latex2exp)
library(ggplot2)
library(dplyr)
library(tictoc)
library(gridExtra)
#Keras is a high-level neural networks API developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

#Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
#Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
#For convenience, words are indexed by overall frequency in the dataset, so that for instance 
#the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering 
#operations such as: "only consider the top 10,000 most common words, but eliminate the 
#top 20 most common words".
# https://blogs.rstudio.com/ai/posts/2017-12-07-text-classification-with-keras/
#Lists are the R objects which contain elements of different types like − numbers, strings, vectors and another list inside it. A list can also contain a matrix or a function as its elements. 
p                   =     2500
imdb                =     dataset_imdb(num_words = p, skip_top = 10) #, skip_top = 10
train_data          =     imdb$train$x
train_labels        =     imdb$train$y
test_data           =     imdb$test$x
test_labels         =     imdb$test$y
word_index                   =     dataset_imdb_word_index() 
reverse_word_index           =     names(word_index)
names(reverse_word_index)    =     word_index

numberWords.train   =   max(sapply(train_data, max))
numberWords.test    =   max(sapply(test_data, max))


#create train dataset and test dateset of imbalance data 

train_data_positive = train_data[which(train_labels == 1)][1:4000]
train_data_negetive = train_data[which(train_labels == 0)]
imbalance_train_data = c(train_data_positive,train_data_negetive)
imbalance_label = c(rep(1,4000),rep(0,12500))

test_data_positive = test_data[which(test_labels == 1)][1:4000]
test_data_negetive = test_data[which(test_labels == 0)]
imbalance_test_data = c(test_data_positive,test_data_negetive)

word_index                   =     dataset_imdb_word_index() 
reverse_word_index           =     names(word_index) 
names(reverse_word_index)    =     word_index

review_index                 =     10

decoded_review = function(review_index){
  decoded_review <- sapply(train_data[[review_index]], function(index) {
    word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    if (!is.null(word)) word else "?"
  }) 
  decoded_review
}

#decoded_review <- sapply(train_data[[review_index]], function(index) {
#  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
#  if (!is.null(word)) word else "?"
#}) 

vectorize_sequences <- function(sequences, dimension = p) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}
cat(decoded_review(4272))

X.train          =        vectorize_sequences(imbalance_train_data)
X.test           =        vectorize_sequences(imbalance_test_data)
y.train          =        as.numeric(imbalance_label)
n.train          =        length(y.train)
y.test           =        as.numeric(imbalance_label)
n.test           =        length(y.test)



fit                     =        glmnet(X.train, y.train, family = "binomial", lambda=0.0)

beta0.hat               =        fit$a0
beta.hat                =        as.vector(fit$beta)

y.hat.train = X.train %*% beta.hat +  beta0.hat

distance.P     =    X.train[y.train==1, ] %*% beta.hat + beta0.hat 
distance.N     =    X.train[y.train==0, ] %*% beta.hat + beta0.hat 

breakpoints = pretty( (min(c(distance.P,distance.N))-0.001):max(c(distance.P,distance.N)),n=200)
hg.pos = hist(distance.P, breaks=breakpoints,plot=FALSE) 
hg.neg = hist(distance.N, breaks=breakpoints,plot=FALSE) 
color1 = rgb(0,0,230,max = 255, alpha = 80, names = "lt.blue")
color2 = rgb(255,0,0, max = 255, alpha = 80, names = "lt.pink")
par(mfrow=c(1,1))
plot(hg.pos,col=color1,xlab=TeX('$x^T  \\beta +  \\beta_0$'),main =   paste("train: histogram  ")) # Plot 1st histogram using a transparent color
plot(hg.neg,col=color2,add=TRUE) # Add 2nd histogram using different color


#lambda1      =        c(0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10)
#plot(cv.lasso)
cv.lasso     =        cv.glmnet(X.train, y.train, family = "binomial", alpha = 1,  intercept = TRUE,   nfolds = 10, type.measure="auc")
lasso.fit    =        glmnet(X.train, y.train, lambda = cv.lasso$lambda.min, family = "binomial", alpha =1,  intercept = TRUE)


cv.ridge     =        cv.glmnet(X.train, y.train, family = "binomial", alpha = 0,  intercept = TRUE,   nfolds = 10, type.measure="auc")
ridge.fit    =        glmnet(X.train, y.train, lambda = cv.ridge$lambda.min, family = "binomial", alpha =0,  intercept = TRUE)


cv.ela       =        cv.glmnet(X.train, y.train, family = "binomial", alpha = 0.5,  intercept = TRUE,   nfolds = 10, type.measure="auc")
ela.fit      =        glmnet(X.train, y.train, lambda = cv.ela$lambda.min, family = "binomial", alpha = 0.5,  intercept = TRUE)

par(mfrow=c(3,1))

#may show margins too large error, expand plot area to fix it.
plot(cv.ridge)
title("Ridge Cross-Validation",line = 3)
plot(cv.lasso)
title("Lasso Cross-Validation",line = 3)
plot(cv.ela)
title("Elastic-net Cross-Validation",line = 3)


#Ridge
ridge.beta0.hat         =        ridge.fit$a0
ridge.beta.hat          =        as.vector(ridge.fit$beta)
ridge.train.prob.hat    =        predict(ridge.fit, newx = X.train,  type = "response")

ridge.test.prob.hat     =        predict(ridge.fit, newx = X.test,  type = "response")




dt                      =        0.01
thta                    =        1-seq(0,1, by=dt)
thta.length             =        length(thta)

FPR.train               =        matrix(0, thta.length)
TPR.train               =        matrix(0, thta.length)
FPR.test                =        matrix(0, thta.length)
TPR.test                =        matrix(0, thta.length)
for (i in c(1:thta.length)){
  # calculate the FPR and TPR for train data 
  y.hat.train             =        ifelse(ridge.train.prob.hat > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
  
  # calculate the FPR and TPR for test data 
  y.hat.test              =        ifelse(ridge.test.prob.hat > thta[i], 1, 0)
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
  # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
}
ridge.auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
ridge.auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))

print(paste("train AUC =",sprintf("%.4f", ridge.auc.train)))
print(paste("test AUC  =",sprintf("%.4f", ridge.auc.test)))


errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
ridge.errs      =   rbind(errs.train, errs.test)

ggplot(ridge.errs) + geom_line(aes(x,y,color=type),size = 1.5) + labs(x="False positive rate", y="True positive rate") +
  ggtitle("ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",ridge.auc.train,ridge.auc.test)))


#Lasso
lasso.beta0.hat         =        lasso.fit$a0
lasso.beta.hat          =        as.vector(lasso.fit$beta)
lasso.train.prob.hat    =        predict(lasso.fit, newx = X.train,  type = "response")

lasso.test.prob.hat     =        predict(lasso.fit, newx = X.test,  type = "response")




dt                      =        0.01
thta                    =        1-seq(0,1, by=dt)
thta.length             =        length(thta)

FPR.train               =        matrix(0, thta.length)
TPR.train               =        matrix(0, thta.length)
FPR.test                =        matrix(0, thta.length)
TPR.test                =        matrix(0, thta.length)
for (i in c(1:thta.length)){
  # calculate the FPR and TPR for train data 
  y.hat.train             =        ifelse(lasso.train.prob.hat > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
  
  # calculate the FPR and TPR for test data 
  y.hat.test              =        ifelse(lasso.test.prob.hat > thta[i], 1, 0)
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
  # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
}
lasso.auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
lasso.auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))

print(paste("train AUC =",sprintf("%.4f", lasso.auc.train)))
print(paste("test AUC  =",sprintf("%.4f", lasso.auc.test)))


errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
lasso.errs      =   rbind(errs.train, errs.test)

ggplot(lasso.errs) + geom_line(aes(x,y,color=type),size = 1.5) + labs(x="False positive rate", y="True positive rate") +
  ggtitle("ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",lasso.auc.train,lasso.auc.test)))


#ela
ela.beta0.hat         =        ela.fit$a0
ela.beta.hat          =        as.vector(ela.fit$beta)
ela.train.prob.hat    =        predict(ela.fit, newx = X.train,  type = "response")

ela.test.prob.hat     =        predict(ela.fit, newx = X.test,  type = "response")




dt                      =        0.01
thta                    =        1-seq(0,1, by=dt)
thta.length             =        length(thta)

FPR.train               =        matrix(0, thta.length)
TPR.train               =        matrix(0, thta.length)
FPR.test                =        matrix(0, thta.length)
TPR.test                =        matrix(0, thta.length)
for (i in c(1:thta.length)){
  # calculate the FPR and TPR for train data 
  y.hat.train             =        ifelse(ela.train.prob.hat > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train                =        sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.train                =        sum(y.hat.train[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
  P.train                 =        sum(y.train==1) # total positives in the data
  N.train                 =        sum(y.train==0) # total negatives in the data
  FPR.train[i]            =        FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i]            =        TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
  
  # calculate the FPR and TPR for test data 
  y.hat.test              =        ifelse(ela.test.prob.hat > thta[i], 1, 0)
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    
  # print(paste("K=", K, " ki=",ki, ", K-fold CV=", Kfold.CV.err[i]))
}
ela.auc.train     =       sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
ela.auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))

print(paste("train AUC =",sprintf("%.4f", ela.auc.train)))
print(paste("test AUC  =",sprintf("%.4f", ela.auc.test)))


errs.train      =   as.data.frame(cbind(FPR.train, TPR.train))
errs.train      =   data.frame(x=errs.train$V1,y=errs.train$V2,type="Train")
errs.test       =   as.data.frame(cbind(FPR.test, TPR.test))
errs.test       =   data.frame(x=errs.test$V1,y=errs.test$V2,type="Test")
ela.errs      =   rbind(errs.train, errs.test)

ggplot(ela.errs) + geom_line(aes(x,y,color=type),size = 1.5) + labs(x="False positive rate", y="True positive rate") +
  ggtitle("ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",ela.auc.train,ela.auc.test)))


g1 = ggplot(ridge.errs) + geom_line(aes(x,y,color=type),size = 1.5) + labs(x="False positive rate", y="True positive rate") +
  ggtitle("Ridge ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",ridge.auc.train,ridge.auc.test)))

g2 = ggplot(lasso.errs) + geom_line(aes(x,y,color=type),size = 1.5)  +
  ggtitle("Lasso ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",lasso.auc.train,lasso.auc.test)))

g3 = ggplot(ela.errs) + geom_line(aes(x,y,color=type),size = 1.5) + 
  ggtitle(" Elastic-net ROC curve",(sprintf("train AUC=%.4f,test AUC =%0.4f",ela.auc.train,ela.auc.test)))

grid.arrange(g1, g2,g3, nrow = 1)



#top 5 words
mw                     =       5

#Ridge
obh                    =       order(ridge.beta.hat) 
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]

ridge.negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(ridge.negative.Words)


ridge.positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(ridge.positive.Words)



#Lasso
obh                    =       order(lasso.beta.hat) 
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


lasso.negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(lasso.negative.Words)


lasso.positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(lasso.positive.Words)

#Elastic-net

obh                    =       order(ela.beta.hat) 
word.index.negatives   =       obh[1:mw]
word.index.positives   =       obh[(p-(mw-1)):p]


ela.negative.Words         =       sapply(word.index.negatives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(ela.negative.Words)


ela.positive.Words         =       sapply(word.index.positives, function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})
cat(ela.positive.Words)
