# ------------------------------------------------------------------------------------------
# Data gathering
# ------------------------------------------------------------------------------------------
  setwd("D:/RSA/00 General/DS Course - BGU Nov 2015/") # Set here your working directory
  german.credit = read.csv("German.Credit.csv", header = TRUE, sep = ",")
  attach(german.credit)

# ------------------------------------------------------------------------------------------
# Exploratory data analysis
# ------------------------------------------------------------------------------------------
  library(Hmisc, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
  describe(german.credit)

# Contingency Tables
library(descr, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
crosstab(class, account_balance, max.width = 1, plot = TRUE, digits=1, 
         prop.r=TRUE, prop.c=TRUE, prop.t=FALSE, prop.chisq=FALSE, chisq=TRUE)

# Descriptive statistics of the numerical predictors
# Histogram
  brksCredit <- seq(0, 80, 10)   # Bins for a nice looking histogram
  hist(duration, breaks=brksCredit, 
       xlab = "Loan Period [Months]", ylab = "Frequency", main = " ")       

# Boxplot
boxplot(duration, bty="n",xlab = "Credit Month") 


# ------------------------------------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------------------------------------
# Training and testing data set
set.seed(1017)
frac_train = 0.8
indx = sample(1:nrow(german.credit), size = frac_train*nrow(german.credit))
traindata <- german.credit[indx,] 
testdata <- german.credit[-indx,] 

# Set Misclassification costs
L_BG = 1
L_GB = 5



# ------------------------------------------------------------------------------------------
# Modeling
# ------------------------------------------------------------------------------------------
library( ROCR , warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
label.trn = ifelse(traindata$class == "good" , 1 , 0)
label = ifelse(testdata$class == "good" , 1 , 0)

# -----------------------------------------------------------------------------
#  Logistic regression
# -----------------------------------------------------------------------------
# model
glm.fit=glm(class~.,data=traindata,family=binomial)
summary(glm.fit)

# ROC
glm.prob = predict(glm.fit,newdata=testdata,type="response")
pred.glm <- prediction( glm.prob, label )
ROC.glm <- performance(pred.glm, "tpr", "fpr")
plot(ROC.glm, col='green', lty = 1, lwd=2, 
     main = 'Receiver Operator Characteristic (ROC) Curve')
abline(a = 0, b = 1, lty=2)
legend("bottomright", c('GLM','RNDM'), col = c('green','black'),
       lty = c(1,2), lwd=c(2,1))

# AUC
auc = performance(pred.glm, "auc")
auc = auc@y.values[[1]]
auc.glm = round( auc , digits = 3)
auc.glm

# Optimal threshold
# FOR ALEX: I don't understand what MARCELO tried to do here.... :(
# ---------------------------------->>>>>>>>>>>>>>
glm.prob.trn <- predict(glm.fit, newdata=traindata, type="response")
pred.trn.glm <- prediction( glm.prob.trn , label.trn  )
loss <- performance(pred.trn.glm , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion matrix
pred = rep("bad" , nrow(testdata))
pred[ glm.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS
loss.glm = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.glm
# <<<<<<<<<<<<<<<<----------------------------------

# -----------------------------------------------------------------------------
#  Naive Bayes Classification
# -----------------------------------------------------------------------------
# model
library(e1071, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)        
# naiveBayes and SVM
nb.fit <- naiveBayes(class ~. , traindata)
summary(nb.fit)
nb.fit

# ROC_NB
nb.prob <- predict (nb.fit , testdata , type = "raw")
nb.prob <- as.data.frame(nb.prob)[2]
pred.nb <- prediction( nb.prob , label  )
ROC.nb <- performance(pred.nb, "tpr", "fpr")
plot(ROC.nb, col = 'red', lty = 1, 
     main = 'Receiver Operator Characteristic (ROC) Curve',lwd=2);
abline(a = 0, b = 1, lty=2)
legend("bottomright", c('NB','RNDM'), col = c('red','black'),
       lty = c(1,2), lwd=c(2,1))

# AUC_NB
auc = performance(pred.nb, "auc")
auc = auc@y.values[[1]]
auc.nb = round( auc , digits = 3)
auc.nb

# optimal threshold
nb.prob.trn <- predict(nb.fit , traindata , type = "raw")[,2]
pred.trn.nb <- prediction( nb.prob.trn , label.trn  )
loss <- performance(pred.trn.nb , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# confusion matrix
pred = rep("bad" , nrow(testdata))
pred[ nb.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS
loss.nb = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.nb


# -----------------------------------------------------------------------------
#  Linear discriminant analysis
# -----------------------------------------------------------------------------
# model_LDA
library(MASS, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
lda.fit <- lda(class ~ . , data = traindata )
lda.fit

# ROC_LDA
lda.pred <- predict(lda.fit, newdata=testdata, method="predictive")
lda.prob = lda.pred$posterior[,2]
label = ifelse(testdata$class == "good" , 1 , 0)
pred.lda <- prediction( lda.prob, label )
ROC.lda <- performance(pred.lda, "tpr", "fpr")
plot(ROC.lda, col = 'blue', lty = 1 , lwd = 2, 
     main = 'Receiver Operator Characteristic (ROC) Curve')
abline(a = 0, b = 1, lty=2)
legend("bottomright", c('LDA','RNDM'), col = c('blue','black'),
       lty = c(1,2), lwd=c(2,1))


# AUC
auc = performance(pred.lda, "auc")
auc = auc@y.values[[1]]
auc.lda = round( auc , digits = 3)
auc.lda

# Optimal Threshold
lda.pred.trn <- predict(lda.fit, newdata = traindata, method="predictive")
lda.prob.trn = lda.pred.trn$posterior[,2]
label.trn = ifelse(traindata$class == "good" , 1 , 0)
pred.trn.lda <- prediction( lda.prob.trn , label.trn  )
loss <- performance(pred.trn.lda , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion Matrix_LDA
pred = rep("bad" , nrow(testdata))
pred[ lda.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS_LDA
loss.lda = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.lda

# -----------------------------------------------------------------------------
#  Quadratic discrimminant analysis
# -----------------------------------------------------------------------------
# model_QDA
qda.fit <- lda(class ~ . , data = traindata )
qda.fit

# ROC
qda.pred <- predict(qda.fit, newdata=testdata, method="predictive")
qda.prob = qda.pred$posterior[,2]
label = ifelse(testdata$class == "good" , 1 , 0)
pred.qda <- prediction( qda.prob, label )
ROC.qda <- performance(pred.qda, "tpr", "fpr")
plot(ROC.qda, col = 'orange', lty = 1, lwd = 2, 
     main = 'Receiver Operator Characteristic (ROC) Curve')
abline(a = 0, b = 1, lty = 2)
legend("bottomright", c('QDA','RNDM'), col = c('orange','black'),
       lty = c(1,2), lwd=c(2,1))


# AUC
auc = performance(pred.qda, "auc")
auc = auc@y.values[[1]]
auc.qda = round( auc , digits = 3)
auc.qda 

# Optimal Threshold
qda.pred.trn <- predict(qda.fit, newdata = traindata, method="predictive")
qda.prob.trn = qda.pred.trn$posterior[,2]
label.trn = ifelse(traindata$class == "good" , 1 , 0)
pred.trn.qda <- prediction( qda.prob.trn , label.trn  )
loss <- performance(pred.trn.qda , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion Matrix
pred = rep("bad" , nrow(testdata))
pred[ qda.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS_QDA
loss.qda = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.qda


# -----------------------------------------------------------------------------
#  Supporting Vector machine (SVM)
# -----------------------------------------------------------------------------
# Model
svm.fit <- svm(class ~ ., data = traindata, probability = TRUE)
print(svm.fit )
summary(svm.fit)

# ROC
pred.svm <- predict(svm.fit, testdata, decision.values = TRUE, probability = TRUE)
svm.prob = attr(pred.svm, "probabilities")[,2]
label = ifelse(testdata$class == "good" , 1 , 0)
pred.svm <- prediction( svm.prob, label )
ROC.svm <- performance(pred.svm, "tpr", "fpr")
plot(ROC.svm, col = 'navyblue', lty = 1, lwd = 2, 
     main = 'Receiver Operator Characteristic (ROC) Curve')
abline(a = 0, b = 1, lty = 2 )
legend("bottomright", c('SVM','RNDM'), col = c('navyblue','black'),
       lty = c(1,2), lwd=c(2,1))

# AUC
auc = performance(pred.svm, "auc")
auc = auc@y.values[[1]]
auc.svm = round( auc , digits = 3)
auc.svm

# Optimal Threshold
pred.trn.svm <- predict(svm.fit, traindata, decision.values = TRUE, probability = TRUE)
svm.prob.trn = attr(pred.trn.svm, "probabilities")[,2]
label.trn = ifelse(traindata$class == "good" , 1 , 0)
pred.trn.svm <- prediction( svm.prob.trn , label.trn  )
loss <- performance(pred.trn.svm , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion Matrix
pred = rep("bad" , nrow(testdata))
pred[ svm.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS_SVM
loss.svm = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.svm

# -----------------------------------------------------------------------------
#  Recursive partitioning trees
# -----------------------------------------------------------------------------
# model_Tree
library(rpart, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
tree.fit <- rpart( class ~. , data = traindata , 
                   control=rpart.control(minsplit=20,cp=0,maxdepth=3))
printcp(tree.fit)

# Tree visualization
library(rattle, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
fancyRpartPlot(tree.fit)

# ROC
tree.class <- predict(tree.fit , type = 'class', testdata)
tree.prob <- predict(tree.fit, type = 'prob', testdata)[,2]
pred.tree <- prediction(tree.prob, testdata$class)
ROC.tree <- performance(pred.tree, "tpr", "fpr")
plot(ROC.tree, col='pink', lty = 1, lwd = 2, main = 'Receiver Operator Characteristic (ROC) Curve')
abline(a = 0, b = 1, lty = 2)
legend("bottomright", c('TREE','RNDM'), col = c('pink','black'),
       lty = c(1,2), lwd=c(2,1))

# AUC
auc = performance(pred.tree, "auc")
auc = auc@y.values[[1]]
auc.tree = round( auc , digits = 3)
auc.tree

# Optimal Threshold
tree.prob.trn <- predict(tree.fit, type = 'prob', traindata)[,2]
label.trn = ifelse(traindata$class == "good" , 1 , 0)
pred.trn.tree <- prediction( tree.prob.trn , label.trn  )
loss <- performance(pred.trn.tree , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion Matrix
pred = rep("bad" , nrow(testdata))
pred[ tree.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS_Tree
loss.tree = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.tree


# -----------------------------------------------------------------------------
#  Random forest
# -----------------------------------------------------------------------------
# model
library(randomForest, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE)
raf.fit <- randomForest( class ~. , data = traindata,
                         importance = TRUE, proximity = TRUE, 
                         ntree = 500, keep.forest = TRUE)

# plot variable importance
varImpPlot(raf.fit, cex=0.5, main = "Random Forests Variable Importance")

# ROC_RaF
raf.prob <- predict(raf.fit , testdata ,type = 'prob')[,2]
pred.raf <- prediction(raf.prob , testdata$class)
ROC.raf <- performance(pred.raf, "tpr", "fpr")
plot(ROC.raf, col='skyblue4', lty = 1 , lwd = 2, 
     main = 'Receiver Operator Characteristic (ROC) Curve')
abline( a = 0 , b = 1 , lty = 2)
legend("bottomright", c('RAF','RNDM'), col = c('skyblue4','black'),
       lty = c(1,2), lwd=c(2,1))

# AUC
auc = performance(pred.raf, "auc")
auc = auc@y.values[[1]]
auc.raf = round( auc , digits = 3)
auc.raf

# Optimal Threshold
raf.prob.trn <- predict(raf.fit , traindata ,type = 'prob')[,2]
label.trn = ifelse(traindata$class == "good" , 1 , 0)
pred.trn.raf <- prediction( raf.prob.trn , label.trn  )
loss <- performance(pred.trn.raf , "cost", cost.fp=L_GB, cost.fn=L_BG)
opt_th = loss@x.values[[1]][ which( loss@y.values[[1]] == min(loss@y.values[[1]]))]
opt_th

# Confusion Matrix
pred = rep("bad" , nrow(testdata))
pred[ raf.prob > opt_th ] = "good"
conf_mat = table(pred,testdata$class)
conf_mat

# LOSS
loss.raf = L_GB*conf_mat[2,1]+L_BG*conf_mat[1,2]
loss.raf

