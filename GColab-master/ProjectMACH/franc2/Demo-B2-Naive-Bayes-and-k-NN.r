# Year: 2016
# Title: R Stats
# Author: Jacob Cybulski
#
# Data: http://docs.health.vic.gov.au/docs/doc/2013-LGA-profiles-data
# Source: Australian Department of Health & Human Services
#
# Using Naive Bayes model for prediction
# We will try to predict people's wellbeing
# based on several socio-economic factors
# We will also compare NB model vs k-NN

# Define your working directory
# Ensure to use the forward / slash
setwd("C:/Users/pafil/Dropbox/Ubuntu - PFCB(CIN)/GColab-master/ProjectMACH/R")

# Install these packages first if working on your computer

# Includes pairs.panels
#install.packages("psych", dependencies = TRUE)
library("psych")

# Includes naiveBayes
#install.packages("e1071", dependencies = TRUE)
library("e1071")

# Includes confusionMatrix
#install.packages("caret", dependencies = TRUE)
library(caret)


# - Read all of the LGA profiles into a data frame
# - Get sample vectors (as follows) and explore them

lga.profile <- read.csv("dataTeste.csv")

# Select candidate variables (based on my personal hunch)
lga <- lga.profile$LGA # Names and the rest
crime <- lga.profile$SocEngag.11 # Was crimes per 1000 population
soc <- lga.profile$SocEngag.21 # % of time using social media 
sitting <- lga.profile$Health.25 # % of people sitting longer than 7 hours
safe <- lga.profile$SocEngag.13 # % of people feeling safe walking alone
education <- lga.profile$Edu.10 # % of people with higher education
highblood <- lga.profile$Medical.5 # % of people reporting high blood pressure
poorteeth <- lga.profile$Medical.31 # % of people with poor dental health
ad <- lga.profile$Injury.9 # Number of avoidable deaths
avoidabledeath <- (ad - min(ad))/(max(ad)-min(ad)) # Standardised avodable deaths
doctors <- lga.profile$HealthServices.4 # GPs per 1000, small number 0-1.5
wellbeing <- lga.profile$WellBeing.15 # % of people with adequate work / life balance

# Let us look again at the data
wb.raw <- data.frame(safe, sitting, highblood, poorteeth,
                     avoidabledeath, wellbeing)
pairs.panels(wb.raw)

# Define a categorical class variable
wellcl <- ifelse(wellbeing < 0.47, "Low", 
                 ifelse(wellbeing < 0.55, "Medium", "High"))

# After investigation the following vars were selected
# to remove those not promising and those highly inter-correlated vars
wb <- data.frame(safe, sitting, highblood, poorteeth,
                 avoidabledeath, wellcl)

# Split data into training and validation parts
# set random seed to some value so that results are consistent

set.seed(2016)

wb.size <- length(wellbeing)
wb.train.size <- round(wb.size * 0.7) # 70% for training
wb.validation.size <- wb.size - wb.train.size # The rest for testing
wb.train.idx <- sample(seq(1:wb.size), wb.train.size) # Indeces for training
wb.train.sample <- wb[wb.train.idx,]
wb.validation.sample <- wb[-wb.train.idx,]

##### Validate NB classifiers, check their performance and refine them

# Let's see the performance of all variables, excluding wellbeing
classf <- naiveBayes(
  subset(wb.train.sample, select = -wellcl), 
  wb.train.sample$wellcl, laplace=1)
classf
preds <- predict(classf, 
  subset(wb.validation.sample, select = -wellcl))
table(preds, wb.validation.sample$wellcl)
round(sum(preds == wb.validation.sample$wellcl, na.rm=TRUE) / 
        length(wb.validation.sample$wellcl), digits = 2)

# We could better report the performance using "caret" package
#   which has lots of other very useful functions in there
confusionMatrix(table(preds, wb.validation.sample$wellcl))


##### Many mislassifications, how about the following ideas:
#     - Change the mix of variables, e.g. replace sitting with soc
#     - Remove badly skewed vars, or transform them, e.g. safe
wb <- data.frame(highblood, poorteeth, avoidabledeath, wellcl)
#     - Redefine "wellcl", e.g.
wellcl <- ifelse(wellbeing < 0.49, "Low", ifelse(wellbeing < 0.57, "Medium", "High"))
#     - Add more vars and make a better selection of vars

##### And then compare NB with k-NN

# Create a simple k-NN classifier
# Install this package if working on your computer
# install.packages("class", dependencies = TRUE)
library(class)

preds <- knn(
  subset(wb.train.sample, select = -wellcl), 
  subset(wb.validation.sample, select = -wellcl),
  factor(wb.train.sample$wellcl),
  k = 3, prob=TRUE, use.all = TRUE)
confusionMatrix(table(preds, wb.validation.sample$wellcl))

preds <- knn(
  subset(wb.train.sample, select = -wellcl), 
  subset(wb.validation.sample, select = -wellcl),
  factor(wb.train.sample$wellcl),
  k = 10, prob=TRUE, use.all = TRUE)
confusionMatrix(table(preds, wb.validation.sample$wellcl))


### Our R code above involved a lot of luck
#   We split our data randomly into training and validation parts.
#   Play with the argument passed into set.seed(V) and see the results.
set.seed(2000)

### There are better ways of doing so, e.g. in the "caret" package.
#   More on this in the future.
library(caret)
set.seed(2016)
train_control <- trainControl(method="repeatedcv", number=10, repeats=9)
model <- train(wellcl~., data=wb, trControl=train_control, method="nb")
print(model)

##### Summary

# Which one is better?
# What does it all mean?
# Can we still improve the NB performance?
# How can we improve its performance?

# Let us look again at the data, does it fullfill NB requirements, i.e.
# variables are independent and they are normally distributed
wb.raw <- data.frame(safe, sitting, highblood, poorteeth,
                     avoidabledeath, wellbeing)

# We've done a variable selection based on their correlation, see data
# install.packages("psych", dependencies = TRUE)
library("psych")
pairs.panels(wb.raw)

##### If you got stuck earlier on...

### If you need to grasp variable selection from a data frame
#   Note that you can select or take away data frame vars

wb[1:5,]
subset(wb, select = -wellcl)[1:5,]
subset(wb, select = c(safe, highblood, wellcl))[1:5,]
subset(wb, select = c(-safe, -highblood))[1:5,]

### We can also consider categorical variables
#   Test their independence of wellcl
#   We can also consider some discrete vars
#   Use Chi-Sq to test their independence
remoteness <- lga.profile$Geo.7
osbirth1 <- lga.profile$Ancestry.8
osbirth2 <- lga.profile$Ancestry.10
oslang1 <- lga.profile$Ancestry.20
oslang2 <- lga.profile$Ancestry.22

# Got it? Thank you!
