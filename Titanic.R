########################################################################################################
############################## FORECASTING SURVIVAL RATES FOR TITANIC DATASET ##########################
#########################################################################################################
"""
Created on April 24,2019

@author: Naveen Balaraju
"""
#Loading dataset
data = read.csv("/Users/naveen/Desktop/SDM\ 2/Final\ Assignment/titanic/train.csv")
str(data)
names(data)

################################ FEATURE SELECTION &  DATA PREPROCESSING ###############################
# Here, I'm assuming Name of the person as the unique identity of an individual and have 
# selected features that are necessary for forecasting the survival rates. The Age variable
# contains 177 missing values.So, I've replaced missing values with the median value of Age i.e 28

data[["PassengerId"]]=NULL
data[["Ticket"]]=NULL
data[["Fare"]]=NULL
data[["Cabin"]]=NULL

# reordering the levels for Embarker varibale
data$Embarked=factor(data$Embarked,levels = c("C","Q","S"))
f =  which(is.na(data$Embarked))
data = data[-c(f),]

# re-naming the row names
row.names(data)=data$Name
data[["Name"]]=NULL

# Binning varible Age into "Chil" and "Adult"
ind = which(is.na(data$Age))
data[["Age"]]=replace(data$Age,c(ind),28)
data[["Age"]]=ifelse(data$Age<18,"Child","Adult")
data$Age=factor(data$Age,levels = c("Child","Adult"))

# Binning Survived variable
data$Survived=factor(data$Survived,levels= c(0,1))
data[["Survived"]]=ifelse(data$Survived==0,"Not Survived","Survived")
data$Survived=factor(data$Survived,levels= c("Not Survived","Survived"))

# Binning level of pclass variable
data$Pclass=factor(data$Pclass,levels = c(1,2,3))

# Binning Sibsp varible
data$SibSp=factor(data$SibSp,levels = c(1,0,3,4,2,5,8))

# Binning Parch variable
data$Parch=factor(data$Parch,levels = c(0,1,2,5,3,4,6))

str(data)

##################################### EXPLORATORY DATA ANALYSIS #########################################

# Analysis of Survived variable
quartz()
per1=round((table(data$Survived)/889)*100)
lab1=paste(names(table(data$Survived)),"[",per1,"%]")
pie(table(data$Survived),labels = lab1,main = "PIE CHART FOR SURVIVAL RATE,[0:Not Survived, 1:Survived]",
    col = c('green','blue'),cex=0.6)
# From the piechart, it shows that only 38% of the people aboard survived in Titanic sinking incident

# Analysis of pclass variable
quartz()
per2=round((table(data$Pclass)/889)*100)
lab2=paste(names(table(data$Pclass)),"[",per2,"%]")
pie(table(data$Pclass),labels = lab2,main = "PIE CHART FOR TICKET CLASS",
    col = c('yellow','blue','green'),cex=0.6)
# The piechart shows that there are 55% of people were in class 3, 21% in class 2 and 24% in class 1

# Analysis if variable sex
quartz()
per3=round((table(data$Sex)/889)*100)
lab3=paste(names(table(data$Sex)),"[",per3,"%]")
pie(table(data$Sex),labels = lab3,main = "PIE CHART FOR GENDER",
    col = c('yellow','blue'),cex=0.6)
# The data contains 65% male populatin and 35% female population

# Analysis of the Variable Age
# For the further analysis, I've considered all the people of age below 18 as "Children"
# and of age 18 and above as "Adults"
quartz()
per=round((table(data$Age)/889)*100)
lab=paste(names(table(data$Age)),"[",per,"%]")
pie(table(data$Age),labels = lab,main = "PIE CHART FOR AGE GROUP",
    col = c('yellow','green'),cex=0.6)

# The data contains 113 people with age below 18 i.e 13% of the population
# on board of Titanic are children and rest 87% are Adults

# Analysis of variable SibSp
quartz()
per4=round((table(data$SibSp)/889)*100)
lab4=paste(names(table(data$SibSp)),"[",per4,"%]")
pie(table(data$SibSp),labels = lab4,main = "PIE CHART FOR  NUMBER OF SIBLINGS/SPOUSES ABOARD THE TITANIC",
    col = rainbow(14),cex=0.6)
# The piechart shows that 68% of the people were travelling alone.While,23% were aboard with 1 sibling/Spouse
# and rest 9% were aboard the titanic with 2 siblings or more.



#Analysis of variable Parch
quartz()
per5=round((table(data$Parch)/889)*100)
lab5=paste(names(table(data$Parch)),"[",per5,"%]")
pie(table(data$Parch),labels = lab5,main = "PIE CHART FOR  NUMBER OF PARENTS/CHILDREN ABOARD THE TITANIC",
    col = rainbow(14),cex=0.6)
# The piechart shows that 76% of the people aboard the Titanic were travelling alone or with spouse.
# Whilst,13% aborad were with one parent/child and rest 9% were traveeling with more than one
# parents/children

# Analysis of variable Embarked
quartz()
per6=round((table(data$Embarked)/889)*100)
n=c('Cherbourg','Queenstown','Southampton')
lab6=paste(n,"[",per6,"%]")
pie(table(data$Embarked),labels = lab6,main = "PIE CHART FOR PORT OF EMBARKATION(C:Cherbourg,Q:Queenstown,S:Southampton)",
    col = c('green','yellow','blue'),cex=0.6)
# The piechart shows that 72% of the people had port of departure as Cherbourg,19% Queenstown and rest
# 9% had the port of departure as Southampton.

############################## BAYESIAN NETWORK FOR FORECASTING SURVIVAL RATES ###############################
# To find the optimal bayesian network,I've have used Max-min Hill climbing alogorithm  with "k2" as the score
library(bnlearn)
library(Rgraphviz)
library(gRain)

dag = hc(data,score = "k2")
quartz()
graphviz.plot(dag,highlight = list(nodes,arcs)) #optimal DAG structure
nodes(dag)
arcs(dag)


bn_cpt= bn.fit(dag,data = data,method = "bayes")
bn_cpt$Survived

#Now, given the evidence we have to absorb into the network
junction = compile(as.grain(bn_cpt))
junction = propagate(junction)
summary(junction)
jun_evd=setEvidence(junction,nodes = c("Pclass","Age","Sex"),states = c("1","Adult","female"))
jun_evd$cptlist$Survived
jun_evd_m=setEvidence(junction,nodes = c("Pclass","Age","Sex"),states = c("3","Adult","male"))
jun_evd_m$cptlist$Survived

 
########################################## CONCLUSIONS ###############################################
# 1. Female and adult's belonging to pclass-1 and pclass-2 survived with the probability 
#    of 0.97 and 0.90 respectively.While,Female adults pclass-3 did not survive
# 2. Most of the children and female seated in the pclass1,pclass2 and pclass3 survived
#    with the probability of 0.87,0.99,0.54 respectively.While,the probability of
#    child and male in pclass1 and pclass2 survived with probability of 0.98 and
#    0.81 respectively,while children who are male beloning to pclass 3 did not survive.
# 3. The results also shows that Adult males did not survive as their probability of being
#    dead is high for all the three classes:
#                    pclass1 death rate - 65.7%
#                    pclass2 death rate - 91.7%  
#                    pclass3 death rate - 87.8%
#    Overall, most of the adult male did not survive
# 4. All the above findings gives an intuition that most of the Women and children were
#    evacuated first.Although, child(male) and adult(female) belonging to pclass did not survive.
# 5. Female adult belonging to pclass1 - SURVIVED
# 6. Male adult belonging to pclass3- DID NOT SURVIVE








