library(NbClust)
vehicle = read.csv("C:/Users/Gihan/Documents/Data mining course work/Q2 clustering/vehicle.csv")

vehicle$Class<-c(van=1,bus=2,opel=3,saab=4)[vehicle$Class]

head(vehicle)

# Preprocess - Dropping highly correlated columns
set.seed(50)

# load the library
library(caret)

correlationMatrix <- cor(vehicle[,1:19])
print(correlationMatrix)
highycorrelated <- findCorrelation(correlationMatrix,cutoff = 0.75)
print(highycorrelated)

namelist = list()

for (x in highycorrelated) {
  #print(x)
  name <- colnames(vehicle[x])
  print(name[1])
  namelist <- append(namelist,name)
  #print(namelist)
}
print(namelist)
drop <- c(namelist)
print(drop)
vehicle = vehicle[,!(names(vehicle) %in% drop)]
print(vehicle)

classes = vehicle$Class
vehicle$Class = NULL
data(vehicle)

control <- trainControl(method = "repeatedcv",number = 10,repeats = 3)

model <- train(vehicle, classes, method="lars2", preProcess="scale", 
               trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# remove less import columns
namelist = list("Max.L.Ra","Skew.maxis","Max.L.Rect","Samples","Pr.Axis.Ra")

drop <- c(namelist)
print(drop)
vehicle = vehicle[,!(names(vehicle) %in% drop)]
print(vehicle)




k=3:5
set.seed(55)
WSS=sapply(k,function(k){kmeans(vehicle[1:4],centers = k)$tot.withinss})
plot(k, WSS, type="l", xlab= "Number of k", ylab="Within sum of squares")

str(vehicle)
data.train<-scale(vehicle[-1])
summary(data.train)

set.seed(1500)
nc<-NbClust(data.train,
            min.nc = 2, max.nc = 10,
            method="kmeans")
table(nc$Best.n[1,])

barplot(table(nc$Best.n[1,]),
        xlab="Number of Clusters",
        ylab="Number of criteria",
        main="Number of Clusters Chosen by 30 Criteria")

set.seed(1500)
fit.km<-kmeans(data.train,3)
fit.km

fit.km$centers

fit.km$size

installed.packages("fpc")
library(fpc)
plotcluster(data.train, fit.km$cluster)

library(MASS)
parcoord(data.train, fit.km$cluster)

confuseTable.km <- table(classes, fit.km$cluster)
confuseTable.km
installed.packages("flexclust")
library(flexclust)
randIndex(confuseTable.km)

