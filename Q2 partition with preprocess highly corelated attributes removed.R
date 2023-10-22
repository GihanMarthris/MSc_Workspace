library(NbClust)
vehicle = read.csv("C:/Users/Gihan/Documents/Data mining course work/Q2 clustering/vehicle.csv")

vehicle$Class<-c(van=1,bus=2,opel=3,saab=4)[vehicle$Class]

head(vehicle)

# Preprocess - Dropping highly correlated columns
set.seed(50)

# load the library
library(caret)

#data(vehicle)

# Compute the correlation matrix for the first 19 columns of the data
correlationMatrix <- cor(vehicle[,1:19])
print(correlationMatrix)

# Find the highly correlated features using a correlation cutoff of 0.75
highycorrelated <- findCorrelation(correlationMatrix,cutoff = 0.75)
print(highycorrelated)

#Create an empty list to store the names of the highly correlated features
namelist = list()

# Extract the name of the column corresponding to the current index
for (x in highycorrelated) {
  #print(x)
  name <- colnames(vehicle[x])
  print(name[1])
  namelist <- append(namelist,name)
  #print(namelist)
}
#Print the list of highly correlated feature names
print(namelist)
# Remove the namelist
drop <- c(namelist)
print(drop)
#Print the dataset without highly correlated features
vehicle = vehicle[,!(names(vehicle) %in% drop)]
print(vehicle)


Classes = vehicle$Class
vehicle$class=NULL
#head(vehicle)
set.seed(26)
clusterNo=NbClust(vehicle,distance="euclidean", min.nc=2,max.nc=15,method="kmeans",index="all")
#clusterNo=NbClust(vehicle,distance="manhattan", min.nc=2,max.nc=15,method="kmeans",index="all")

#Calculate the within sum of squares for each k value
k=3:8
set.seed(55)
WSS=sapply(k,function(k){kmeans(vehicle[1:8],centers = k)$tot.withinss})
plot(k, WSS, type="l", xlab= "Number of k", ylab="Within sum of squares")

# Selecting the dataframe and scaling the data
str(vehicle)
data.train<-scale(vehicle[-1])
summary(data.train)

# Setting the parameter settings
set.seed(1500)
nc<-NbClust(data.train,
            min.nc = 2, max.nc = 10,
            method="kmeans")
table(nc$Best.n[1,])

# Plotting the bar chart
barplot(table(nc$Best.n[1,]),
        xlab="Number of Clusters",
        ylab="Number of criteria",
        main="Number of Clusters Chosen by 30 Criteria")

#k=3/Set the number of clusters to 3
#Set the seed for reproducibility
set.seed(1500)
#Perform k-means clustering on scaled data with 3 clusters
fit.km<-kmeans(data.train,3)
fit.km
#Display cluster centers
fit.km$centers
#Display number of observations in each cluster
fit.km$size

#Load fpc package
installed.packages("fpc")
library(fpc)
plotcluster(data.train, fit.km$cluster)

#Load the library
library(MASS)
# Plot parallel coordinates with colored clusters using k-means results
parcoord(data.train, fit.km$cluster)

#create a confusion matrix
confuseTable.km <- table(classes, fit.km$cluster)
confuseTable.km
installed.packages("flexclust")
library(flexclust)
randIndex(confuseTable.km)