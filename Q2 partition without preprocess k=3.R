# loading the dataset
vehicle = read.csv("C:/Users/Gihan/Documents/Data mining course work/Q2 clustering/vehicle.csv")

# Converting the vehicle models into numbers
vehicle$Class<-c(van=1,bus=2,opel=3,saab=4)[vehicle$Class]

# Print the dataset
head(vehicle)

# Loading the NbClust library
library(NbClust)

classes = vehicle$Class
vehicle$class=NULL
#head(vehicle)
set.seed(55)
clusterNo=NbClust(vehicle,distance="euclidean", min.nc=2,max.nc=15,method="kmeans",index="all")

#Define a range of k values 
k=2:19

#Set the seed for reproducibility
set.seed(55)

#Calculate the within sum of squares for each k value using k-means clustering
WSS=sapply(k,function(k){kmeans(vehicle[1:19],centers = k)$tot.withinss})

#Plot the results
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

#Plot data points with colored clusters using k-means results
plotcluster(data.train, fit.km$cluster)

#Load the library
library(MASS)

# Plot parallel coordinates with colored clusters using k-means results
parcoord(data.train, fit.km$cluster)

#create a confusion matrix
confuseTable.km <- table(classes, fit.km$cluster)
confuseTable.km

# Load the flexclust package
installed.packages("flexclust")
library(flexclust)

# Calculate the Adjusted Rand Index(ARI) 
randIndex(confuseTable.km)

