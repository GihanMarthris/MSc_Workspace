# Load the dataset and preprocess
vehicle <- read.csv("C:/Users/Gihan/Documents/Data mining course work/Q2 clustering/vehicle.csv")
vehicle <- vehicle[, -c(1, ncol(vehicle))]  # Remove "Samples" and "Class" columns
vehicle_norm <- scale(vehicle[, -1])  # Normalize the data
z_scores <- apply(vehicle_norm, 1, function(x) max(abs(scale(x))))  # Identify and remove outliers using z-score
vehicle_clean <- vehicle_norm[z_scores < 1,]
vehicle_reduced <- vehicle_norm[, c("D.Circ", "Scat.Ra", "Sc.Var.maxis")]  # Reduce selected columns
library(colorspace)

# Calculate the correlation matrix
cor_matrix <- cor(vehicle_reduced)
cor_matrix

# Create a correlation plot using the corrplot function
library(corrplot)
corrplot(cor_matrix, method = "pie", type = "upper", tl.col = "black", tl.srt = 45)

# Perform hierarchical clustering using Single linkage method
d_vehicle <- dist(vehicle_reduced)
hc_vehicle <- hclust(d_vehicle, method = "single")

# Create a dendrogram object and modify it for single method
library(dendextend)
dend <- as.dendrogram(hc_vehicle)
dend <- rotate(dend, 1:nrow(vehicle_reduced))  # Order the dendrogram to match the order of the rows in vehicle_reduced
dend <- color_branches(dend, k = 6)  # Color the branches based on three clusters
labels_colors(dend) <- rainbow_hcl(6)[sort_levels_values(cutree(hc_vehicle, k = 6))]  # Color the labels based on the clusters
labels(dend) <- paste("(", labels(dend), ")", sep = "")  # Add parentheses to the labels
dend <- hang.dendrogram(dend, hang_height = 0.1)  # Adjust the height of the dendrogram
dend <- set(dend, "labels_cex", 0.5)  # Reduce the size of the labels

# Plot the dendrogram with a legend
par(mar = c(3, 3, 3, 7))
plot(dend,
     main = "Clustered Vehicle data set using Single Linkage Method",
     horiz = FALSE, nodePar = list(cex = .007))
legend("topleft", legend = c("Cluster 1", "Cluster 2", "Cluster 3","Cluster 4","Cluster 5","Cluster 6"), fill = rainbow_hcl(6))

# Perform hierarchical clustering using Average linkage method
d_vehicle_avg <- dist(vehicle_reduced)
hc_vehicle_avg <- hclust(d_vehicle_avg, method = "average")
rect.hclust(hc_vehicle_avg, k=4, border="black")


# Create a dendrogram object and modify it for average linkage method
dend_avg <- as.dendrogram(hc_vehicle_avg)
dend_avg <- rotate(dend_avg, 1:nrow(vehicle_reduced))  # Order the dendrogram to match the order of the rows in vehicle_reduced
dend_avg <- color_branches(dend_avg, k = 4)  # Color the branches based on three clusters
labels_colors(dend_avg) <- rainbow_hcl(4)[sort_levels_values(cutree(hc_vehicle_avg, k = 4))]  # Color the labels based on the clusters
labels(dend_avg) <- paste("(", labels(dend_avg), ")", sep = "")  # Add parentheses to the labels
dend_avg <- hang.dendrogram(dend_avg, hang_height = 0.1)  # Adjust the height of the dendrogram
dend_avg <- set(dend_avg, "labels_cex", 0.5)  # Reduce the size of the labels

# Plot the dendrogram with a legend for average linkage method
par(mar = c(3, 3, 3, 7))
plot(dend_avg,
     main = "Clustered Vehicle data set using Average Linkage Method",
     horiz = FALSE, nodePar = list(cex = .007))
legend("topleft", legend = c("Cluster 1", "Cluster 2", "Cluster 3","Cluster 4"), fill = rainbow_hcl(4))

# Perform hierarchical clustering using Complete linkage method
d_vehicle_comp <- dist(vehicle_reduced)
hc_vehicle_comp <- hclust(d_vehicle_comp, method = "complete")

# Create a dendrogram object and modify it for complete linkage method
dend_comp <- as.dendrogram(hc_vehicle_comp)
dend_comp <- rotate(dend_comp, 1:nrow(vehicle_reduced))  # Order the dendrogram to match the order of the rows in vehicle_reduced
dend_comp <- color_branches(dend_comp, k = 3)  # Color the branches based on three clusters
labels_colors(dend_comp) <- rainbow_hcl(3)[sort_levels_values(cutree(hc_vehicle_comp, k = 3))]  # Color the labels based on the clusters
labels(dend_comp) <- paste("(", labels(dend_comp), ")", sep = "")  # Add parentheses to the labels
dend_comp <- hang.dendrogram(dend_comp, hang_height = 0.1)  # Adjust the height of the dendrogram
dend_comp <- set(dend_comp, "labels_cex", 0.5)  # Reduce the size of the labels

# Plot the dendrogram with a legend for complete linkage method
par(mar = c(3, 3, 3, 7))
plot(dend_comp,
     main = "Clustered Vehicle data set using Complete Linkage Method",
     horiz = FALSE, nodePar = list(cex = .007))
legend("topleft", legend = c("Cluster 1", "Cluster 2", "Cluster 3"), fill = rainbow_hcl(3))
                                                                                   
library(dendextend)
# Perform hierarchical clustering using Single linkage method
d_vehicle <- dist(vehicle_reduced)
hc_vehicle <- hclust(d_vehicle, method = "single")

# Calculate the cophenetic correlation for the single linkage method
cc_single <- cophenetic(hc_vehicle)

# Perform hierarchical clustering using Average linkage method
d_vehicle_avg <- dist(vehicle_reduced)
hc_vehicle_avg <- hclust(d_vehicle_avg, method = "average")

# Calculate the cophenetic correlation for the average linkage method
cc_avg <- cophenetic(hc_vehicle_avg)

# Perform hierarchical clustering using Complete linkage method
d_vehicle_comp <- dist(vehicle_reduced)
hc_vehicle_comp <- hclust(d_vehicle_comp, method = "complete")

# Calculate the cophenetic correlation for the complete linkage method
cc_comp <- cophenetic(hc_vehicle_comp)

# Create a dendlist object containing the three dendrograms
dendlist <- dendlist(dend, dend_avg, dend_comp)

# Calculate the correlation between the cophenetic distances of the three dendrograms
correlation <- cor.dendlist(dendlist, method = "cophenetic")
correlation

# Plot the correlation matrix
corrplot(correlation, method = "pie", type = "upper", tl.col = "black", tl.srt = 45)
