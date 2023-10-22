#Loading the dataset
retail = read.csv("C:/Users/Gihan/Documents/Data mining course work/online_retail_II-2010-2011.csv",
                  header = TRUE, stringsAsFactors = FALSE)
print(retail)

#Inspect the dataset to check for missing values, data types, and other issues
str(retail)
summary(retail)

#Remove missing values from the dataset
data <- retail[complete.cases(retail),]

#Remove rows with negative quantities or unit prices
data <- retail[retail$Quantity > 0,]
data <- retail[retail$UnitPrice > 0,]


#dropping null values
retail <- na.omit(retail)

# Create a logical condition to identify the rows you want to remove
condition <- grepl("C", retail$Invoice)

# Use the condition to subset the data frame and remove the rows
retail <- retail[!condition, ]
print(retail)

# Create a logical condition to identify the rows you want to remove
condition <- grepl("DISCOUNT|POSTAGE", retail$Description, ignore.case = TRUE)

# Use the condition to subset the data frame and remove the rows
retail <- retail[!condition, ]


#creating  rows corresponding to transactions in the United Kingdom
retail <- retail[retail$Country == "United Kingdom", ]

print(retail)

#drop columns such as StockCode, Price, Country, InvoiceDate,  and Quantity
retail <- retail[, !(names(retail) %in% c("StockCode", "Price", "Country", "Quantity", "InvoiceDate"))]
print(retail)

summary(retail)

# load the arules package
library(arules)
library(arulesViz)
# convert the preprocessed data into a transaction format
transactions <- transactions(split(retail[,"Description"],retail[,"Invoice"]))
print(transactions)

# find association rules using Apriori algorithm
rules <- apriori(transactions, parameter = list(support = 0.025, confidence = 0.7))

# inspect the top 10 rules by confidence
inspect(head(sort(rules, by = "lift"), n = 150))

#Plotting the rules in  Scatter plot
plot(rules)

# convert the rules to a data frame
rules <- as(rules, "data.frame")

# display the rules as a table
View(rules)




