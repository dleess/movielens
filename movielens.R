#########################################################
# Movie Recommending System
# Donghan Lee
# 2021-10-25
##############
# The codes
# 2.0. creating data, which is provided by Dr. Irizarry.
# 2.1. Data Analysis
# 3.1. Cross Validation
# Models
# 3.1.1. Navie estimate
# 3.1.2. Linear Model
# 3.1.2.1 Movie effect
# 3.1.2.2. User effect
# 3.1.2. Regularization
# 3.1.3. Matrix Factorizing
# 3.2. Final Hold-out Test
# 3.2.1. Regularization with Movie and User effects
# 3.2.2. Matrix Factorizing
# 4. Conclusion Summary
##########################################################
# 2.0. creating data, which is provided by Dr. Irizarry.
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, removed)

##########################################################
# 2.1. Data Analysis
##########################################################
# dimension of data
dim(movielens)

# rating distribution
movielens %>% group_by(rating) %>% summarize(n = n())


# finding how mamy unique userId and movieId
movielens %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

library(dplyr)
library(knitr)

#### userId 10-15's rating on the top 5 numbers of movies
# top 5 numbers of movie
keep <- movielens %>%
  count(movieId) %>%
  top_n(5) %>%
  pull(movieId)

# select userId 10-15 and show the rating
tab <- movielens %>%
  filter(userId %in% c(10:13)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)

tab %>% kable()

# showing sparseness of data using heat map of movieId and userId
# select unique 100 users
users <- sample(unique(movielens$userId), 100)
rafalib::mypar()

movielens %>% filter(userId %in% users) %>% # select only 100 unique users
  select(userId, movieId, rating) %>% # get userId, movieId, and rating
  mutate(rating = 1) %>% # for heat map, set rating to 1
  spread(movieId, rating) %>% # make data as row with userId and column with movieId with value of rating
  select(sample(ncol(.), 100)) %>% # choose 100 movieId
  as.matrix() %>% # make as matrix
  t(.) %>% # transpose the matrix. So, row is movieId, column is userId
  image(1:100, 1:100,. , xlab="Movies", ylab="Users") # heat map
# create grid for easy look
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# histogram of how the rate of the movie is provided. 
movielens %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# histogram of how users provide rating
movielens %>%
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")


# rating period calculation. lubridate origin is 1970-01-01.
library(lubridate)
# first date, last day, and duration of rating
tibble(first_date = date(as_datetime(min(movielens$timestamp), 
                                     origin="1970-01-01")),
       last_date = date(as_datetime(max(movielens$timestamp), 
                                    origin="1970-01-01"))) %>%
  mutate(duration = duration(max(movielens$timestamp)-min(movielens$timestamp)))

## how many movies came out in various years. lubridate origin is 1970-01-01. 
movielens %>% 
  mutate(year = year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(year))) %>%
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# relationship between rate of rating on movie and rating 
library(gam)
movielens %>%
  mutate(year = year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  group_by(movieId) %>%
  summarize(n = n(), years = 2011 - first(year),
            title = title[1],
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()

# how the rating is related with genres
# considering multiple genres in a movie
movielens %>% separate_rows(genres, sep = "\\|") %>% # genres could be multiple. First separate
  group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# clearing workspace
rm(movielens)

##########################################################
# 3.1. Cross Validation
# a. edx from 2.0. creating data is divided into train (90%) 
#    and test (10%) sets for cross-validation.
# b. RMSE function that will be the benchmark for all the models.
##########################################################
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

# create train index p = 0.9
train_index <- createDataPartition(y = edx$rating, times = 1,
                                   p = 0.9, list = FALSE)

train <- edx[train_index,]
temp <- edx[-train_index,]

# Make sure userId and movieId in validation set are also in edx set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# clean up the workspace
rm(train_index, temp, removed)

# Benchmark RMSE function
RMSE <- function(predictions, true_ratings){
  sqrt(mean((true_ratings - predictions)^2))
}

##########################################################
# Models
# 3.1.1. Navie estimate
# 3.1.2. Linear Model
# 3.1.2.1 Movie effect
# 3.1.2.2. User effect
# 3.1.2. Regularization
# 3.1.3. Matrix Factorizing
###########################################################
# 3.1.1. Navie estimate
# average (mu) of rating in train set
mu <- mean(train$rating)

# prediction; rating is expected to be average for all.
predictions <- rep(mu, nrow(test))

# rmse from average
naive_rmse <- RMSE(predictions, test$rating)

# storing the rmse to rmse_results
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

# print table for RMSE
rmse_results %>% kable()

# 3.1.2.1 Movie effect
# model: Y = mu + b_i + error ; where b_i for movieId
# rational: The contents of movie should be related to rating
b_i <- train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# prediction
predictions <- test %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# rmse calculation
model_bi_rmse <- RMSE(predictions, test$rating)

# storing the rmse to rmse_results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect",
                                     RMSE = model_bi_rmse ))
# print table for RMSE
rmse_results %>% kable()

# 3.1.2.2. User effect
# model Y = mu + b_i + b_u + error ; where b_u for userId
# rational: The taste of user should be related to rating
b_u <- train %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predictions
predictions <- test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# rmse calculation
model_bi_bu_rmse <- RMSE(predictions, test$rating)

# storing the rmse to rmse_results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_bi_bu_rmse ))
# print table for RMSE
rmse_results %>% kable()

# 3.1.2. Regularization
# Regularization of b_i and b_u.
#    a. b_i = sum(rating - mu)/(n()+lambda)
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predictions <- test %>%
    left_join(b_i, by='movieId') %>% 
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predictions, test$rating))
})

plot(lambdas,rmses, xlab = "b_i")

# get lambda for the lowest rmse
lambdas[which.min(rmses)]

model_reg_bi_rmse <- min(rmses)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regular Movie Effects Model",  
                                     RMSE = model_reg_bi_rmse ))
# print table for RMSE
rmse_results %>% kable()

#    b. b_i regularization (1) and b_u = sum(rating - b_i - mu)/(n()+l)
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predictions <- 
    test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predictions, test$rating))
})

plot(lambdas,rmses, xlab = "b_i & b_u")

# set lambda for the lowest rmse
lambdas[which.min(rmses)]

model_reg_bi_bu_rmse <- min(rmses)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie User Effect ",  
                                     RMSE = model_reg_bi_bu_rmse ))

# print table for RMSE
rmse_results %>% kable()

##########################################################
# 3.1.3. Matrix Factorizing
##########################################################
# recosystem
# this library is doing non-negative matrix factorizing
# model: Y = P*Q 
library(recosystem)
set.seed(1, sample.kind = "Rounding") # This is a randomized algorithm

####################################################################
# recosystem read only three columns.
# user_index needs to be user index, integer. userId for this data
# item_index needs to be item index, integer. movieId
# rating needs to be numeric. rating
# using data_memory function to include data
train_reco <- data_memory(user_index = train$userId, 
                          item_index = train$movieId,
                          rating = train$rating)

test_reco <- data_memory(user_index = test$userId, 
                         item_index = test$movieId,
                         rating = test$rating)

# Reco() function creates data files
rec <- Reco()

set.seed(1, sample.kind = "Rounding") # This is a randomized algorithm


# Actual calculation
set.seed(1, sample.kind = "Rounding") # This is a randomized algorithm
rec$train(train_reco, opts = c(nthread  = 4))

# predition from recosystem. For the return, out_memory() should be used.
predictions <- rec$predict(test_reco, out_memory())

# calculate RMSE
model_mat_fac_rmse <- RMSE(predictions, test$rating)

# stroring RMSE
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix Factorized",  
                                     RMSE = model_mat_fac_rmse ))

# print table for RMSE
rmse_results %>% kable()

# free some of workspace
rm(train, test, b_i, b_u, predictions, opts, lambda, lambdas, rec)

##########################################################
# 3.2. Final Hold-out Test
# using edx and validation sets
##########################################################

# 3.2.1. Regularization with Movie and User effects
# b_i regularization (1) and b_u = sum(rating - b_i - mu)/(n()+l)
# getting lambda for the lowest rmse for regularized bi and bu

# mu calculation
mu <- mean(edx$rating)

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predictions <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predictions, validation$rating))
})

plot(lambdas,rmses, xlab = "b_i & b_u")

# set lambda for the lowest rmse
lambda <- lambdas[which.min(rmses)]

model_reg_bi_bu_rmse <- min(rmses)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Regularized Movie User Effect ",  
                                     RMSE = model_reg_bi_bu_rmse ))

# print table for RMSE
rmse_results %>% kable()

# 3.2.2. Matrix Factorizing
# recosystem
# this library is doing non-negative matrix factorizing
# model: Y = P*Q 
library(recosystem)
set.seed(1, sample.kind = "Rounding") # This is a randomized algorithm

# data_memory function to store in recosystem
edx_reco <- data_memory(user_index = edx$userId, 
                        item_index = edx$movieId,
                        rating = edx$rating)

validation_reco <- data_memory(user_index = validation$userId, 
                               item_index = validation$movieId,
                               rating = validation$rating)

# Reco() function creates data files
rec <- Reco()

# Actual calculation
set.seed(1, sample.kind = "Rounding") # This is a randomized algorithm
rec$train(edx_reco, opts = c(nthread  = 4))

# predition from recosystem. For the return, out_memory() should be used.
predictions <- rec$predict(validation_reco, out_memory())

# store predictions for the movie recommendation
MF_pred <- predictions

# calculate RMSE
model_mat_fac_rmse <- RMSE(predictions, validation$rating)

# stroring RMSE
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Matrix Factorized",  
                                     RMSE = model_mat_fac_rmse ))

# print table for RMSE
rmse_results %>% kable()

##########################################################
# 4. Conclusion
##########################################################
# Summary table for RMSE
rmse_results %>% kable()




