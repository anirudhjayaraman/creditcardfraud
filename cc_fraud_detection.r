require(tidyverse)  # for data operations
require(reticulate)  # for using Python
require(corrplot)  # for plotting correlations
require(caret)  # for Confusion Matrix
require(e1071)  # for Confusion Matrix
require(ModelMetrics)  # for Confusion Matrix
require(caTools)  # for sampling
require(reshape2)  # for reshaping data for plots eg, multivariable boxplots
require(gridExtra)  # for plotting multiple plots together
require(rpart)  # For CART models
require(randomForest) # For Random Forest
require(ROCR)  # For plotting ROC Curve
require(party)  # Conditional inference trees

parent_dir <- "~/Documents/git/Kaggle/cc_fraud_detection/"
plots_dir <- paste0(parent_dir, "plots/")

# Read the data ===============================================================
cc_data <- read_csv(paste0(parent_dir, "creditcard.csv"))
cc_data %>% glimpse()

# General EDA =================================================================

# Check that specific plotting directories exist
dir.create(plots_dir, showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_01"), showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_02"), showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_03"), showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_04"), showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_05"), showWarnings = FALSE)
dir.create(paste0(plots_dir, "plots_06"), showWarnings = FALSE)


# Generate and save histograms using all data
for (var in names(cc_data)[2:29]) {
  plt <- cc_data %>% ggplot() +
    geom_histogram(mapping = aes_string(x = var), bins = 1e3)
  ggsave(plt, filename = paste0(plots_dir, "plots_01/", var, ".png"))
}


# Generate and save histograms using only fraud cases
for (var in names(cc_data)[2:29]) {
  plt <- cc_data %>% filter(Class == 1) %>%
    ggplot() +
    geom_histogram(mapping = aes_string(x = var), bins = 50)
  ggsave(plt, filename = paste0(plots_dir, "plots_02/", var, ".png"))
}


# Plot and save histograms using fraud and non-fraud cases simultaneously
for (var in names(cc_data)[2:29]) {
  plt1 <- cc_data %>% filter(Class == 1) %>%
    ggplot() +
    geom_histogram(mapping = aes_string(x = var), bins = 50) +
    ggtitle("Fraud Cases")
  
  plt2 <- cc_data %>% filter(Class == 0) %>%
    ggplot() +
    geom_histogram(mapping = aes_string(x = var), bins = 1e3) +
    ggtitle("Non-fraud Cases")
  
  g <- arrangeGrob(plt1, plt2, nrow = 2)
  ggsave(g, filename = paste0(plots_dir, "plots_03/", var, ".png"))
}

# Box plots of all variables indicate very thin interquartile ranges:

plt1 <- cc_data %>%
  select(V1:V28, Class) %>% melt(id.var = "Class") %>%
  ggplot() + geom_boxplot(mapping = aes(x = variable, y = value, fill = Class))

# The above plot indicates that outliers skew the distributions of each
# variable

plt2 <- cc_data %>% filter(Class == 1) %>%
  select(V1:V28, Class) %>% melt(id.var = "Class") %>%
  ggplot() + geom_boxplot(mapping = aes(x = variable, y = value))

# Density / Histogram plot of the Amount Variable indicate outliers
# skewing the data

hist_plt <- cc_data %>% ggplot() + 
  geom_histogram(mapping = aes(x = Amount), bins = 1e3)

density_plt <- cc_data %>% ggplot() + 
  geom_density(mapping = aes(x = Amount))

# Plotting individual variables with time

for (var in names(cc_data)[2:30]) {
  plt <- cc_data %>% 
    mutate(Class = factor(Class)) %>% 
    filter(Class == 0) %>% 
    ggplot() + 
    geom_point(mapping = aes_string(x = "Time", 
                                    y = var),
               alpha = 0.1,
               color = "steelblue") +
    annotate("point", 
             cc_data$Time[cc_data$Class == 1],
             as.numeric(unlist(cc_data[,var]))[cc_data$Class == 1],
             color = "red4") + 
    theme_minimal()  
  
  ggsave(plt, filename = paste0(plots_dir, "plots_06/", var, ".png"))
}

# Outlier Analysis ============================================================
# What fraction of the data falls outside IQR

outlier_bounds <- function(x){
  res <- rep(0,2)
  res[1] <- quantile(x, probs = 0.25) - 1.5 * IQR(x)
  res[2] <- quantile(x, probs = 0.75) + 1.5 * IQR(x)
  return(res)
}

outlier_indices <- function(x){
  bounds <- outlier_bounds(x)
  idx <- (x < bounds[1]) | (x > bounds[2])
  return(which(idx))
}

# Correlation Plot
# The variables are output of PCA, so correlations among variables is expected
# to be low
correlations <- cor(cc_data %>% select(V1:V28, Amount))
corrplot(correlations, number.cex = .9, 
         method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

# Building a Model ===========================================================

# Convert `Class` to a factor variable
cc_data$Class <- as.factor(cc_data$Class)

# PRINT: Percent of `Class` vales that are 0 or 1
100 * summary(cc_data$Class) / nrow(cc_data)


# Split the Data
set.seed(42)
spl <- sample.split(Y = cc_data$Class, SplitRatio = 0.6)
train <- cc_data[spl,]
test <- cc_data[!spl,]
rm(spl)


# Model 1 - Logistic Regression ===============================================

mdl_01 <- glm(Class ~ ., family = 'binomial', data = train)
mdl_01_sum <- summary(mdl_01)
mdl_01_preds <- predict(modl_01, newdata = test, type = "response")

thresholds <- seq(0.05, 0.95, 0.01)
opt_spec <- opt_F1 <- rep(0, length(thresholds))

for (i in 1:length(thresholds)) {
  fraud_threshold <- thresholds[i]
  test$predicted <- factor(as.numeric(mdl_01_preds > fraud_threshold))
  cM <- caret::confusionMatrix(table(test$predicted,
                                     test$Class))
  opt_spec[i] <- cM$byClass['Specificity'] %>% as.numeric()
  opt_F1[i] <- cM$byClass['F1'] %>% as.numeric()
}

fraud_threshold <- last(thresholds[which(max(opt_spec) == opt_spec)])
test$predicted <- factor(as.numeric(mdl_01_preds > fraud_threshold))
cM1 <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM1)


# Model 2 - Logistic Regression using significant variables ===================

# significant variables based on different training data sets (30)
signif_vars <- character()

for (i in 1:30) {
  set.seed(i)
  sp <- sample.split(Y = cc_data$Class, SplitRatio = 0.6)
  tr <- cc_data[sp,]
  mdl <- glm(Class ~ ., family = 'binomial', data = tr)
  mdl_summary <- summary(mdl_01)
  significant_idx <- which(mdl_summary$coefficients[,"Pr(>|z|)"] < 0.05)
  significant_vars <- rownames(mdl_summary$coefficients)[significant_idx]
  signif_vars <- unique(c(signif_vars, significant_vars))
}

print(signif_vars[signif_vars != "(Intercept)"])
# [1] "V8"     "V27"    "V28"    "Amount"

# Prepare train and test sets
set.seed(42)
spl <- sample.split(Y = cc_data$Class, SplitRatio = 0.6)
train <- cc_data[spl,]
test <- cc_data[!spl,]
mdl_02 <- glm(Class ~ Amount + V8 + V27 + V28, 
                family = 'binomial', data = train)
mdl_02_sum <- summary(mdl_02)
mdl_02_preds <- predict(mdl_02, newdata = test, type = "response")

opt_spec <- opt_F1 <- rep(0, length(thresholds))

for (i in 1:length(thresholds)) {
  fraud_threshold <- thresholds[i]
  test$predicted <- factor(as.numeric(mdl_02_preds > fraud_threshold))
  cM <- caret::confusionMatrix(table(test$predicted,
                                     test$Class))
  opt_spec[i] <- cM$byClass['Specificity'] %>% as.numeric()
  opt_F1[i] <- cM$byClass['F1'] %>% as.numeric()
}

fraud_threshold <- last(thresholds[which(max(opt_spec) == opt_spec)])
test$predicted <- factor(as.numeric(mdl_02_preds > fraud_threshold))
cM2 <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM2)

# Model 3 - Logistic Regression on stepwise features addition =================

mdl_03 <- step(glm(formula = Class ~ ., family = 'binomial', data = train),
                direction = 'forward', trace = FALSE, k = 5)
mdl_03_sum <- summary(mdl_03)
mdl_03_preds <- predict(mdl_03, newdata = test, type = "response")

test$predicted <- factor(as.numeric(mdl_03_preds > fraud_threshold))
cM <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM)

# Refit on significant variables
mdl_03a <- step(glm(Class ~ V8 + V14 + V27 + V28 + Amount, data = train, 
               family = "binomial"))
mdl_03a_sum <- summary(mdl_03a)
mdl_03a_preds <- predict(mdl_03a, newdata = test, type = "response")
test$predicted <- factor(as.numeric(mdl_03a_preds > fraud_threshold))
cM <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM)

# Respecify model
mdl_03 <- glm(Class ~ V14 + V27 + V28, data = train, family = 'binomial')
mdl_03_sum <- summary(mdl_03)
mdl_03_preds <- predict(mdl_03, newdata = test, type = "response")
test$predicted <- factor(as.numeric(mdl_03_preds > fraud_threshold))
cM3 <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM3)


# Model 4 -  CART =============================================================
set.seed(42)
mdl_04 <- rpart(Class ~ ., data = train, method = 'class')
mdl_04_summary <- summary(mdl_04)
mdl_04_preds <- predict(mdl_04, newdata = test)

test$predicted <- factor(as.numeric(mdl_04_preds[,2] > fraud_threshold))
cM4 <- caret::confusionMatrix(table(test$predicted, test$Class))
print(cM4)

rocr_pred <- prediction(mdl_04_preds[,2], test$Class)
perf <- performance(rocr_pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

printcp(mdl_04)
plotcp(mdl_04)

par(mfrow = c(1,2)); rsq.rpart(mdl_04); par(mfrow = c(1,1))

# Model 5 - Random Forest =====================================================

mdl_05 <- randomForest(Class ~ ., data = train)  # My Mac can't handle this


