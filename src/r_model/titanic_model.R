# Titanic survival prediction - R version
# Plan: load data -> poke around -> clean up a bit -> logistic regression (glm) -> check accuracy -> predict test.csv

library(readr)
library(dplyr)

# where the csv files live. inside docker this is /app/data (set in the Dockerfile),
# if you run the script locally from src/r_model it looks in ../data
data_dir <- Sys.getenv("DATA_DIR", unset = file.path("..", "data"))
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")

# ---------------------------------------------------------------
cat("=== Step 1: loading the data ===\n")
cat("Looking for data in:", data_dir, "\n")

if (!(file.exists(train_path) && file.exists(test_path))) {
  cat("Oops, can't find train.csv and/or test.csv.\n")
  cat("Download them from Kaggle and drop them in src/data (the README shows how).\n")
  quit(status = 1)
}

train_df <- read_csv(train_path, show_col_types = FALSE)
test_df <- read_csv(test_path, show_col_types = FALSE)
cat("train.csv has", nrow(train_df), "rows and", ncol(train_df), "columns\n")
cat("test.csv has", nrow(test_df), "rows and", ncol(test_df), "columns\n")
cat("\nFirst few rows of train.csv:\n")
print(head(train_df))

# ---------------------------------------------------------------
cat("\n=== Step 2: quick look at the data ===\n")
missing <- colSums(is.na(train_df))
cat("Columns with missing values in train.csv:\n")
print(missing[missing > 0])

cat("\nOverall survival rate:", round(mean(train_df$Survived) * 100, 1), "%\n")
cat("\nSurvival rate by sex:\n")
train_df %>% group_by(Sex) %>% summarise(survival_rate = round(mean(Survived), 3)) %>% print()
cat("\nSurvival rate by passenger class:\n")
train_df %>% group_by(Pclass) %>% summarise(survival_rate = round(mean(Survived), 3)) %>% print()
cat("\n-> Women and first class folks did way better. Makes sense, 'women and children first'.\n")

# ---------------------------------------------------------------
cat("\n=== Step 3: splitting train.csv ===\n")
# test.csv from Kaggle has NO Survived column, so I can't check accuracy on it.
# So I hold out 20% of train.csv and use that as my "real" test accuracy.
set.seed(42)
train_idx <- sample(seq_len(nrow(train_df)), size = floor(0.8 * nrow(train_df)))
train_raw <- train_df[train_idx, ]
holdout_raw <- train_df[-train_idx, ]
cat("Training part:", nrow(train_raw), "rows, hold-out part:", nrow(holdout_raw), "rows\n")

# ---------------------------------------------------------------
cat("\n=== Step 4: cleaning up + making new features ===\n")
# fill missing stuff using ONLY the training part (so the hold-out stays "unseen")
age_fill <- median(train_raw$Age, na.rm = TRUE)
fare_fill <- median(train_raw$Fare, na.rm = TRUE)
embarked_fill <- names(sort(table(train_raw$Embarked), decreasing = TRUE))[1]
cat("Missing Age -> filling with median age:", age_fill, "\n")
cat("Missing Fare -> filling with median fare:", round(fare_fill, 2), "\n")
cat("Missing Embarked -> filling with most common port:", embarked_fill, "\n")

clean_up <- function(df) {
  df %>%
    mutate(
      Age = ifelse(is.na(Age), age_fill, Age),
      Fare = ifelse(is.na(Fare), fare_fill, Fare),
      Embarked = ifelse(is.na(Embarked), embarked_fill, Embarked),
      FamilySize = SibSp + Parch + 1,           # me + siblings/spouse + parents/kids
      IsAlone = as.integer(FamilySize == 1),    # traveling solo?
      IsChild = as.integer(Age < 12),           # kids under 12
      # make categories into factors with fixed levels so train and test match
      Pclass = factor(Pclass, levels = c(1, 2, 3)),
      Sex = factor(Sex, levels = c("female", "male")),
      Embarked = factor(Embarked, levels = c("C", "Q", "S"))
    )
}

train_clean <- clean_up(train_raw)
cat("New features added: FamilySize, IsAlone, IsChild\n")
cat("Missing values left in the features:",
    sum(is.na(train_clean[, c("Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "IsChild")])), "\n")

# ---------------------------------------------------------------
cat("\n=== Step 5: training the logistic regression ===\n")
model <- glm(
  Survived ~ Pclass + Sex + Age + Fare + Embarked + FamilySize + IsAlone + IsChild,
  data = train_clean,
  family = binomial
)
cat("Done training. Coefficients (positive = more likely to survive):\n")
print(round(coef(summary(model)), 3))

# ---------------------------------------------------------------
cat("\n=== Step 6: accuracy on the training set ===\n")
train_pred <- as.integer(predict(model, newdata = train_clean, type = "response") > 0.5)
train_acc <- mean(train_pred == train_clean$Survived)
cat("Training accuracy:", round(train_acc * 100, 1), "%\n")

# ---------------------------------------------------------------
cat("\n=== Step 7: predicting on test.csv ===\n")
test_clean <- clean_up(test_df)
test_pred <- as.integer(predict(model, newdata = test_clean, type = "response") > 0.5)
cat("Predicted", length(test_pred), "passengers:", sum(test_pred == 1), "survive,", sum(test_pred == 0), "don't\n")
cat("Predicted survival rate on test.csv:", round(mean(test_pred) * 100, 1), "%\n")

out_path <- file.path(data_dir, "predictions_r.csv")
write_csv(data.frame(PassengerId = test_df$PassengerId, Survived = test_pred), out_path)
cat("Saved predictions to", out_path, "(Kaggle submission format)\n")

# ---------------------------------------------------------------
cat("\n=== Step 8: accuracy on the test side ===\n")
holdout_clean <- clean_up(holdout_raw)
holdout_pred <- as.integer(predict(model, newdata = holdout_clean, type = "response") > 0.5)
holdout_acc <- mean(holdout_pred == holdout_clean$Survived)
baseline <- 1 - mean(holdout_clean$Survived)
cat("(test.csv has no answers, so this uses the 20% hold-out from train.csv)\n")
cat("Hold-out accuracy:", round(holdout_acc * 100, 1), "%\n")
cat("For comparison, just guessing 'everyone dies' gets:", round(baseline * 100, 1), "%\n")
cat("\nAll done!\n")
