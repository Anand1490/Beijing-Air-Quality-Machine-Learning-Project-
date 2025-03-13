#### Anandhan Manoharan
#### Final Project

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# List of files to load
files = [
    'PRSA_Data_Changping_20130301-20170228.csv',
    'PRSA_Data_Shunyi_20130301-20170228.csv',
    'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    'PRSA_Data_Dingling_20130301-20170228.csv',
    'PRSA_Data_Dongsi_20130301-20170228.csv',
    'PRSA_Data_Guanyuan_20130301-20170228.csv',
    'PRSA_Data_Gucheng_20130301-20170228.csv',
    'PRSA_Data_Huairou_20130301-20170228.csv',
    'PRSA_Data_Nongzhanguan_20130301-20170228.csv',
    'PRSA_Data_Tiantan_20130301-20170228.csv',
    'PRSA_Data_Wanliu_20130301-20170228.csv',
    'PRSA_Data_Wanshouxigong_20130301-20170228.csv'
]

# Initialize an empty list to store dataframes
df_list = []

# Read and clean datasets in batches
for file in files:
    # Load each dataset in a chunk
    chunk = pd.read_csv(file)

    # Handle missing values for this chunk
    chunk = chunk.fillna(chunk.mean())

    # Append the cleaned chunk to the list
    df_list.append(chunk)

# Concatenate the list of cleaned dataframes
merged_data_cleaned = pd.concat(df_list, ignore_index=True)

# Check for any remaining missing values
missing_values = merged_data_cleaned.isnull().sum()

# Display missing values count and final merged dataframe
print("Missing values after cleaning:")
print(missing_values)
# Set pandas to display all columns
pd.set_option('display.max_columns', None)
print(merged_data_cleaned)



#### EDA ###########################################################
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
merged_data_cleaned[pollutants].hist(bins=30, figsize=(15, 10))
plt.show()


# Create a 'season' column based on the month
merged_data_cleaned['season'] = merged_data_cleaned['month'].apply(lambda x:
    'Winter' if x in [12, 1, 2] else
    'Spring' if x in [3, 4, 5] else
    'Summer' if x in [6, 7, 8] else
    'Autumn')


plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='NO2', data=merged_data_cleaned)
plt.title('NO2 Levels Across Seasons')
plt.show()


### Scatterplot ###
# Scatterplot of NO2 vs WSPM
sns.scatterplot(x='TEMP', y='NO2', data=merged_data_cleaned)
plt.title('Scatterplot of NO2 vs Temperature (TEMP)')
plt.show()




#########################Random Forest ########################################


merged_data_cleaned = merged_data_cleaned.drop('season', axis=1)




# Select features and target variable
features = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
target = 'NO2'

# Define X (features) and y (target)
X = merged_data_cleaned[features]
y = merged_data_cleaned[target]

# Train a Random Forest Regressor model with parallelism and other optimizations
rf = RandomForestRegressor(
    n_estimators=100,          # Number of trees (you can experiment with this)
    random_state=42,           # For reproducibility
    n_jobs=-1,                 # Use all available CPU cores for parallelism
    max_depth=10,              # Limit tree depth to reduce complexity and speed up training
    min_samples_split=10,      # Increase min samples required to split nodes
    warm_start=True            # Reuse previous models if needed
)

# Fit the model
rf.fit(X, y)

# Get feature importance from the trained model
importance = rf.feature_importances_

# Create a DataFrame to visualize the importance of each feature
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.show()

# Print out the sorted feature importance
print(importance_df)




######## Discretization ###########
# Define a threshold for NO2 safety (let's use 10 µg/m³ as an example)
threshold_no2 = 10

# Label the values based on the threshold
merged_data_cleaned['NO2_safety'] = merged_data_cleaned['NO2'].apply(
    lambda x: 'safe' if x <= threshold_no2 else 'unsafe'
)

# Now 'NO2_safe' will contain 'safe' or 'unsafe' instead of 1 or 0
print(merged_data_cleaned.head())

############### Subset the Data set ############
# Define the list of features you want to keep
subset_columns = ['NO2', 'TEMP', 'PRES', 'DEWP', 'wd', 'WSPM', 'station', 'NO2_safety']

# Create a subset of the dataset
subset_data = merged_data_cleaned[subset_columns]

# Display the first few rows of the subset
print(subset_data.head())


############## VIF ##################################################
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add constant to the features to account for the intercept
X = subset_data[['NO2', 'TEMP', 'PRES', 'DEWP', 'WSPM']]

# Calculate the VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF
print(vif_data)


####### Second Subset of the Data ##################################
# Define the list of features you want to keep
subset_columns2 = ['NO2', 'TEMP', 'DEWP', 'wd', 'WSPM', 'station', 'NO2_safety']

# Create a subset of the dataset
subset_data2 = merged_data_cleaned[subset_columns2]

# Display the first few rows of the subset
print(subset_data2.head())

############## VIF ##################################################
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add constant to the features to account for the intercept
X = subset_data2[['NO2', 'TEMP', 'DEWP', 'WSPM']]

# Calculate the VIF for each feature
vif_data2 = pd.DataFrame()
vif_data2['Feature'] = X.columns
vif_data2['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF
print(vif_data2)


############## Correlation Matrix ##################################
numerical_columns = subset_data2.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_columns.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Sample Pearson Correlation Coefficients Heatmap')
plt.show()


###### Balanced and imbalanced data #################################
# Check the distribution of the target variable
class_distribution = subset_data2['NO2_safety'].value_counts()

print("Class Distribution:")
print(class_distribution)

# Visualize the class distribution
import matplotlib.pyplot as plt

class_distribution.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution of NO2_safety')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
###### Balancing Data ####
import imblearn

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
# Define the feature matrix (X) and target vector (y)
X = subset_data2.drop('NO2_safety', axis=1)
y = subset_data2['NO2_safety']

# Perform random over-sampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Check the new class distribution

print("Class distribution after over-sampling:", Counter(y_resampled))

### Show plot ####
# Visualize the balanced class distribution
balanced_class_distribution = pd.Series(y_resampled).value_counts()

# Plot the distribution after balancing
plt.figure(figsize=(8, 6))
balanced_class_distribution.plot(kind='bar', color=['orange', 'skyblue'])
plt.title('Class Distribution of NO2_safety After Balancing')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=balanced_class_distribution.index, rotation=0)
plt.show()


########################################################################################
########################### Regression #################################################
########################################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from itertools import combinations

# get variables to use for x and y
X_regs = subset_data2[['TEMP', 'DEWP', 'WSPM']]
y_regs = subset_data2['NO2']

print(X_regs.info())
print(y_regs.info())


# Add a constant for the intercept
X_with_const = sm.add_constant(X_regs)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_with_const, y_regs, test_size=0.2, random_state=42)

# Reset indices to avoid misalignment
X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

# Train the regression model using statsmodels
air_NO2_model = sm.OLS(y_train, X_train).fit()

# Make predictions
y_pred_train = air_NO2_model.predict(X_train)
y_pred_test = air_NO2_model.predict(X_test)

# Calculate R-squared, Adjusted R-squared, AIC, BIC, and MSE
r_squared = air_NO2_model.rsquared
adj_r_squared = air_NO2_model.rsquared_adj
aic = air_NO2_model.aic
bic = air_NO2_model.bic
mse = mean_squared_error(y_test, y_pred_test)

# Display the predicted values
print("Predicted NO2 values:")
print(y_pred_test)

# Display the results
results = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Value': [r_squared, adj_r_squared, aic, bic, mse]
})
print(results)

# Display the summary for T-tests and F-test
print(air_NO2_model.summary())


# Display confidence intervals
confidence_intervals = air_NO2_model.conf_int()
confidence_intervals.columns = ['Lower Bound', 'Upper Bound']
print(confidence_intervals)



# function for stepwise regression
def stepwise_regression(X, y, initial_features=[], criterion='AIC'):

    remaining_features = list(X.columns)
    selected_features = list(initial_features)
    best_features = None
    best_model = None
    best_criterion = float('inf')

    while remaining_features:
        candidate_models = []

        # Add one feature at a time
        for feature in remaining_features:
            model_features = selected_features + [feature]
            X_selected = sm.add_constant(X[model_features])  # Add constant
            model = OLS(y, X_selected).fit()
            crit_value = model.aic if criterion == 'AIC' else model.bic
            candidate_models.append((model, model_features, crit_value))

        # Find the best candidate model
        candidate_models.sort(key=lambda x: x[2])  # Sort by criterion
        best_candidate_model, best_candidate_features, candidate_criterion = candidate_models[0]

        # Stop if the criterion does not improve
        if candidate_criterion < best_criterion:
            best_criterion = candidate_criterion
            best_model = best_candidate_model
            best_features = best_candidate_features
            selected_features = best_candidate_features[1:]  # Remove constant
            remaining_features = [f for f in remaining_features if f not in selected_features]
        else:
            break

    return best_model, best_features


# run stepwise regression
stepwise_model, stepwise_features = stepwise_regression(X_regs, y_regs, criterion='AIC')

# Display the final model summary
print("Selected Features:", stepwise_features)
print(stepwise_model.summary())

# Calculate adjusted R-squared
adjusted_r_squared = stepwise_model.rsquared_adj
print("Adjusted R-squared:", adjusted_r_squared)


# Visualize the train, test, and predicted values
plt.figure(figsize=(12, 6))

# Plot train data
plt.plot(range(len(y_train)), y_train, label="Train Data", color='blue', marker='o', linestyle='-', alpha=0.7)

# Plot test data
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label="Test Data", color='green', marker='x', linestyle='-', alpha=0.7)

# Plot predictions on the test set
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred_test, label="Predicted Data", color='red', marker='s', linestyle='--', alpha=0.7)

# Add labels, legend, and title
plt.title("Train, Test, and Predicted NO2 Levels")
plt.xlabel("Sample Index")
plt.ylabel("NO2 Level")
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()


####################################################################################
########################### Classification Analysis #################################
####################################################################################

################# Decision Tree Classifier ##########################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sampling the dataset
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)
X_sample = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y_sample = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})  # Convert to numeric for classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

# 1. Decision Tree with Pre-pruning
dt_prepruned = DecisionTreeClassifier(
    criterion='gini', max_depth=5, min_samples_split=10, random_state=42
)
dt_prepruned.fit(X_train, y_train)

# 2. Post-pruning with Cost Complexity Pruning
path = dt_prepruned.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Exclude the last alpha (results in an empty tree)

# Train trees with different alphas
trees = []
for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# alpha with the highest cross-validated score
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tree_scores = [cross_val_score(tree, X_train, y_train, cv=kfold).mean() for tree in trees]
optimal_alpha = ccp_alphas[np.argmax(tree_scores)]

# Train final tree with the best alpha
dt_postpruned = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
dt_postpruned.fit(X_train, y_train)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0, optimal_alpha]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=kfold,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_tree = grid_search.best_estimator_

# 4. Evaluation Metrics
def evaluate_model(model, X_eval, y_eval, title="Confusion Matrix"):
    # Predictions and probabilities
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_eval, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(cm)

    # Metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fscore = 2 * (precision * recall) / (precision + recall)
    roc_auc = roc_auc_score(y_eval, y_proba)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F-score: {fscore:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")



    # ROC Curve with AUC
    fpr, tpr, _ = roc_curve(y_eval, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue', lw=2)
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue', label="AUC Area")
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="approximate line")  # Diagonal line
    plt.title("ROC Curve with AUC for Decision Tree")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


# Evaluate the best tree
print("Evaluation for the Best Decision Tree")
evaluate_model(best_tree, X_test, y_test)

# Stratified K-Fold Cross-Validation for Best Tree
cv_scores = cross_val_score(best_tree, X_train, y_train, cv=kfold, scoring='roc_auc')
print(f"Stratified K-Fold ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.2f}, Std: {cv_scores.std():.2f}")

###################### Logistic Regression ###############################################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample the dataset
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)

# Prepare your data that was sampled by selecting features and target variable
X_sample = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]  # Select relevant features
y_sample = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})  # Convert target to numeric

#  Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Set up Logistic Regression
log_reg = LogisticRegression(max_iter=1000)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Solvers
}

# Grid search with Stratified K-Fold cross-validation
grid_search = GridSearchCV(log_reg, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
grid_search.fit(X_train, y_train)

# best model from GridSearchCV
best_log_reg = grid_search.best_estimator_

# predictions
y_pred = best_log_reg.predict(X_test)
y_pred_prob = best_log_reg.predict_proba(X_test)[:, 1]  # Probability for ROC/AUC

print("Evaluate Logistic Regression")
# Evaluate performance metrics
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Precision, Recall, Specificity, F-Score
precision = precision_score(y_test, y_pred, pos_label=1)  # 'unsafe' is now 1
recall = recall_score(y_test, y_pred, pos_label=1)  # 'unsafe' is now 1
f_score = f1_score(y_test, y_pred, pos_label=1)  # 'unsafe' is now 1
specificity = recall_score(y_test, y_pred, pos_label=0)  # Specificity is recall for the 'safe' class (0)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)  # 'unsafe' is now 1
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Display metrics
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-Score: {f_score:.2f}")
print(f"AUC: {roc_auc:.2f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, tpr, color='blue', alpha=0.2)  # Shading the AUC area
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Cross-validation
cross_val_scores = cross_val_score(best_log_reg, X_sample, y_sample, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Average Cross-Validation Accuracy: {np.mean(cross_val_scores):.2f}")

################################# KNN ##################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Prepare Data
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)
X_sample = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y_sample = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})  # Convert target to numeric

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find Optimum K using the Elbow Method
error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_range, error_rates, marker='o', linestyle='-', color='blue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Train and Evaluate the Model with Optimal K
optimal_k = error_rates.index(min(error_rates)) + 1
print(f"Optimal K: {optimal_k}")

knn_best = KNeighborsClassifier(n_neighbors=optimal_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)
y_pred_prob_best = knn_best.predict_proba(X_test_scaled)[:, 1]

print("Evaluation for KNN")
# Metrics and Visualizations
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
print(conf_matrix)

# Precision, Recall (Sensitivity), Specificity, F-Score
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
specificity = recall_score(y_test, y_pred_best, pos_label=0)
f_score = f1_score(y_test, y_pred_best)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_best)
roc_auc = roc_auc_score(y_test, y_pred_prob_best)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, tpr, color='lightblue', alpha=0.5)  # Shade the AUC curve
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve for KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Display Metrics
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-Score: {f_score:.2f}")
print(f"AUC: {roc_auc:.2f}")

print("Evaluation of KNN: ")
# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(knn_best, X_sample, y_sample, cv=skf, scoring='accuracy')

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

################################## SVM ################################################################################
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Prepare Data
subset_data2_sampled = subset_data2.sample(n=500, random_state=42)
X_sample = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y_sample = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})  # Convert target to numeric

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Function to Train and Evaluate SVM
def evaluate_svm(kernel_type, degree=None, gamma=None, coef0=None, C=1.0):
    print(f"\n--- SVM with {kernel_type} kernel ---")


    svm_model = SVC(kernel=kernel_type, degree=degree if degree else 3, gamma=gamma if gamma else 'scale',
                    coef0=coef0 if coef0 else 0, C=C, probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predictions and probabilities
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_prob = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for ROC and AUC

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f_score = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F-Score: {f_score:.2f}")
    print(f"AUC: {roc_auc:.2f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.fill_between(fpr, tpr, color='lightblue', alpha=0.5)  # Shade the AUC curve
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'ROC Curve ({kernel_type} kernel)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    print("Evaluate SVM: ")
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=3)
    cv_scores = cross_val_score(svm_model, X_sample, y_sample, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")


# Hyperparameter tuning using GridSearchCV for different kernels
def tune_svm(kernel_type):
    if kernel_type == 'linear':
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
        }
    elif kernel_type == 'poly':
        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'coef0': [0, 1, 5]
        }
    elif kernel_type == 'rbf':
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    else:
        raise ValueError("Invalid kernel type. Choose 'linear', 'poly', or 'rbf'.")

    #  SVM model
    svm_model = SVC(kernel=kernel_type, probability=True, random_state=42)

    #  GridSearchCV
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')

    # Fit GridSearchCV
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and score
    print(f"\nBest parameters found by GridSearchCV for {kernel_type} kernel:")
    print(grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    evaluate_svm(kernel_type=kernel_type, C=grid_search.best_params_['C'],
                 degree=grid_search.best_params_.get('degree', 3),
                 gamma=grid_search.best_params_.get('gamma', 'scale'),
                 coef0=grid_search.best_params_.get('coef0', 0))


# Tune and evaluate the SVM with different kernels

# Linear Kernel
tune_svm(kernel_type='linear')

# Polynomial Kernel
tune_svm(kernel_type='poly')

# Radial Basis Function (RBF) Kernel
tune_svm(kernel_type='rbf')

############################################ Naive Bayes ################################################################
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Prepare the data
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)
X = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the parameter grid for Naive Bayes
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

#  Naive Bayes classifier
nb = GaussianNB()

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_nb = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# Make predictions on the test set
y_pred = best_nb.predict(X_test)

# Evaluate the model
print("Evaluate Naive Bayes: ")
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, Recall, F-score
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f_score = f1_score(y_test, y_pred, pos_label=1)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {f_score}")

# Specificity (True Negative Rate)
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Specificity: {specificity}")

# ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, best_nb.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
print("AUC: ", roc_auc)
# Plot ROC Curve
# Plot ROC Curve with shading for AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.fill_between(fpr, tpr, color='blue', alpha=0.2)  # Shade the area under the ROC curve
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve with AUC Shading for Naive Bayes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation performance
cv_results = grid_search.cv_results_
print("Grid Search CV results:")
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"Accuracy: {mean_score:.4f} for {params}")

############################## Random Forest #########################################################################


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Sample data
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)
X = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#  Random Forest Classifier
rf = RandomForestClassifier()

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}


# Parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5]
}

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5]}, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

# Stacking Model with base models
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('svc', SVC(probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_best_model = grid_search_rf.best_estimator_
y_pred_rf = rf_best_model.predict(X_test)

# Evaluate Gradient Boosting
gb_best_model = grid_search_gb.best_estimator_
y_pred_gb = gb_best_model.predict(X_test)

# Evaluate Stacking
y_pred_stacking = stacking_model.predict(X_test)

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)

# Confusion Matrix for Gradient Boosting
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
print("Gradient Boosting Confusion Matrix:")
print(conf_matrix_gb)

# Confusion Matrix for Stacking
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)
print("Stacking Confusion Matrix:")
print(conf_matrix_stacking)

# Precision, Recall, F-score for Random Forest
precision_rf = precision_score(y_test, y_pred_rf, pos_label=1)
recall_rf = recall_score(y_test, y_pred_rf, pos_label=1)
f_score_rf = f1_score(y_test, y_pred_rf, pos_label=1)
# Specificity for Random Forest
TN_rf, FP_rf, FN_rf, TP_rf = conf_matrix_rf.ravel()
specificity_rf = TN_rf / (TN_rf + FP_rf)
print(f"Random Forest - Precision: {precision_rf}, Recall: {recall_rf}, F-score: {f_score_rf}, Specificity: {specificity_rf}")

# Precision, Recall, F-score for Gradient Boosting
precision_gb = precision_score(y_test, y_pred_gb, pos_label=1)
recall_gb = recall_score(y_test, y_pred_gb, pos_label=1)
f_score_gb = f1_score(y_test, y_pred_gb, pos_label=1)
# Specificity for Gradient Boosting
TN_gb, FP_gb, FN_gb, TP_gb = conf_matrix_gb.ravel()
specificity_gb = TN_gb / (TN_gb + FP_gb)
print(f"Gradient Boosting - Precision: {precision_gb}, Recall: {recall_gb}, F-score: {f_score_gb}, Specificity: {specificity_gb}")

# Precision, Recall, F-score for Stacking
precision_stacking = precision_score(y_test, y_pred_stacking, pos_label=1)
recall_stacking = recall_score(y_test, y_pred_stacking, pos_label=1)
f_score_stacking = f1_score(y_test, y_pred_stacking, pos_label=1)
# Specificity for Stacking
TN_stacking, FP_stacking, FN_stacking, TP_stacking = conf_matrix_stacking.ravel()
specificity_stacking = TN_stacking / (TN_stacking + FP_stacking)
print(f"Stacking - Precision: {precision_stacking}, Recall: {recall_stacking}, F-score: {f_score_stacking}, Specificity: {specificity_stacking}")

# ROC and AUC for Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_best_model.predict_proba(X_test)[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
print("AUC for Random Forest: ")
print(roc_auc_rf)
# ROC and AUC for Gradient Boosting
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, gb_best_model.predict_proba(X_test)[:,1])
roc_auc_gb = auc(fpr_gb, tpr_gb)

# ROC and AUC for Stacking
fpr_stacking, tpr_stacking, thresholds_stacking = roc_curve(y_test, stacking_model.predict_proba(X_test)[:,1])
roc_auc_stacking = auc(fpr_stacking, tpr_stacking)

# Plot ROC Curve for Random Forest
plt.figure(figsize=(10, 7))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest AUC = {roc_auc_rf:.2f}')
plt.fill_between(fpr_rf, tpr_rf, alpha=0.2, color='blue')  # AUC shading
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# classification report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# classification report for Gradient Boosting
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# classification report for Stacking
print("Stacking Classification Report:")
print(classification_report(y_test, y_pred_stacking))


################################ Neural Networks #######################################################################
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Sample data
subset_data2_sampled = subset_data2.sample(n=10000, random_state=42)
X = subset_data2_sampled[['TEMP', 'DEWP', 'WSPM']]
y = subset_data2_sampled['NO2_safety'].map({'safe': 0, 'unsafe': 1})

# Initialize StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize Multi-layer Perceptron (Neural Network)
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)

# Perform Stratified K-fold cross-validation for the Neural Network (MLP)
print("\nEvaluating Multi-layer Perceptron (MLP) with Stratified K-fold Cross Validation:")

# Cross-validation scores (accuracy)
accuracy_scores = cross_val_score(mlp_model, X, y, cv=cv, scoring='accuracy')
print(f"Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")

# Cross-validation scores (precision, recall, f1-score)
precision_scores = cross_val_score(mlp_model, X, y, cv=cv, scoring='precision')
recall_scores = cross_val_score(mlp_model, X, y, cv=cv, scoring='recall')
f1_scores = cross_val_score(mlp_model, X, y, cv=cv, scoring='f1')

print(f"Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std():.4f})")
print(f"Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")
print(f"F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")

# Fit the model on the entire dataset
mlp_model.fit(X, y)
y_pred = mlp_model.predict(X)

# Confusion Matrix
conf_matrix_mlp = confusion_matrix(y, y_pred)
print("Confusion Matrix for Multi-layer Perceptron (MLP):")
print(conf_matrix_mlp)

# Extract TN, FP, FN, TP from the confusion matrix
tn, fp, fn, tp = conf_matrix_mlp.ravel()

# Calculate specificity
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

# ROC Curve and AUC for Multi-layer Perceptron
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y, mlp_model.predict_proba(X)[:, 1])
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
print("AUC: ", roc_auc_mlp)

# Plot ROC Curve for Multi-layer Perceptron
plt.figure(figsize=(10, 7))
plt.plot(fpr_mlp, tpr_mlp, color='purple', label=f'MLP AUC = {roc_auc_mlp:.2f}')
plt.fill_between(fpr_mlp, tpr_mlp, alpha=0.2, color='purple')  # AUC shading
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve for MLP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Display classification report for MLP
print("Classification Report for Multi-layer Perceptron (MLP):")
print(classification_report(y, y_pred))


##### Combined ROC Curve ###################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Sample data (assuming you have X_train, X_test, y_train, y_test ready)
# X_train, X_test, y_train, y_test already defined

# Initialize classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),  # SVM needs probability=True for predict_proba
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# Colors for the ROC curves
colors = {
    "Decision Tree": "blue",
    "Logistic Regression": "green",
    "KNN": "orange",
    "SVM": "red",
    "Naive Bayes": "purple",
    "Random Forest": "brown",
    "Neural Network": "pink"
}

# Plot ROC curves
plt.figure(figsize=(12, 8))

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC = {roc_auc:.2f})")
    plt.fill_between(fpr, tpr, alpha=0.1, color=colors[name])

# Plot diagonal line for random guessing
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

# Add labels, title, and legend
plt.title("ROC Curve Comparison of Multiple Classifiers")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

######################################################################################
############ Phase 4: Clustering and Association #####################################
######################################################################################

############################# K Means ################################################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Down sample the data for faster run time
down_sample_data = subset_data2.sample(frac=0.05, random_state=42)  # Use 5% of the data for faster processing
X_down = down_sample_data[['NO2', 'TEMP', 'DEWP', 'WSPM']]


#  range of k values to test for silhouette analysis
silhouette_scores = []
inertia_values = []
k_range = range(2, 10)  # Trying values of k from 2 to 10

#fit KMeans models for different k and calculate silhouette scores
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++',max_iter=100, random_state=42)
    kmeans.fit(X_down)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_down, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

    # Calculate inertia (within-cluster variation)
    inertia_values.append(kmeans.inertia_)

# Plot Silhouette Scores and Inertia
plt.figure(figsize=(12, 6))

# Plot Silhouette Scores
plt.subplot(1, 2, 1)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Analysis for K Selection')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

# Plot Inertia (Within-cluster variation)
plt.subplot(1, 2, 2)
plt.plot(k_range, inertia_values, marker='o')
plt.title('Within-cluster Variation (Inertia)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.tight_layout()
plt.show()

# Choose the best k based on silhouette score and fit the final KMeans model
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters (k): {best_k}")

# Fit the final KMeans model with the best k
kmeans_final = KMeans(n_clusters=best_k, init='k-means++',max_iter=100, random_state=42)
kmeans_final.fit(X_down)

# Cluster labels
down_sample_data['KMeans_Labels'] = kmeans_final.labels_

optimal_silhouette_score = max(silhouette_scores)
print(f"Silhouette Score for optimal k ({best_k}): {optimal_silhouette_score:.3f}")

######################## DBSCAN ##################################################################
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

down_sample_data = subset_data2.sample(frac=0.05, random_state=42)  # Use 5% of the data for faster processing
X_down = down_sample_data[['NO2', 'TEMP', 'DEWP', 'WSPM']]
# Standardize the data for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_down)

# Fit DBSCAN with a chosen epsilon and min_samples
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add DBSCAN labels to the dataset
down_sample_data['DBSCAN_Labels'] = dbscan_labels

# Number of clusters and outliers (-1 indicates outliers)
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters found by DBSCAN: {num_clusters}")
print(f"Number of outliers: {list(dbscan_labels).count(-1)}")

# Visualize DBSCAN clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TEMP', y='NO2', hue='DBSCAN_Labels', data=down_sample_data, palette='Set1', style='DBSCAN_Labels')
plt.title('DBSCAN Clustering')
plt.xlabel('Temperature (TEMP)')
plt.ylabel('NO2')
plt.legend(title='Cluster Labels')
plt.show()

######################### Apriori #####################################################3#####

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
''''''
# Discretize continuous features
def discretize_feature(column, bins, labels):
    return pd.cut(column, bins=bins, labels=labels)

# Discretize the features (NO2, TEMP, DEWP, WSPM)
subset_data2['NO2_Discretized'] = discretize_feature(subset_data2['NO2'], bins=[0, 10, 20, 30, 40, 50], labels=['Low', 'Medium', 'High', 'Very High', 'Extremely High'])
subset_data2['TEMP_Discretized'] = discretize_feature(subset_data2['TEMP'], bins=[-10, 0, 10, 20, 30, 40], labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
subset_data2['DEWP_Discretized'] = discretize_feature(subset_data2['DEWP'], bins=[-10, 0, 10, 20, 30], labels=['Very Low', 'Low', 'Medium', 'High'])
subset_data2['WSPM_Discretized'] = discretize_feature(subset_data2['WSPM'], bins=[0, 5, 10, 15, 20], labels=['Low', 'Medium', 'High', 'Very High'])

# Convert to one-hot encoding for Apriori
subset_data2_onehot = pd.get_dummies(subset_data2[['NO2_Discretized', 'TEMP_Discretized', 'DEWP_Discretized', 'WSPM_Discretized']])

# Apply Apriori algorithm to find frequent item sets
frequent_itemsets = apriori(subset_data2_onehot, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))

# Display rules
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Prevent line wrapping
print(f"Number of association rules found: {len(rules)}")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])




