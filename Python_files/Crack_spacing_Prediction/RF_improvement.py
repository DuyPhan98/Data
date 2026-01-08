# Import libraries.............................................................................
import pandas as pd
import pandas
import numpy as np
from matplotlib import pyplot
from matplotlib import lines
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# Set font.....................................................................................
mpl.rc('font', family='serif', serif='Times New Roman',size=13)
#Import data...................................................................................
Database="Database for crack spacing with fiber type 241.csv"
df = pandas.read_csv(Database)
df.head()
df_encoded = pd.get_dummies(df, columns=['Fiber type 1',"Fiber type 2"])
print(df_encoded)
df_encoded.to_excel('encoded_dataframe.xlsx', index=False)
#Check for missing values......................................................................
print(df.isnull().sum())
#Splitting the data into independent and dependent attributes
X=df_encoded.drop(columns=['Crack spacing (mm)']) 
print(X)
y_raw = df_encoded['Crack spacing (mm)']
# Log-transform target
y = np.log(y_raw + 1e-6)
print(y)

#Splitting the data into two set for training and testing...................................
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Inport and training pure KNN.............................................................................
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state = 42, max_depth=100,max_features=1, max_leaf_nodes=131, n_estimators=5)

#Result of K fold
scores1 = cross_val_score (regressor, X_train, y_train, cv=10, scoring='r2',n_jobs = -1)

print('10-fold mean r2:', abs(np.mean(scores1)))
print('10-fold std deviation r2:', np.std(scores1))

scores2 = cross_val_score (regressor, X_train, y_train, cv=10, scoring='neg_mean_absolute_error', n_jobs = -1)
print('10-fold mean mean_absolute_error:', abs(np.mean(scores2)))
print('10-fold std deviation of MAE:', np.std(scores2))

scores3 = cross_val_score (regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs = -1)
print('10-fold mean mean_square_error:', abs(np.mean(scores3)))
print('10-fold std deviation of MSE:', np.std(scores3))

scores4 = np.sqrt(-scores3)
print('10-fold mean RMSE:', np.mean(scores4))
print('10-fold std deviation of RMSE:', np.std(scores4))
#Prrint to excel file
# Compute metrics
results = {
    'Metric': ['R2', 'MAE', 'MSE', 'RMSE'],
    'Mean': [
        abs(np.mean(scores1)),
        abs(np.mean(scores2)),
        abs(np.mean(scores3)),
        np.mean(scores4)
    ],
    'Standard Deviation': [
        np.std(scores1),
        np.std(scores2),
        np.std(scores3),
        np.std(scores4)
    ]
}

# Create DataFrame
results_df = pd.DataFrame(results)

# Export to Excel
results_df.to_excel('cross_validation_results from RF-BO model.xlsx', index=False, engine='openpyxl')

print("Cross-validation results saved to 'cross_validation_results from RF-BO model.xlsx'")

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)
print(y_pred)
# Testing........................................................................................
y_predtest = regressor.predict(X_test)
print(y_predtest)
y_train= np.array(y_train)
print(y_train)
print(y_pred)
#Export data to Excel.............................................................................
pd.DataFrame(y_train, columns=['Experiment training_RFBO']).to_csv('Experiment training_RFBO.csv')
pd.DataFrame(y_pred, columns=['Prediction training_RFBO']).to_csv('Prediction training_RFBO.csv')
# Collection  testing data
y_test= np.array(y_test)
print(y_predtest)
pd.DataFrame(y_test, columns=['Experiment testing_RFBO']).to_csv('Experiment testing_RFBO.csv')
pd.DataFrame(y_predtest, columns=['Prediction testing_RFBO']).to_csv('Prediction testing_RFBO.csv')

# Ploting relationship between predicted and experimental value in training.......................
Number1 = list(range(1, 169,1))
Number2 = list(range(1, 74,1))
plt.figure(figsize=(5,5))
plt.xlabel("Number of specimen",font='Times New Roman',size=15)
plt.ylabel("DIF of tensile strength",font='Times New Roman',size=15)
plt.plot(Number1, y_train,color="gray",linewidth=2.0,marker="o",markersize=7)
plt.plot(Number1, y_pred,color="red",linestyle='-.',linewidth=2.0,marker="*",markersize=7)
plt.legend(["Experiment","Pure RF"])
plt.xlim(0, 350)
plt.ylim(0,10)
plt.tick_params(axis="both", labelsize=16,width=1.5)
# Ploting relationship between predicted and experimental value in testing.......................
plt.figure(figsize=(5,5))
plt.xlabel("Number of specimen",font='Times New Roman',size=15)
plt.ylabel("DIF of tensile strength ",font='Times New Roman',size=15)
plt.plot(Number2, y_test,color="gray",linewidth=2.0,marker="o",markersize=7)
plt.plot(Number2, y_predtest,color="red",linestyle='--',linewidth=2.0,marker="^",markersize=7)
plt.legend(["Experiment","Pure RF"])
plt.xlim(0, 150)
plt.ylim(0,10)
plt.tick_params(axis="both", labelsize=16,width=1.5)
plt.show()

#Caculate the statictical metrics.................................................................
from sklearn import metrics
print('Mean Absolute Error_Train:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error_Train:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error_Train:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
R2_train = metrics.r2_score(y_train, y_pred)
print('R-squared_Train = ', round(R2_train, 10))
# Testing
print('Mean Absolute Error_Test:', metrics.mean_absolute_error(y_test, y_predtest))
print('Mean Squared Error_Test:', metrics.mean_squared_error(y_test, y_predtest))
print('Root Mean Squared Error_Test:', np.sqrt(metrics.mean_squared_error(y_test, y_predtest)))
R2_test = metrics.r2_score(y_test, y_predtest)
print('R-squared_Test = ', round(R2_test, 10))
plt.show()


