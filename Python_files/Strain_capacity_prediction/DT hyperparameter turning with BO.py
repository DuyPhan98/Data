#Programmer: Tan-Duy PHAN
#Install the scikit-optimize library using the following command:
#pip install scikit-optimize
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import lines
from skopt import BayesSearchCV
import pandas as pd, numpy as np
import seaborn as sns
import pandas
import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from skopt.plots import plot_convergence
from scipy.interpolate import griddata
mpl.rc('font', family='serif', serif='Times New Roman',size=12)

#Load dataset
Database="Database for strain capacity with fiber type 241.csv"
df = pandas.read_csv(Database)
df.head()
print(df)
df_encoded = pd.get_dummies(df, columns=['Fiber type 1','Fiber type 2'])

#print(df_encoded)
#Check for missing values
#print(df.isnull().sum())
X=df_encoded.drop(columns=['Strain capacity (%)']) 
#print(X)
y = df_encoded['Strain capacity (%)'] #output variable
#y = df.iloc[:, 8].values 
#print(y)
#Check for missing values
#print(df.isnull().sum())
#Splitting the data into independent and dependent attributes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.tree import DecisionTreeRegressor
rf = DecisionTreeRegressor()
from sklearn.metrics import r2_score
# Define search space
search_space = {
    "max_features": (1, 50),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1,50),
    "max_depth": (1,100)
}

# Define Bayesian Optimization
opt = BayesSearchCV(
    rf,
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)

# Fit model
opt.fit(X_train, y_train)
# Evaluate best model
best_model = opt.best_estimator_
y_pred = best_model.predict(X_test)
R2 = r2_score(y_test, y_pred)

print("Best parameters:", opt.best_params_)
print("Test R2:", R2)

# Plot fitness curve (convergence of MAE)
val_mae = -opt.cv_results_['mean_test_score']

# Cumulative best MAE at each step (convergence line)
cumulative_best = np.minimum.accumulate(val_mae)
cumulative_best_plot=abs(cumulative_best)
plt.figure(figsize=(4, 4))
plt.plot(cumulative_best_plot,linestyle="--",color="green", linewidth=1.5, marker="*")
#plt.title("Bayesian Optimization Fitness Curve (MAE)")
plt.xlabel("Number of interation")
#plt.ylim(1.7,1.8)
plt.ylabel("Best cost (R\u00B2)")
plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.xticks([0, 10, 20, 30, 40, 50])
#plt.grid(True)
plt.tight_layout()
plt.savefig('DT covert curves.JPG', dpi=500, bbox_inches = 'tight')
plt.show()
df = pd.DataFrame({
    'cumulative_best': cumulative_best_plot
})

# Export to Excel
df.to_excel("DT_cumulative_best_plot.xlsx", index=False)





