#import lib
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', family='serif', serif='Times New Roman',size=16)
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import shap
shap.initjs()
# Define dataset
Dataset="Database for strain capacity with no fiber type 241.csv"
df = pandas.read_csv(Dataset)
df.head()
X = df.iloc[:, 0:7].values 
print(X)
y = df.iloc[:, 7].values
print(y)
#Check for missing values
print(df.isnull().sum())
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 42, max_depth=7,max_features=4, max_leaf_nodes=164, n_estimators=152)
model.fit(X, y)

from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
# Generate colormap through matplotlib
newCmap = LinearSegmentedColormap.from_list("", ['#c4cfd4','#3345ea'])
explainer = shap.Explainer(model, X)
#shap_values = explainer.shap_values(X,check_additivity=True)
#sv=explainer(X)
#shap.plots.bar(sv)
#shap.summary_plot(shap_values, X) # ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
#shap.force_plot(explainer.expected_value, shap_values,show=False,link='logit') 
#shap.decision_plot(explainer.expected_value, shap_values)

#SHAP dependence plot
#shap.dependence_plot("Feature 8", shap_values, X, dot_size=60,show=True,cmap="rainbow")
#shap.plots.waterfall(shap_values[1], max_display=14)
# Partial denpendence plot
from sklearn.inspection import PartialDependenceDisplay

disp1 = PartialDependenceDisplay.from_estimator(model, X, [6], kind="both", #BOTH
                                                ice_lines_kw={'color':'slategray','alpha':0.2},
                                                pd_line_kw={'color':'deeppink','linewidth':3,'linestyle':'-'},
                                                line_kw={'label':'Average line'})
# Set figure size using disp1.figure
disp1.figure_.set_size_inches(4.5, 4.5)
plt.legend(loc='upper left') #upper left
plt.ylim(0,10)
plt.yticks([0, 2, 4, 6, 8, 10])
plt.xticks([50, 100, 150, 200, 250])
#plt.xlim(0,30)
plt.xlabel(r'$GL$ (mm)',font='Times New Roman',size=18) #(kg/$m^{3}$) r'$\Sigma_{fu1}$'
plt.ylabel(" Strain capacity (%)",font='Times New Roman',size=18)
#plt.xlim(50,250)
plt.savefig('Effect of gauge length.JPG', dpi=500, bbox_inches = 'tight')
plt.show()
