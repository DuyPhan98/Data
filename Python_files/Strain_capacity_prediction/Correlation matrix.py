#Import libraries..............................................
import pandas
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rc('font', family='serif', serif='Times New Roman',size=14)
# Define dataset...............................................
Data="Database for crack spacing with no fiber type 241.csv"
df = pandas.read_csv(Data)
df.head()
#df_encoded = pd.get_dummies(df, columns=['Fiber type'])
# Calculate and plot correlation matrix........................
matrix = df.corr()
print(matrix)
sns.heatmap(matrix, annot = True,cmap="RdYlBu") # ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
plt.show()
	#of white, dark, whitegrid, darkgrid, ticks
plt.figure(figsize=(4.5, 4.5))

sns.histplot(df, x="Crack spacing (mm)", kde=True,  bins=20, color='green')
#plt.ylim(0,140)
#plt.xlim(0,150)
plt.show()
exit()





#Separately strain capacity for steel fiber and PE fiber
Data="Separate strain capacity.csv"
df1 = pandas.read_csv(Data)
df1.head()
plt.figure(figsize=(4.5, 4.5))
sns.histplot(df1, x="Strain capacity (%)", kde=True,  bins=20, color='red')
plt.ylim(0,10)
#plt.xlim(0,1)
plt.show()