import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from Knn import knn_model
from svm import svm_model
#Loading dataset
df = pd.read_csv('creditcard.csv')
#Description of data
print(df.info())
print(df.describe())
#class values 
occ = df['Class'].value_counts()
print(occ)
#PDF
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
for i in range(len(x[0])):
    axes[i].hist(x[:, i]) 
    axes[i].set_title(f'Feature {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')
plt.tight_layout()
plt.show()
#Boxplots
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
for i in range(len(x[0])):
    axes[i].boxplot(x[:, i]) 
    axes[i].set_title(f'Feature {i+1}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')
plt.tight_layout()
plt.show()
#Relation between features and the output 
num_rows = 10
num_cols = 3
total_plots = num_rows * num_cols
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))
axes = axes.flatten()
summary_stats=df.describe()
#for i in range(len(x[0])):
  #  axes[i].scatter(x[:, i],y) 
   # axes[i].set_title(f'Feature {i+1}')
    #axes[i].set_xlabel('Value')
    #axes[i].set_ylabel('Frequency')
    #stats_text = f"Mean: {summary_stats.iloc[1, i]:.2f}\nStd Dev: {summary_stats.iloc[2, i]:.2f}\nMin: {summary_stats.iloc[3, i]:.2f}\nMax: {summary_stats.iloc[7, i]:.2f}"
    #axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8, color='black')
#plt.tight_layout()
#plt.show()
corr = df.corr()
ax, fig = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, vmin=-1, cmap=plt.cm.Blues, annot=True)
plt.show()
#correlation 
corr[abs(corr['Class']) < 0.3]['Class']
#undersampling

#We used gridsearch to get the best hyperparametrs of each model
#Training
svm_model(df,'rbf',0.1, 1.0, 1e-5,'enn')
knn_model(df,3,'nearmiss1')






