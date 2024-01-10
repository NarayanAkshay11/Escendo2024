import numpy as np #Mathematical Operations
import pandas as pd #For accessing dataframes
import matplotlib.pyplot as plt #Plotting


import warnings #Who likes warnings anyway
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\Narayanakshay\\Downloads\\forestfires.csv")
pd.set_option('display.max_columns', None)
print(df.head(20))
print(df.describe())
df = df.drop(['X','Y'],axis = 1)
df = df[df['area'] != 0.0]
#for column in df.columns[2:-1]:  # Exclude the 'area' column
    #plt.scatter(df[column], df['area'], label=column)

#plt.xlabel('Column Values')
#plt.ylabel('Area')
#plt.legend()
#plt.show()

for i, column in enumerate(df.columns[2:-1]):  # Exclude the 'area' column
    row, col = divmod(i, 3)
    axes[row, col].scatter(df[column], df['area'])
    axes[row, col].set_title(f'{column} vs. area')
    axes[row, col].set_xlabel(column)
    axes[row, col].set_ylabel('area')

plt.tight_layout()
plt.show()
