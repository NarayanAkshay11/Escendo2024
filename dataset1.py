import numpy as np #Mathematical Operations
import pandas as pd #For accessing dataframes
import matplotlib.pyplot as plt #Plotting


import warnings #Who likes warnings anyway
warnings.filterwarnings('ignore')
def augment_data(df, columns, num_augmentations=5, factor=0.1):
    augmented_data = []

    for _ in range(num_augmentations):
        augmented_instance = df.copy()

        for column in columns:
            # Introduce random perturbation to the numerical feature
            perturbation = np.random.uniform(low=-1, high=1) * factor * df[column]
            augmented_instance[column] += perturbation

        augmented_data.append(augmented_instance)

    augmented_df = pd.concat(augmented_data, ignore_index=True)
    df = pd.concat([df,augmented_df],ignore_index=True)
    return df

df = pd.read_csv("C:\\Users\\Narayanakshay\\Downloads\\forestfires.csv")
pd.set_option('display.max_columns', None)
print(df.head(20))
print(df.describe())
df = df.drop(['X','Y','month','day'],axis = 1)
df = df[df['area'] != 0.0]


# List of numerical columns to augment
numerical_columns = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

for i in range(5):
    #Data augmentation
    aug_df = augment_data(df, numerical_columns)
    # Concatenation
    df = pd.concat([df,aug_df],ignore_index=True)
    
print("Combined Dataset:")
print(df.describe())
df.drop(['FFMC','DMC','DC','ISI'],axis=1)

#Plotting Graphs
#fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
#for i, column in enumerate(df.columns[:-1]):  # Exclude the 'area' column
#    row, col = divmod(i, 3)
#    axes[row, col].scatter(df[column], df['area'])
#    axes[row, col].set_title(f'{column} vs. area')
#    axes[row, col].set_xlabel(column)
#    axes[row, col].set_ylabel('area')

#plt.tight_layout()
#plt.show()

#for column in df.columns[:-1]:  # Exclude the 'area' column
   # plt.scatter(df[column], df['area'], label=column)

#plt.xlabel('Column Values')
#plt.ylabel('Area')
#plt.legend()
#plt.show()
df.to_csv("C:\\Users\\Narayanakshay\\OneDrive\\Desktop\\escendo.csv", index=False)

