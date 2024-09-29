import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import pickle

# Example data (assuming you have a DataFrame with the extracted features)
df1 = pd.DataFrame(pd.read_excel('Features_Training_Initial_55.xlsx'))
df21 = pd.DataFrame(pd.read_excel('Features_Training_Final_30_Set_1.xlsx'))
df22 = pd.DataFrame(pd.read_excel('Features_Training_Final_30_Set_2.xlsx'))
df23 = pd.DataFrame(pd.read_excel('Features_Training_Final_30_Set_3.xlsx'))
df24 = pd.DataFrame(pd.read_excel('Features_Training_Final_30_Set_4.xlsx'))
df2 = pd.concat([df21, df22, df23, df24], axis=0)

df1_columns = set(df1.columns)
df2_columns = set(df2.columns)
common_columns = df1_columns.intersection(df2_columns)

df2 = df2.reset_index(drop=True)

df2.drop(columns=['mode', 'iv', 'Encrypted Data (Hex)', 'Encrypted Data (Binary)', 'Original Text', 'Length', 'Algorithm'], inplace=True)
df = pd.concat([df1, df2], axis=1)

y_train = df['Algorithm']
df.drop(columns=['Original Text', 'Length', 'Encrypted Data (Binary)', 'Encrypted Data (Hex)', 'Algorithm', 'iv'], inplace=True)
X_train = df



from sklearn.preprocessing import LabelEncoder

label_encoder1 = LabelEncoder()
label_encoder1.fit(X_train[['mode']])
X_train['mode'] = label_encoder1.transform(X_train[['mode']])

X_train.drop(columns=['byte_value_histogram'], inplace=True)

X_train.drop(columns='byte_value_percentiles', inplace=True)

X_train.drop(columns=['freq_byte_value_diff', 'run_length_encoding', 'byte_value_transition_matrix', 'freq_byte_value_2grams', 'freq_byte_value_3grams', 'freq_byte_value_4grams', 'byte_value_acf', 'byte_value_power_spectrum'], inplace=True)

from sklearn.preprocessing import StandardScaler
exclude_columns = ['mode']
columns_to_scale = X_train.columns.difference(exclude_columns)

scaler = StandardScaler()

scaler.fit(X_train[columns_to_scale])
X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])







# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)



pickle.dump(rf_model, open("model.pkl", "wb"))