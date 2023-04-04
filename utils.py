import pandas as pd
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2


def generate_datasets_bcw():
    data = pd.read_csv('C:/Users/falve/PycharmProjects/tcc2023/data/bcw.csv')
    data.drop(['id'], inplace=True, axis=1)
    data.replace('?', -99999, inplace=True)
    data['benormal'] = data['benormal'].map(lambda x: 1 if x == 4 else 0)
    # First 9 column is input parameters.
    X = data.iloc[:, 0:9]
    # Last Column is output data. So (0 is benign, 1 is malignant)
    y = data.iloc[:, -1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2)


def generate_datasets_pid():
    df = pd.read_csv('C:/Users/falve/PycharmProjects/tcc2023/data/pid.csv')
    df = df.dropna(subset=['BloodPressure'])
    df = df.dropna(subset=['BMI'])
    df = df.dropna(subset=['Glucose'])
    # define imputer
    imputer = KNNImputer()
    # fit on the dataset
    imputer.fit(df)
    # transform the dataset
    df_filled = imputer.transform(df)
    df_filled = pd.DataFrame(df_filled)
    # df_filled.info()
    df2 = df_filled.rename(
        {0: 'Pregnancies', 1: 'Glucose', 2: 'BloodPressure', 3: 'SkinThickness', 4: 'Insulin', 5: 'BMI', 6: 'DBF',
         7: 'Age', 8: 'Outcome'}, axis=1)  # new method
    # BMI	DiabetesPedigreeFunction	Age	Outcome
    X = df2[['Glucose', 'Pregnancies', 'Age', 'Insulin', 'BMI', 'BloodPressure', 'SkinThickness']]
    y = df2['Outcome']
    Scaler = StandardScaler()
    Scaler.fit(X)
    X = Scaler.transform(X)

    return train_test_split(X, y, test_size=TEST_SIZE)


def format_y_preds(y_preds):
    threshold = 0.5
    tensor_y_preds = tf.convert_to_tensor(y_preds)
    threshold = tf.cast(threshold, tensor_y_preds.dtype)
    return tf.cast(tensor_y_preds > threshold, tensor_y_preds.dtype)
