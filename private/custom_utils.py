import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2


def generate_datasets_bcw():
    data = pd.read_csv('/home/febarusso/TFE/tf-encrypted/examples/custom/bcw.csv')
    # Remove ID column
    data.drop(['id'], inplace=True, axis=1)
    # Replace missing values
    data.replace('?', -99999, inplace=True)
    # Replace label from 2/4 to 0/1
    data['benormal'] = data['benormal'].map(lambda x: 1 if x == 4 else 0)
    # First 9 columns are input parameters
    X = data.iloc[:, 0:9]
    # Last column is labels
    y = data.iloc[:, -1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=TEST_SIZE)


def generate_datasets_pid():
    df = pd.read_csv('/home/febarusso/TFE/tf-encrypted/examples/custom/pid.csv')
    # Preprocessing

    # 48.69 % of the values are zeros which makes data not reliable for Insulin
    df.drop('Insulin', axis=1, inplace=True)

    # Replace missing values with median
    df.Glucose.replace(0, df.BMI.median(), inplace=True)
    df.BloodPressure.replace(0, df.BloodPressure.median(), inplace=True)
    df.SkinThickness.replace(0, df.SkinThickness.median(), inplace=True)
    df.BMI.replace(0, df.BMI.median(), inplace=True)

    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']

    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    return train_test_split(X, y, test_size=TEST_SIZE)


def format_y_preds(y_preds):
    threshold = 0.5
    tensor_y_preds = tf.convert_to_tensor(y_preds)
    threshold = tf.cast(threshold, tensor_y_preds.dtype)
    return tf.cast(tensor_y_preds > threshold, tensor_y_preds.dtype)


class DatasetClass:
    def __init__(self, x, y, batch_size, train=True) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.train = train
        self.num_features = x.shape[1]
        self.num_classes = 2
        self.num_samples = len(x)
        self.iterations = len(x) // batch_size

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.iterations = self.num_samples // self.batch_size

    @property
    def batch_shape(self):
        return [self.batch_size, self.num_features]

    def _data_generator(self):
        def norm(x, y):
            return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

        x_raw = tf.convert_to_tensor(self.x)
        y_raw = tf.convert_to_tensor(self.y)

        dataset = tf.data.Dataset.from_tensor_slices((x_raw, y_raw))
        dataset = (
            dataset.map(norm)
            .cache()
            .shuffle(buffer_size=self.batch_size)
            .batch(self.batch_size, drop_remainder=True)
        )
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        data_iter = iter(dataset)

        return data_iter

    def _predict_generator(self):
        def norm(x):
            return tf.cast(x, tf.float32)

        x_raw = tf.convert_to_tensor(self.x)

        dataset = tf.data.Dataset.from_tensor_slices((x_raw))
        dataset = (
            dataset.map(norm)
            .cache()
            .shuffle(buffer_size=self.batch_size)
            .batch(self.batch_size, drop_remainder=True)
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        data_iter = iter(dataset)

        return data_iter

    def generator_builder(self):
        return self._data_generator if self.train else self._predict_generator
