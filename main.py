import tensorflow as tf
from utils import generate_datasets_bcw, generate_datasets_pid, format_y_preds
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, f1_score

# Dataset BCW/PID
USE_BCW = False

# Global parameters
EPOCHS = 20
EXECUTIONS = 5
TRAIN_BATCH_SIZE = 8 if USE_BCW else 64
TEST_BATCH_SIZE = 1
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = generate_datasets_bcw() if USE_BCW else generate_datasets_pid()
DATASET_FEATURES = X_TRAIN.shape[1]
TRAIN_STEPS = len(X_TRAIN) // TRAIN_BATCH_SIZE

# Results
res_recall = []
res_precision = []
res_f1 = []
res_mcc = []


def create_model_bcw(is_train=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu",
                                    batch_input_shape=[TRAIN_BATCH_SIZE if is_train else TEST_BATCH_SIZE,
                                                       DATASET_FEATURES]))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model


def create_model_pid(is_train=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, activation="relu",
                                    batch_input_shape=[TRAIN_BATCH_SIZE if is_train else TEST_BATCH_SIZE,
                                                       DATASET_FEATURES]))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["binary_accuracy"])
    return model


def train_data_generator():
    def norm(x, y):
        return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

    x_raw = tf.convert_to_tensor(X_TRAIN)
    y_raw = tf.convert_to_tensor(Y_TRAIN)

    dataset = tf.data.Dataset.from_tensor_slices((x_raw, y_raw))
    dataset = (
        dataset.map(norm)
        .cache()
        .shuffle(buffer_size=TRAIN_BATCH_SIZE)
        .batch(TRAIN_BATCH_SIZE, drop_remainder=True)
    )
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    data_iter = iter(dataset)

    return data_iter


def test_data_generator():
    def norm(x):
        return tf.cast(x, tf.float32)

    x_raw = tf.convert_to_tensor(X_TEST)

    dataset = tf.data.Dataset.from_tensor_slices(x_raw)
    dataset = (
        dataset.map(norm)
        .cache()
        .shuffle(buffer_size=TEST_BATCH_SIZE)
        .batch(TEST_BATCH_SIZE, drop_remainder=True)
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    data_iter = iter(dataset)

    return data_iter


print("Using Breast Cancer Wisconsin Dataset" if USE_BCW else "Using Pima Indian Diabetes Database")
for execution in range(EXECUTIONS):
    # Shuffle data
    if execution > 0:
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = generate_datasets_bcw() if USE_BCW else generate_datasets_pid()

    # Train model
    train_model = create_model_bcw(is_train=True) if USE_BCW else create_model_pid(is_train=True)
    train_model.fit(x=train_data_generator(), epochs=EPOCHS, steps_per_epoch=TRAIN_STEPS, verbose=1)

    # Test model
    test_model = create_model_bcw(is_train=False) if USE_BCW else create_model_pid(is_train=False)
    test_model.set_weights(train_model.weights)

    # Evaluation
    y_preds = test_model.predict(x=test_data_generator(), batch_size=TEST_BATCH_SIZE)
    formatted_y_preds = format_y_preds(y_preds)

    # Results collection
    recall = recall_score(Y_TEST, formatted_y_preds, pos_label=0)
    precision = precision_score(Y_TEST, formatted_y_preds, pos_label=0)
    f1 = f1_score(Y_TEST, formatted_y_preds, pos_label=0)
    mcc = matthews_corrcoef(Y_TEST, formatted_y_preds)

    res_recall.append(recall)
    res_precision.append(precision)
    res_f1.append(f1)
    res_mcc.append(mcc)

print("Recall:")
for j in range(len(res_recall)):
    print(res_recall[j], ",", sep="")

print("\n\nPrecision:")
for j in range(len(res_precision)):
    print(res_precision[j], ",", sep="")

print("\n\nF1:")
for j in range(len(res_f1)):
    print(res_f1[j], ",", sep="")

print("\n\nMCC:")
for j in range(len(res_mcc)):
    print(res_mcc[j], ",", sep="")
