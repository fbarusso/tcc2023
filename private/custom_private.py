import tf_encrypted as tfe
import tensorflow as tf
from tf_encrypted.protocol.aby3 import ABY3
from tf_encrypted.player import DataOwner
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, f1_score
from custom_utils import format_y_preds, DatasetClass, generate_datasets_bcw, generate_datasets_pid

# Dataset BCW/PID
USE_BCW = True

# Global parameters
EXECUTIONS = 5
EPOCHS = 20
TRAIN_BATCH_SIZE = 8 if USE_BCW else 64
TEST_BATCH_SIZE = 1

# TFE parameters
PRECISION = "high"


def create_model_bcw(batch_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu",
                                    batch_input_shape=batch_shape))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return tfe.keras.models.clone_model(model)


def create_model_pid(batch_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, activation="relu",
                                    batch_input_shape=batch_shape))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return tfe.keras.models.clone_model(model)


# Results
res_recall = []
res_precision = []
res_f1 = []
res_mcc = []

print("Using Breast Cancer Wisconsin Dataset" if USE_BCW else "Using Pima Indian Diabetes Database")
for execution in range(EXECUTIONS):
    # TFE Setup
    config = tfe.LocalConfig(
        player_names=[
            "server0",
            "server1",
            "server2",
            "train-client",
            "test-client",
        ]
    )

    tfe.set_config(config)
    tfe.set_protocol(ABY3(fixedpoint_config=PRECISION))

    # Data creation
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = generate_datasets_bcw() if USE_BCW else generate_datasets_pid()

    # Train DataOwner
    train_dataset = DatasetClass(x=X_TRAIN, y=Y_TRAIN, batch_size=TRAIN_BATCH_SIZE, train=True)
    train_data_owner = DataOwner(config.get_player("train-client"), train_dataset.generator_builder())

    # Test DataOwner
    test_dataset = DatasetClass(x=X_TEST, y=Y_TEST, batch_size=TEST_BATCH_SIZE, train=False)
    test_data_owner = DataOwner(config.get_player("test-client"), test_dataset.generator_builder())

    # Train model
    train_batch_shape = train_dataset.batch_shape
    train_model = create_model_bcw(batch_shape=train_batch_shape) if USE_BCW else create_model_pid(
        batch_shape=train_batch_shape)

    # Compile train model
    loss = tfe.keras.losses.BinaryCrossentropy()
    optimizer = tfe.keras.optimizers.Adam(learning_rate=0.01)
    train_model.compile(optimizer, loss)

    # Train model
    train_data_iter = train_data_owner.provide_data()
    train_model.fit(x=train_data_iter, epochs=EPOCHS, steps_per_epoch=train_dataset.iterations, verbose=1)

    # Test model
    test_batch_shape = test_dataset.batch_shape
    test_model = create_model_bcw(batch_shape=test_batch_shape) if USE_BCW else create_model_pid(
        batch_shape=test_batch_shape)
    test_model.set_weights(train_model.weights)

    test_data_iter = test_data_owner.provide_data()
    y_preds = test_model.predict(x=test_data_iter, reveal=True)

    formatted_y_preds = format_y_preds(y_preds)

    # Results collection
    recall = recall_score(test_dataset.y, formatted_y_preds, pos_label=0)
    precision = precision_score(test_dataset.y, formatted_y_preds, pos_label=0)
    f1 = f1_score(test_dataset.y, formatted_y_preds, pos_label=0)
    mcc = matthews_corrcoef(test_dataset.y, formatted_y_preds)

    res_recall.append(recall)
    res_precision.append(precision)
    res_f1.append(f1)
    res_mcc.append(mcc)

    print(recall, precision, f1, mcc)

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
