import os
import random

import pandas as pd

import flwr as fl
import tensorflow as tf

import shap
import matplotlib.pyplot as plt

import sage

from sklearn.metrics import roc_auc_score



# source: https://github.com/adap/flower/blob/main/examples/simulation_tensorflow/sim.py

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('INFO')


# select by hand
BIGGEST_CENTERS = ["ANT",
                   "AMC",
                   "LUM",
                   "AZM",
                   "RIJ",
                   "MCH"]

RANDOM_CENTERS = ["random1",
                  "random2",
                  "random3",
                  "random4",
                  "random5",
                  "random6"]

#SELECTED_CENTERS = BIGGEST_CENTERS

EPOCH_LIST = [2, 10, 20]


CLIENT_ID_LIST = [RANDOM_CENTERS]





# FULL_DATA_SET = pd.read_csv("../02-preprocessing/clinic_federated_preprocessed.csv")
TRAIN_SET = pd.read_csv("../02-preprocessing/clinic_fed_train.csv", index_col=0)
VAL_SET = pd.read_csv("../02-preprocessing/clinic_fed_val.csv", index_col=0)
TEST_SET = pd.read_csv("../02-preprocessing/clinic_fed_test.csv", index_col=0)

X_TRAIN = TRAIN_SET.iloc[:, :-1]
Y_TRAIN = TRAIN_SET.iloc[:, -1]

X_VAL = VAL_SET.iloc[:, :-1].drop("StudySubjectID", axis="columns")
Y_VAL = VAL_SET.iloc[:, -1]

X_TEST = TEST_SET.iloc[:, :-1].drop("StudySubjectID", axis="columns")
Y_TEST = TEST_SET.iloc[:, -1]


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val, cid) -> None:
        super().__init__()
        self.model = model

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.cid = cid

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"Start train model {self.cid}")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=EPOCHS, verbose=0, batch_size=6) # was 20
        # model save zB h5 file
        self.model.save(f"./models/local/{EPOCHS}-{FED_ROUNDS}/{self.cid}", save_format='h5')
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.save(f"./models/{EPOCHS}-{FED_ROUNDS}/{self.cid}", save_format='h5')
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    # define the keras model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(14, input_shape=(10,), activation='relu'),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(units=4, activation="relu"),
            tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.AUC())

    train_set_center = pd.read_csv(f"../02-preprocessing/datasets/train_norm_{cid}.csv")
    val_set_center = pd.read_csv(f"../02-preprocessing/datasets/val_norm_{cid}.csv")

    x_train_center = train_set_center.iloc[:, :-1].drop("StudySubjectID", axis="columns")
    y_train_center = train_set_center.iloc[:, -1]

    x_val_center = val_set_center.iloc[:, :-1].drop("StudySubjectID", axis="columns")
    y_val_center = val_set_center.iloc[:, -1]

    # Create and return client
    return FlwrClient(model, x_train_center, y_train_center, x_val_center, y_val_center, cid)


def evaluate_models():
    auc_sum = 0
    x_test_data_per_center = {}
    y_test_data_per_center = {}

    for center in SELECTED_CENTERS:

        test_set_center = pd.read_csv(f"../02-preprocessing/datasets/test_norm_{center}.csv")

        x_test_center = test_set_center.iloc[:, :-1].drop("StudySubjectID", axis="columns")
        y_test_center = test_set_center.iloc[:, -1]

        center_model = tf.keras.models.load_model(f"./models/{EPOCHS}-{FED_ROUNDS}/{center}")

        y_predict = center_model.predict(x_test_center)
        auc = roc_auc_score(y_test_center, y_predict)

        auc_sum = auc_sum + auc
        print(f"Model in Center {center} has AUC of {auc}")

        x_test_data_per_center[center] = x_test_center
        y_test_data_per_center[center] = y_test_center

    print(f"Average AUC: {auc_sum / len(SELECTED_CENTERS)}")

    return x_test_data_per_center, y_test_data_per_center


def apply_shap(test_data_per_center) -> None:

    full_test = pd.read_csv(f"../02-preprocessing/datasets/test_norm_full.csv").drop("StudySubjectID",
                                                                                           axis="columns")
    x_test_full = full_test.iloc[:, :-1]
    y_test_full = full_test.iloc[:, -1]

    for center in SELECTED_CENTERS:
        test_center = pd.read_csv(f"../02-preprocessing/datasets/test_norm_{center}.csv").drop("StudySubjectID",
                                                                                               axis="columns")

        x_test_center = test_center.iloc[:, :-1]
        y_test_center = test_center.iloc[:, -1]

        print(f"SHAP for center {center}")

        # federated model aggregated
        center_model = tf.keras.models.load_model(f"./models/{EPOCHS}-{FED_ROUNDS}/{center}")

        # Federated model before aggregation -> after local training (like transfer learning with one training cycle
        center_model_local = tf.keras.models.load_model(f"./models/local/{EPOCHS}-{FED_ROUNDS}/{center}")

        save_shap(center_model.predict, x_test_center, y_test_center, center)
        save_shap(center_model.predict, x_test_full, y_test_full, f"benchmark-{center}")

        save_shap(center_model_local.predict, x_test_center, y_test_center, f"local-{center}")
        save_shap(center_model_local.predict, x_test_full, y_test_full, f"benchmark-local-{center}")


def save_shap(predict_func, x_test, y_test, center_name) -> None:
    center_explainer = shap.KernelExplainer(predict_func, x_test)
    center_shap_values = center_explainer.shap_values(x_test)

    y_pred = pd.DataFrame(predict_func(x_test))
    auc = roc_auc_score(y_test, y_pred)

    shap.summary_plot(center_shap_values[0], x_test, show=False)
    plt.title(f"SHAP Summary Plot for Center {center_name}; AUC = {round(auc, 2)}; Epochs={EPOCHS}; {FED_ROUNDS}")

    plt.savefig(f'./results/{EPOCHS}-{FED_ROUNDS}/summary_plot_{center_name}.png', facecolor="white", pad_inches=0.3, bbox_inches="tight")
    plt.clf()
    plt.cla()

    shap.summary_plot(center_shap_values, x_test, show=False)
    plt.title(f"SHAP Feature Importance Plot for Center {center_name}; AUC = {round(auc, 2)}; Epochs={EPOCHS}; fed_rounds = {FED_ROUNDS} ")

    plt.savefig(f'./results/{EPOCHS}-{FED_ROUNDS}/importance_plot_{center_name}.png', facecolor="white", pad_inches=0.3, bbox_inches="tight")
    plt.clf()
    plt.cla()


def apply_sage(x_test_data_per_center, y_test_data_per_center):

    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()

    feature_names = X_TEST.columns.to_list()

    for center in SELECTED_CENTERS:
        x_test_center = x_test_data_per_center[center].to_numpy()
        y_test_center = y_test_data_per_center[center].to_numpy()

        center_model = tf.keras.models.load_model(f"./models/{center}")

        # Setup and calculate
        imputer = sage.MarginalImputer(center_model, x_test_center[:5])
        estimator = sage.KernelEstimator(imputer, 'mse', )

        print(f"SAGE for center {center}")

        sage_values = estimator(x_test_center, y_test_center)

        # Plot results
        sage_values.plot(feature_names, return_fig=False)
        plt.title(f'Feature Importance for Center {center} ')

        plt.savefig(f'./results/sage_plot_{center}.png', facecolor="white")
        plt.clf()

        print(f"sensitivity for center {center}")

        sensitivity = estimator(x_test_center)

        # Plot results
        sensitivity.plot(feature_names, return_fig=False)
        plt.title(f'Model Sensitivity for Center {center}')

        plt.savefig(f'./results/sensitivity_plot_{center}.png', facecolor="white")
        plt.clf()


def reset_seeds():
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


def main(epochs: int, federated_rounds: int) -> None:

    #for epochs in EPOCH_LIST:

    global EPOCHS
    EPOCHS = epochs

    global FED_ROUNDS
    FED_ROUNDS = federated_rounds

    print(f"EPOCH: {EPOCHS}")
    print(f"ROUNDS: {FED_ROUNDS}")
    try:
        os.mkdir(f"./results/{EPOCHS}-{FED_ROUNDS}")
    except OSError as error:
        pass

    try:
        os.mkdir(f"./models/{EPOCHS}-{FED_ROUNDS}")
    except OSError as error:
        pass
    try:
        os.mkdir(f"./models/local/{EPOCHS}-{FED_ROUNDS}")
    except OSError as error:
        pass

    for clientIDs in CLIENT_ID_LIST:
        global SELECTED_CENTERS
        SELECTED_CENTERS = clientIDs

        num_clients = len(clientIDs)

        reset_seeds()

        # Start Flower simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            clients_ids=clientIDs,
            client_resources={"num_cpus": 6},
            config=fl.server.ServerConfig(num_rounds=FED_ROUNDS),
            # toDo: define how federated averaging should work
            strategy=fl.server.strategy.FedAvg(
                fraction_fit=1,
                fraction_evaluate=1,
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
            ),
        )

        x_test_data_per_center, y_test_data_per_center = evaluate_models()
        apply_shap(x_test_data_per_center)
        #apply_sage(x_test_data_per_center, y_test_data_per_center)
        print("done with centers")


    print("done")


if __name__ == "__main__":

    main(2, 10)
    main(5, 4)
    main(10, 2)
