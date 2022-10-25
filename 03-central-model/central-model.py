import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import shap

CENTER_NAMES = ["full",
                "ANT",
                "AMC",
                "LUM",
                "AZM",
                "RIJ",
                "MCH",
                "random1",
                "random2",
                "random3",
                "random4",
                "random5",
                "random6"]


def load_data(center):
    print(f"Load Data center {center}")
    train = pd.read_csv(f"../02-preprocessing/datasets/train_norm_{center}.csv").drop("StudySubjectID", axis="columns")
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    val = pd.read_csv(f"../02-preprocessing/datasets/val_norm_{center}.csv").drop("StudySubjectID", axis="columns")
    X_val = val.iloc[:, :-1]
    y_val = val.iloc[:, -1]

    test = pd.read_csv(f"../02-preprocessing/datasets/test_norm_{center}.csv").drop("StudySubjectID", axis="columns")
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_x_test_full():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("full")
    return X_test, y_test


def build_model() -> tf.keras.Model:

    # define the keras model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(14, input_shape=(10,), activation='sigmoid'),
   #         tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(units=4, activation="relu"),
            #tf.keras.layers.Dense(units=2, activation="relu"),
    #        tf.keras.layers.Dropout(0.05),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    return model


def train_model(model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
    print(f"Start training")
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.AUC())

    model_history = model.fit(X_train, y_train, epochs=20,  verbose=0, validation_data=(X_val, y_val), batch_size=16)

    logs = pd.DataFrame(model_history.history)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(logs.loc[0:, "loss"], lw=2, label='training loss')
    plt.plot(logs.loc[0:, "val_loss"], lw=2, label='validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(logs.loc[0:, logs.columns.tolist()[1]], lw=2, label='training ROC AUC score')
    plt.plot(logs.loc[0:, logs.columns.tolist()[3]], lw=2, label='validation ROC AUC score')
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend(loc='lower right')
    plt.show()


def calculate_metrics(model, X_test, y_test, center):

    print("")
    print("------")
    y_pred = pd.DataFrame(model.predict(X_test))
    auc = roc_auc_score(y_test, y_pred)
    print(f"Model {center} AUC = {auc}")

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred.round(0), average="binary")
    print(f"Model {center} precision = {precision}")
    print(f"Model {center} recall = {recall}")
    print(f"Model {center} f1 = {f1}")
    print("------")
    print("")


def calculate_SHAP(model, x_test, y_test, center, x_test_full, y_test_full):
    print(f"Shap center {center}")

    # Shap based on test data of center
    save_shap(model.predict, x_test, y_test, center)

    # SHAP based on full test data as baseline
    save_shap(model.predict, x_test_full, y_test_full, f"benchmark-{center}")


def save_shap(predict_func, x_test, y_test, center_name) -> None:

    center_explainer = shap.KernelExplainer(predict_func, x_test)
    center_shap_values = center_explainer.shap_values(x_test)

    y_pred = pd.DataFrame(predict_func(x_test))
    auc = roc_auc_score(y_test, y_pred)

    # summary plot
    shap.summary_plot(center_shap_values[0], x_test, show=False)
    plt.title(f"Vanilla setting SHAP Summary Plot for Center {center_name}; AUC = {round(auc, 2)}")

    plt.savefig(f'./results/summary_plot_{center_name}.png', facecolor="white", pad_inches=0.3, bbox_inches="tight")
    plt.clf()
    plt.cla()

    # feature importance
    shap.summary_plot(center_shap_values, x_test, show=False)
    plt.title(f"Vanilla setting SHAP Feature Importance Plot for Center {center_name}; AUC = {round(auc, 2)}")

    plt.savefig(f'./results/importance_plot_{center_name}.png', facecolor="white", pad_inches=0.3, bbox_inches="tight")
    plt.clf()
    plt.cla()


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


def main():

    reset_seeds()

    for center in CENTER_NAMES:
        print(f"Start center {center}")

        X_train, y_train, X_val, y_val, X_test, y_test = load_data(center)

        model = build_model()

        train_model(model, X_train, y_train, X_val, y_val)

        calculate_metrics(model, X_test, y_test, center)

        x_test_full, y_test_full = load_x_test_full()

        calculate_SHAP(model, X_test, y_test, center, x_test_full, y_test_full)

    print("done")


if __name__ == "__main__":
    main()
