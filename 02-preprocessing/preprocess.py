import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ["r_gender", "age", "r_ivtrom", "r_treall", "b_pvstr", "b_pvdm", "b_pvrr", "b_pvaf", "b_pvhc", "nihsco_abl_c", "StudySubjectID"]
TARGET_FEATURE = "mrs_d90d_c"

BIGGEST_CENTERS = ["ANT",
                   "AMC",
                   "LUM",
                   "AZM",
                   "RIJ",
                   "MCH"]


def load_data() -> pd.DataFrame:
    print("Loading data")
    return pd.read_csv("clinic.csv")


def select_subset(df: pd.DataFrame, col_names) -> pd.DataFrame:
    return df[col_names]


def show_value_distribution(df: pd.DataFrame):
    distribution = df.value_counts()
    plt.pie(x=distribution.values, labels=distribution.index)
    plt.show()


def preprocess_y(data: pd.DataFrame):
    y_raw = select_subset(data, TARGET_FEATURE)

    y_val_to_replace = {'mRS 0 - No symptoms (code 6)': 0,
                        'mRS 1 - Minor symptoms, no limitations (code 5)': 0,
                        'mRS 2 - Slight disability, no help needed (code 4)': 0,
                        'mRS 3 - Moderate disability, still independent (code 3)': 1,
                        'mRS 4 - Moderately severe disability (code 2)': 1,
                        'mRS 5 - Severe disability, completely dependent (code 1)': 1,
                        'mRS 6 - Death (code 0)': 1}
    y_ready = y_raw.replace(y_val_to_replace)
    return y_ready


def preprocess_x(data: pd.DataFrame) -> pd.DataFrame:
    x_raw = select_subset(data, FEATURE_NAMES)

    x_raw["StudySubjectID"] = x_raw["StudySubjectID"].apply(lambda s: s[:3])

    x_val_to_replace = {'Male' : 0,
                        'Female' : 1,
                        'Yes' : 1,
                        'No' : 0,
                        '1 - Intra-arterial treatment' : 1,
                        '0 - No intra-arterial treatment' : 0
                        }
    return x_raw.replace(x_val_to_replace).rename({"r_gender": "female", 'r_treall': 'intra_arterial_treatment'}, axis="columns")


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("Preprocess data")
    return pd.concat((preprocess_x(data), preprocess_y(data)), axis="columns")


def keep_only_centers(df: pd.DataFrame, centers: list[str]) -> pd.DataFrame:
    return df[df["StudySubjectID"].isin(centers)]


TRAIN_RATIO = 0.75
VAL_RATIO = 0.05
TEST_RATIO = 0.20

# TRAIN_RATIO = 0.65
# VAL_RATIO = 0.15
# TEST_RATIO = 0.20


def split_and_save_df_centers(df: pd.DataFrame):
    assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Split does not sum up to 100%"

    # empty arrays to save data from each center to get a split for full data set
    all_train_sets_org = []
    all_val_sets_org = []
    all_test_sets_org = []

    all_train_sets_norm = []
    all_val_sets_norm = []
    all_test_sets_norm = []

    # do for all centers:
    #   get subset for each center
    #   split train, val, test set
    #   save data sets without normalisation
    #   normalise date based on training data statistics
    #   save normalised data
    for center in df["StudySubjectID"].value_counts().index.values.tolist():
        print(f"split and save center {center}")

        df_center = df[df["StudySubjectID"] == center]

        X_center = df_center.iloc[:, :-1]
        y_center = df_center.iloc[:, -1]

        X_train_center, X_test_temp_center, y_train_center, y_test_temp_center = train_test_split(X_center, y_center, test_size=1-TRAIN_RATIO)

        X_val_center, X_test_center, y_val_center, y_test_center = train_test_split(X_test_temp_center, y_test_temp_center, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO))

        # show distribution
        show_value_distribution(y_train_center)
        show_value_distribution(y_val_center)
        show_value_distribution(y_test_center)


        # save DFs before normalisation
        train_concat_org = pd.concat((X_train_center, y_train_center), axis="columns")
        val_concat_org = pd.concat((X_val_center, y_val_center), axis="columns")
        test_concat_org = pd.concat((X_test_center, y_test_center), axis="columns")

        train_concat_org.to_csv(f"./datasets/train_org_{center}.csv", index=False)
        val_concat_org.to_csv(f"./datasets/val_org_{center}.csv", index=False)
        test_concat_org.to_csv(f"./datasets/test_org_{center}.csv", index=False)

        all_train_sets_org.append(train_concat_org)
        all_val_sets_org.append(val_concat_org)
        all_test_sets_org.append(test_concat_org)

        # calculate normalisation
        NIHSS_mean_center = X_train_center["nihsco_abl_c"].mean()
        age_mean_center = X_train_center["age"].mean()

        NIHSS_std_center = X_train_center["nihsco_abl_c"].std()
        age_std_center = X_train_center["age"].std()

        X_train_center["nihsco_abl_c"] = (X_train_center["nihsco_abl_c"] - NIHSS_mean_center) / NIHSS_std_center
        X_train_center["age"] = (X_train_center["age"] - age_mean_center) / age_std_center

        X_val_center["nihsco_abl_c"] = (X_val_center["nihsco_abl_c"] - NIHSS_mean_center) / NIHSS_std_center
        X_val_center["age"] = (X_val_center["age"] - age_mean_center) / age_std_center

        X_test_center["nihsco_abl_c"] = (X_test_center["nihsco_abl_c"] - NIHSS_mean_center) / NIHSS_std_center
        X_test_center["age"] = (X_test_center["age"] - age_mean_center) / age_std_center

        # save DFs before normalisation
        train_concat_norm = pd.concat((X_train_center, y_train_center), axis="columns")
        val_concat_norm = pd.concat((X_val_center, y_val_center), axis="columns")
        test_concat_norm = pd.concat((X_test_center, y_test_center), axis="columns")

        train_concat_norm.to_csv(f"./datasets/train_norm_{center}.csv", index=False)
        val_concat_norm.to_csv(f"./datasets/val_norm_{center}.csv", index=False)
        test_concat_norm.to_csv(f"./datasets/test_norm_{center}.csv", index=False)

        all_train_sets_norm.append(train_concat_norm)
        all_val_sets_norm.append(val_concat_norm)
        all_test_sets_norm.append(test_concat_norm)

    # concat all subsets from centers to create a full dataset
    pd.concat(all_train_sets_org).to_csv(f"./datasets/train_org_full.csv", index=False)
    pd.concat(all_train_sets_norm).to_csv(f"./datasets/train_norm_full.csv", index=False)

    pd.concat(all_val_sets_org).to_csv(f"./datasets/val_org_full.csv", index=False)
    pd.concat(all_val_sets_norm).to_csv(f"./datasets/val_norm_full.csv", index=False)

    pd.concat(all_test_sets_org).to_csv(f"./datasets/test_org_full.csv", index=False)
    pd.concat(all_test_sets_norm).to_csv(f"./datasets/test_norm_full.csv", index=False)


def split_random_centers(df_name: str, split_ratio_list: np.ndarray):
    # filename is train_org / val_org / test_org / train_norm / val_norm / test_norm

    df = pd.read_csv(f"./datasets/{df_name}_full.csv")

    df_len = len(df)

    # shuffle df
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # calculate number pf members per artificial center and round to int
    split_list_numbers = (split_ratio_list * df_len).round(0).astype(int).tolist()

    print(f"list ", split_list_numbers)

    index = 1
    for size in split_list_numbers:
        print("size")
        print(size)
        print(type(size))
        artificial_center = df_shuffled.iloc[:size, :]
        # remove data from this new center
        df_shuffled = df_shuffled.iloc[size:, :]

        artificial_center.to_csv(f"./datasets/{df_name}_random{index}.csv", index=False)
        index = index + 1


def generate_random_centers(df: pd.DataFrame):

    print(f"Dataframe {df}")

    # get sizes of six biggest sets to simulate
    size_list = df["StudySubjectID"].value_counts().iloc[:6]

    print(f"size_list {size_list}")

    total_len = len(df)
    print(f"total_len {total_len}")

    ratio_list = np.array(size_list) / total_len

    print("rato list", ratio_list)

    # split full dataset  centers
    split_random_centers("train_norm", ratio_list)
    split_random_centers("val_norm", ratio_list)
    split_random_centers("test_norm", ratio_list)


def load_df_biggest_centers():
    train = pd.read_csv("./datasets/train_norm_full.csv")
    val = pd.read_csv("./datasets/val_norm_full.csv")
    test = pd.read_csv("./datasets/test_norm_full.csv")

    return pd.concat([test, val, train])


def main():
    print("Start prepare data")
    clinic_data = load_data()

    #preprocessed_data = preprocess_data(clinic_data)

    #biggest_centers = keep_only_centers(preprocessed_data, BIGGEST_CENTERS)

    #split_and_save_df_centers(biggest_centers)

    biggest_centers = load_df_biggest_centers()

    generate_random_centers(biggest_centers)

    print("done")


if __name__ == "__main__":
    main()
#%%
