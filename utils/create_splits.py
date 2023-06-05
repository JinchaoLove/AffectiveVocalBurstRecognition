import os
from argparse import ArgumentParser
from collections import Counter
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

LABEL_DICT = {
    "high": [
        "high_info",
        [
            "Awe",
            "Excitement",
            "Amusement",
            "Awkwardness",
            "Fear",
            "Horror",
            "Distress",
            "Triumph",
            "Sadness",
            "Surprise",
        ],
    ],
    "two": ["two_info", ["Valence", "Arousal"]],
    "culture": [
        "culture_info",
        [
            "China_Awe",
            "China_Excitement",
            "China_Amusement",
            "China_Awkwardness",
            "China_Fear",
            "China_Horror",
            "China_Distress",
            "China_Triumph",
            "China_Sadness",
            "China_Surprise",
            "United States_Awe",
            "United States_Excitement",
            "United States_Amusement",
            "United States_Awkwardness",
            "United States_Fear",
            "United States_Horror",
            "United States_Distress",
            "United States_Triumph",
            "United States_Sadness",
            "United States_Surprise",
            "South Africa_Awe",
            "South Africa_Excitement",
            "South Africa_Amusement",
            "South Africa_Awkwardness",
            "South Africa_Fear",
            "South Africa_Horror",
            "South Africa_Distress",
            "South Africa_Triumph",
            "South Africa_Sadness",
            "South Africa_Surprise",
            "Venezuela_Awe",
            "Venezuela_Excitement",
            "Venezuela_Amusement",
            "Venezuela_Awkwardness",
            "Venezuela_Fear",
            "Venezuela_Horror",
            "Venezuela_Distress",
            "Venezuela_Triumph",
            "Venezuela_Sadness",
            "Venezuela_Surprise",
        ],
    ],
    "type": ["type_info", ["Voc_Type"]],
}

def get_data(data_dir, data_info, task):
    """ Get split ids and labels for a specific task. """
    labels = pd.read_csv(os.path.join(data_dir, "labels", LABEL_DICT[task][0] + ".csv"))
    classes = LABEL_DICT[task][1]
    df = labels.loc[:, ["File_ID", "Split"] + classes]
    df = pd.merge(df, data_info, on=["File_ID"])
    train_df = df[labels["Split"] == "Train"].reset_index(drop=True)
    val_df = df[labels["Split"] == "Val"].reset_index(drop=True)
    test_df = df[labels["Split"] == "Test"].reset_index(drop=True)
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Split Error"
    return train_df, val_df, test_df

def print_dist(df, indices=None):
    map_keys = {"China": "ZH", "United States": "US", "South Africa": "SA", "Venezuela": "VE"}
    indices = np.arange(len(df)) if indices is None else indices
    country = {map_keys[k]: v for k, v in sorted(dict(Counter(df.loc[indices, "Country"])).items())}
    voc_type = {k: v for k, v in sorted(dict(Counter(df.loc[indices, "Voc_Type"])).items())}
    return country, voc_type

def create_splits_all(data_dir, save_path, task="type", seed=42):
    """ Save file ids and labels for train, val, test, and cv_split.
        Splitted from combination of train and val w.r.t country and vocal types. """
    os.makedirs(save_path, exist_ok=True)
    data_info = pd.read_csv(os.path.join(data_dir, "labels", "data_info.csv"),
                               usecols=["File_ID", "Country"], dtype={"File_ID": str, "Country": str})
    data_info.to_csv(os.path.join(save_path, "country.csv"), index=False)
    # official train/val split
    train_df, val_df, test_df = get_data(data_dir, data_info, task=task)
    train_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "train.csv"), index=False)
    val_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "val.csv"), index=False)
    test_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "test.csv"), index=False)
    # 5-fold CV
    print(f"Default train {len(train_df)}, val {len(val_df)}")
    df = pd.concat([train_df, val_df]).reset_index(drop=True)
    spt = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_split = spt.split(X=df, y=df[[*LABEL_DICT[task][1], "Country"]])
    describes = []
    for i, (train_idxs, val_idxs) in enumerate(cv_split):
        # describe
        c_trn, v_trn = print_dist(df, train_idxs)
        c_val, v_val = print_dist(df, val_idxs)
        describes.extend([{'size': len(train_idxs)} | c_trn | v_trn, {'size': len(val_idxs)} | c_val | v_val])
        # split
        df.loc[train_idxs, ["File_ID"]].to_csv(os.path.join(save_path, f"train_{i}.csv"), index=False)
        val_df_i = df.loc[val_idxs].reset_index(drop=True)
        spt_i = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_i, tst_i = list(spt_i.split(X=val_df_i, y=val_df_i[[*LABEL_DICT[task][1], "Country"]]))[0]
        assert not set(val_idxs).intersection(set(train_idxs)) and not set(val_i).intersection(set(tst_i)), "Data leakage in splitting!"
        val_df_i.loc[val_i, ["File_ID"]].to_csv(os.path.join(save_path, f"val_{i}.csv"), index=False)
        val_df_i.loc[tst_i, ["File_ID"]].to_csv(os.path.join(save_path, f"test_{i}.csv"), index=False)
    print(pd.DataFrame(describes))
    #     size    ZH    SA     US    VE   Cry   Gasp  Groan  Grunt  Laugh  Other  Pant  Scream
    # 0  31509  8086  7919  11329  4175  2938  11157   2183   2078   7718   2194   698    2543
    # 0   7877  2033  1975   2833  1036   741   2802    505    520   1952    547   190     620
    # 1  31509  8129  7933  11253  4194  2918  11201   2151   2064   7696   2200   718    2561
    # 1   7877  1990  1961   2909  1017   761   2758    537    534   1974    541   170     602
    # 2  31508  8101  7870  11387  4150  2938  11179   2149   2104   7725   2197   698    2518
    # 2   7878  2018  2024   2775  1061   741   2780    539    494   1945    544   190     645
    # 3  31509  8058  7945  11348  4158  2936  11184   2131   2094   7775   2158   714    2517
    # 3   7877  2061  1949   2814  1053   743   2775    557    504   1895    583   174     646
    # 4  31509  8102  7909  11331  4167  2986  11115   2138   2052   7766   2215   724    2513
    # 4   7877  2017  1985   2831  1044   693   2844    550    546   1904    526   164     650

def create_splits(data_dir, save_path, task="type", seed=42):
    """ Save file ids and labels for train, val, test, and cv_split.
        Splitted from combination of train and val w.r.t country and vocal types. """
    os.makedirs(save_path, exist_ok=True)
    data_info = pd.read_csv(os.path.join(data_dir, "labels", "data_info.csv"),
                               usecols=["File_ID", "Country"], dtype={"File_ID": str, "Country": str})
    data_info.to_csv(os.path.join(save_path, "country.csv"), index=False)
    # official train/val split
    train_df, val_df, test_df = get_data(data_dir, data_info, task=task)
    train_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "train.csv"), index=False)
    val_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "val.csv"), index=False)
    test_df.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, "test.csv"), index=False)
    # 5-fold CV
    print(f"Default train {len(train_df)}, val {len(val_df)}")
    spt = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_split = spt.split(X=val_df, y=val_df[[*LABEL_DICT[task][1], "Country"]])
    describes = []
    for i, (train_idxs, val_idxs) in enumerate(cv_split):
        # describe
        c_val, v_val = print_dist(val_df, val_idxs)
        describes.append({'size': len(val_idxs)} | c_val | v_val)
        # split
        trn_df_i = pd.concat([train_df, val_df.loc[train_idxs].reset_index(drop=True)]).reset_index(drop=True)
        val_df_i = val_df.loc[val_idxs].reset_index(drop=True)
        trn_df_i.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, f"train_{i}.csv"), index=False)
        val_df_i.loc[:, ["File_ID"]].to_csv(os.path.join(save_path, f"val_{i}.csv"), index=False)
    print(pd.DataFrame(describes))
    #    size    ZH   SA    US   VE  Cry  Gasp  Groan  Grunt  Laugh  Other  Pant  Scream
    # 0  3879   965  978  1414  522  389  1342    270    260    927    283    83     325
    # 1  3879   979  960  1450  490  361  1343    291    237    982    260    85     320
    # 2  3880   998  942  1420  520  324  1386    268    252    965    270    74     341
    # 3  3879  1016  958  1347  558  399  1389    266    241    899    278    88     319
    # 4  3879  1041  966  1389  483  361  1395    227    261    957    302    91     285

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='../A-VB')
    parser.add_argument('-s', '--save_path', type=str, default='./filelists')
    args, _ = parser.parse_known_args()
    create_splits(args.data_dir, args.save_path)
