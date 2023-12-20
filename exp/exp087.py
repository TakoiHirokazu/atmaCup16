# =================================
# libraries
# =================================
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# =============================
# constants
# =============================
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../output")
TRAIN_LOG_PATH = DATA_DIR / "train_log.csv"
TEST_LOG_PATH = DATA_DIR / "test_log.csv"
TRAIN_LABEL_PATH = DATA_DIR / "train_label.csv"
TEST_SESSION_PATH = DATA_DIR / "test_session.csv"
YADO_PATH = DATA_DIR / "yado.csv"
TEST_LAST_PATH = OUTPUT_DIR / "exp" / "exp081" / "exp081_test_last.parquet"


# =============================
# settings
# =============================
exp = "087"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)
sim_item_path = OUTPUT_DIR / "exp" / "exp082" / "exp082_sim_item.pickle"
exp_path1 = OUTPUT_DIR / "exp" / "exp085" / "exp085_train.parquet"
exp_path2 = OUTPUT_DIR / "exp" / "exp086" / "exp086_train.parquet"

# =============================
# main
# =============================
test = pd.read_csv(TEST_LOG_PATH)
train = pd.read_csv(TRAIN_LOG_PATH)
train_label = pd.read_csv(TRAIN_LABEL_PATH)
test_last = pd.read_parquet(TEST_LAST_PATH)

# rankのsort用でcountを計算
test_count = test["yad_no"].value_counts().reset_index()
test_count.columns = ["yad_no", "test_count"]

# trainの中でtestに近いと思われるデータ
train_test1 = pd.read_parquet(exp_path1)
train_test2 = pd.read_parquet(exp_path2)
train_test_all = pd.concat(
    [train_test1, train_test2], axis=0).reset_index(drop=True)
train_test_session_id = train_test_all["session_id"].unique()

# labelのweight、testに近いと思われるtrainのweightを大きくする
train_seq_no_max = train.groupby("session_id")["seq_no"].max().to_dict()
train_label["seq_no"] = train_label["session_id"].map(train_seq_no_max)
train_label["seq_no"] += 1
train["w"] = 1
train_label["w"] = 1.5
train = pd.concat([train, train_label], axis=0)
train = train.sort_values(["session_id", "seq_no"]).reset_index(drop=True)
train = train.drop_duplicates(
    ["session_id", "yad_no"], keep="last").reset_index(drop=True)
train.loc[train["session_id"].isin(
    train_test_session_id), "w"] = train.loc[train["session_id"].isin(
        train_test_session_id), "w"] * 2


# testでlabelとなる確率が高いyad-noを擬似ラベルとする
test_label = test.copy()
test_label = test_label.merge(test_last, on="session_id", how="left")
test_label = test_label[test_label["yad_no"] != test_label["last_yad_no"]]
# 最初を残すのがスコアが高かった
test_label = test_label.drop_duplicates(["session_id"]).reset_index(drop=True)
test_label = test_label[["session_id", "yad_no"]].copy()
test_label.columns = ["session_id", "target"]
test = test.merge(test_label, on="session_id", how="left")
# 擬似ラベルのweightを8
test["w"] = 4
test["match"] = test["yad_no"] == test["target"]
test["match"] = test["match"].astype(int)
test.loc[test["match"] == 1, "w"] = 8
test_ = test.drop_duplicates(["session_id", "yad_no"]).reset_index(drop=True)

log = pd.concat([train, test_], axis=0).reset_index(drop=True)
sim_item = {}
group = log.groupby("session_id")
for s, df in tqdm(group):
    yad_no = df["yad_no"].values
    weight = df["w"].values
    for n, i in enumerate(yad_no):
        for m, j in enumerate(yad_no):
            if (n == m) | (i == j):
                continue
            sim_item.setdefault(i, {})
            sim_item[i].setdefault(j, 0)
            sim_item[i][j] += weight[m]


# sim_itemを保存
with open(exp_dir / f"exp{exp}_sim_item.pickle", "wb") as f:
    pickle.dump(sim_item, f)

# top50を推薦
recommend = []
group = test.groupby("session_id")
item_num = 50
for s, df in tqdm(group):
    rank = {}
    items = df["yad_no"].values
    for i in items:
        if i in sim_item:
            for j, wij in sim_item[i].items():
                rank.setdefault(j, 0)
                rank[j] += wij
    if len(rank) > 0:
        recommend_ = sorted(rank.items(), key=lambda d: d[1], reverse=True)[
            :item_num]
        recommend_ = [[s, i[0], i[1]] for i in recommend_]
        recommend += recommend_
recommend = pd.DataFrame(recommend)
recommend.columns = ["session_id", "yad_no", "covisitation"]

recommend = recommend.merge(test_count, on="yad_no", how="left")
recommend = recommend.sort_values(by=["session_id",
                                      "covisitation", "test_count"], ascending=[
                                  True, False, False]).reset_index(drop=True)
recommend = recommend.merge(test_last, on=["session_id"], how="left")
recommend = recommend[recommend["yad_no"] !=
                      recommend["last_yad_no"]].reset_index(drop=True)
recommend20 = []
group = recommend.groupby("session_id")
for s, df in tqdm(group):
    df["rank"] = np.arange(len(df))
    recommend20.append(df)
recommend20 = pd.concat(recommend20, axis=0).reset_index(drop=True)
recommend20 = recommend20[recommend20["rank"]
                          < 20].reset_index(drop=True)

save_cols = ['session_id', 'yad_no', 'covisitation', 'test_count', 'rank']
recommend20[save_cols].to_parquet(exp_dir / f"exp{exp}_test.parquet")
