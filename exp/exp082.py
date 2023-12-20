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
exp = "082"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)


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

# testでlabelとなる確率が高いyad-noを擬似ラベルとする
test_label = test.copy()
test_label = test_label.merge(test_last, on="session_id", how="left")
test_label = test_label[test_label["yad_no"] != test_label["last_yad_no"]]
# 最初を残すのがスコアが高かった
test_label = test_label.drop_duplicates(["session_id"]).reset_index(drop=True)
test_label = test_label[["session_id", "yad_no"]].copy()
test_label.columns = ["session_id", "target"]
test = test.merge(test_label, on="session_id", how="left")
# 擬似ラベルのweightを2
test["w"] = 1
test["match"] = test["yad_no"] == test["target"]
test["match"] = test["match"].astype(int)
test.loc[test["match"] == 1, "w"] = 2
test_ = test.drop_duplicates(["session_id", "yad_no"]).reset_index(drop=True)

# 共起回数の計算
log = test_.copy()
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

# covisitation、test_countの順にsort
recommend = recommend.merge(test_count, on="yad_no", how="left")
recommend = recommend.sort_values(by=["session_id", "covisitation", "test_count"],
                                  ascending=[
                                  True, False, False]).reset_index(drop=True)

# sessionの最後と同じyad-noを削除
recommend = recommend.merge(test_last, on=["session_id"], how="left")
recommend = recommend[recommend["yad_no"] !=
                      recommend["last_yad_no"]].reset_index(drop=True)

# top20に絞る
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
