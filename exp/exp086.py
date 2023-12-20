# =================================
# libraries
# =================================
import pandas as pd
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
exp = "086"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)
sim_item_path = OUTPUT_DIR / "exp" / "exp082" / "exp082_sim_item.pickle"


# =============================
# main
# =============================
test = pd.read_csv(TEST_LOG_PATH)
train = pd.read_csv(TRAIN_LOG_PATH)
train_label = pd.read_csv(TRAIN_LABEL_PATH)

# rankのsort用でcountを計算
test_count = test["yad_no"].value_counts().reset_index()
test_count.columns = ["yad_no", "test_count"]

# sim_itemをload
with open(sim_item_path, "rb") as f:
    sim_item = pickle.load(f)

# top10を推薦 共起の共起
recommend = []
group = train.groupby("session_id")
item_num = 10
for s, df in tqdm(group):
    rank = {}
    items = df["yad_no"].values
    for i in items:
        if i in sim_item:
            for j, wij in sim_item[i].items():
                if j in sim_item:
                    for h, wij2 in sim_item[j].items():
                        rank.setdefault(h, 0)
                        rank[h] += wij * wij2
    if len(rank) > 0:
        recommend_ = sorted(rank.items(), key=lambda d: d[1], reverse=True)[
            :item_num]
        recommend_ = [[s, i[0], i[1]] for i in recommend_]
        recommend += recommend_
recommend = pd.DataFrame(recommend)
recommend.columns = ["session_id", "yad_no", "covisitation2"]

# sessionの長さが1のものに絞る
recommend = recommend.merge(test_count, on="yad_no", how="left")
recommend = recommend.sort_values(by=["session_id",
                                      "covisitation2", "test_count"], ascending=[
                                  True, False, False]).reset_index(drop=True)
train_count = train["session_id"].value_counts().reset_index()
train_count.columns = ["session_id", "train_count"]
train_count = train_count[train_count["train_count"]
                          == 1].reset_index(drop=True)
recommend = recommend[recommend["session_id"].isin(
    train_count["session_id"].values)].reset_index(drop=True)
label = pd.read_csv(TRAIN_LABEL_PATH)
label = label[label["session_id"].isin(
    train_count["session_id"].values)].reset_index(drop=True)
label.columns = ["session_id", "target"]
recommend = recommend.merge(label, on="session_id", how="left")
recommend["match"] = recommend["yad_no"] == recommend["target"]
recommend["match"] = recommend["match"].astype(int)
recommend = recommend[recommend["match"] == 1].reset_index(drop=True)
save_cols = ['session_id']
recommend[save_cols].to_parquet(
    exp_dir / f"exp{exp}_train.parquet", index=False)
