# =================================
# libraries
# =================================
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
exp = "089"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)
sim_item_path = OUTPUT_DIR / "exp" / "exp082" / "exp082_sim_item.pickle"


# =============================
# main
# =============================
test = pd.read_csv(TEST_LOG_PATH)
train = pd.read_csv(TRAIN_LOG_PATH)
train_label = pd.read_csv(TRAIN_LABEL_PATH)
yad = pd.read_csv(YADO_PATH)

train_seq_no_max = train.groupby("session_id")["seq_no"].max().to_dict()
train_label["seq_no"] = train_label["session_id"].map(train_seq_no_max)
train_label["seq_no"] += 1
train = pd.concat([train, train_label], axis=0)
train = train.sort_values(["session_id", "seq_no"]).reset_index(drop=True)

train["w"] = 1
test["w"] = 3
log = pd.concat([train, test], axis=0).reset_index(drop=True)

log = log.merge(yad, on="yad_no", how="left")
area_rank = {}
key = "sml_cd"

for i in tqdm(log["sml_cd"].unique()):
    tmp = log[log[key] == i].reset_index(drop=True)
    rank = tmp.groupby(by="yad_no")["w"].sum().sort_values(
        ascending=False).index.tolist()[:15]
    area_rank[i] = rank

test = test.merge(yad, on="yad_no", how="left")

group = test.groupby(by="session_id")
recommend = []
for s, df in tqdm(group):
    # sml_cdの最頻値
    sml_cd = df["sml_cd"].value_counts().index[0]
    if sml_cd in area_rank.keys():
        recommend_ = pd.DataFrame()
        recommend_["yad_no"] = area_rank[sml_cd]
        recommend_["rank"] = range(len(area_rank[sml_cd]))
        recommend_["session_id"] = s
    recommend.append(recommend_)

recommend = pd.concat(recommend, axis=0).reset_index(drop=True)
test_last = pd.read_parquet(TEST_LAST_PATH)
recommend = recommend.merge(test_last, on=["session_id"], how="left")
recommend = recommend[recommend["yad_no"] !=
                      recommend["last_yad_no"]].reset_index(drop=True)
recommend.to_csv(exp_dir / f"exp{exp}_test.csv", index=False)
