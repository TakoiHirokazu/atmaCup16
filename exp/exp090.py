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
SUB_PATH = "../data/sample_submission.csv"


# =============================
# settings
# =============================
exp = "090"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)


# =============================
# functions
# =============================
def cust_blend(pred1, pred2, W=[3, 1], top=15):
    # Global ensemble weights
    # W = [1.15,0.95,0.85]

    # Create a list of all model predictions
    REC = []
    REC.append(pred1)
    REC.append(pred2)

    # Create a dictionary of items recommended.
    # Assign a weight according the order of appearance and multiply by global weights
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M] / (n + 1))
            else:
                res[v] = (W[M] / (n + 1))

    # Sort dictionary by item weights
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())

    # Return the top 15 itens only
    return res[:top]


# =============================
# main
# =============================
test_in_session = pd.read_parquet(
    OUTPUT_DIR / "exp" / "exp081" / "exp081_test.parquet")
covisit1 = pd.read_parquet(
    OUTPUT_DIR / "exp" / "exp087" / "exp087_test.parquet")
covisit2 = pd.read_parquet(
    OUTPUT_DIR / "exp" / "exp088" / "exp088_test.parquet")

pred_ensemble1 = pd.read_parquet(
    OUTPUT_DIR / "exp" / "exp084" / "exp084_test.parquet")

sml_rank_pred = pd.read_csv(
    OUTPUT_DIR / "exp" / "exp089" / "exp089_test.csv")

test_in_session["fe_rank"] = 0
covisit1["fe_rank"] = 1
covisit2["fe_rank"] = 1

pred1 = pd.concat([test_in_session, covisit1], axis=0).reset_index(drop=True)
pred2 = pd.concat([test_in_session, covisit2], axis=0).reset_index(drop=True)
pred1 = pred1.drop_duplicates(["session_id", "yad_no"]).reset_index(drop=True)
pred1 = pred1.sort_values(
    ["session_id", "fe_rank", "rank"]).reset_index(drop=True)
pred2 = pred2.drop_duplicates(["session_id", "yad_no"]).reset_index(drop=True)
pred2 = pred2.sort_values(
    ["session_id", "fe_rank", "rank"]).reset_index(drop=True)


# ensemle
test_session = pd.read_csv(TEST_SESSION_PATH)
session_id_list = []
ensemble_list = []
rank_list = []
group1 = pred1.groupby("session_id")
group2 = pred2.groupby("session_id")
for s in tqdm(test_session["session_id"].unique()):
    if (s in group1.groups.keys()) & (s in group2.groups.keys()):
        pred1_ = group1.get_group(s)["yad_no"].values
        pred2_ = group2.get_group(s)["yad_no"].values
        pred_all = cust_blend(pred1_, pred2_, [3, 1])
        session_id_list += [s] * len(pred_all)
        ensemble_list += pred_all
        rank_list += list(range(len(pred_all)))
    elif (s not in group2.groups.keys()) & (s in group1.groups.keys()):
        pred1_ = list(group1.get_group(s)["yad_no"].values[:15])
        session_id_list += [s] * len(pred1_)
        ensemble_list += pred1_
        rank_list += list(range(len(pred1_)))
    elif (s in group2.groups.keys()) & (s not in group1.groups.keys()):
        pred2_ = list(group2.get_group(s)["yad_no"].values[:15])
        session_id_list += [s] * len(pred2_)
        ensemble_list += pred2_
        rank_list += list(range(len(pred2_)))

pred_ensemble2 = pd.DataFrame()
pred_ensemble2["session_id"] = session_id_list
pred_ensemble2["yad_no"] = ensemble_list
pred_ensemble2["rank"] = rank_list

pred_ensemble2.to_parquet(exp_dir / f"exp{exp}_test.parquet")

session_id_list = []
pred_all_list = []
rank_list = []
group1 = pred_ensemble1.groupby("session_id")
group2 = pred_ensemble2.groupby("session_id")
for s in tqdm(test_session["session_id"].unique()):
    if (s in group1.groups.keys()) & (s in group2.groups.keys()):
        pred1_ = group1.get_group(s)["yad_no"].values
        pred2_ = group2.get_group(s)["yad_no"].values
        pred_all = cust_blend(pred1_, pred2_, [3, 1], top=10)
        session_id_list += [s] * len(pred_all)
        pred_all_list += pred_all
        rank_list += list(range(len(pred_all)))
    elif (s not in group2.groups.keys()) & (s in group1.groups.keys()):
        pred1_ = list(group1.get_group(s)["yad_no"].values[:10])
        session_id_list += [s] * len(pred1_)
        pred_all_list += pred1_
        rank_list += list(range(len(pred1_)))
    elif (s in group2.groups.keys()) & (s not in group1.groups.keys()):
        pred2_ = list(group2.get_group(s)["yad_no"].values[:10])
        session_id_list += [s] * len(pred2_)
        pred_all_list += pred2_
        rank_list += list(range(len(pred2_)))

pred_ensemble = pd.DataFrame()
pred_ensemble["session_id"] = session_id_list
pred_ensemble["yad_no"] = pred_all_list
pred_ensemble["rank"] = rank_list


pred_ensemble["fe_rank"] = 0
sml_rank_pred["fe_rank"] = 2
pred = pd.concat([pred_ensemble, sml_rank_pred], axis=0).reset_index(drop=True)
pred = pred.sort_values(
    ["session_id", "fe_rank", "rank"]).reset_index(drop=True)
sub = pd.read_csv(SUB_PATH)
test_session = pd.read_csv(TEST_SESSION_PATH)
sub_test = []
group = pred.groupby(by="session_id")
for session_id, df in tqdm(group):
    yad_no = df["yad_no"].values.tolist()
    if len(yad_no) < 10:
        yad_no = yad_no + [719] * (10 - len(yad_no))
    if len(yad_no) > 10:
        yad_no = yad_no[:10]
    sub_test_ = [session_id] + yad_no
    sub_test.append(sub_test_)
sub_test = pd.DataFrame(sub_test)
sub_test.columns = ["session_id"] + list(sub.columns)
test_session = test_session.merge(sub_test, on="session_id", how="left")
test_session = test_session.fillna(719)
save_cols = ['predict_0', 'predict_1', 'predict_2', 'predict_3',
             'predict_4', 'predict_5', 'predict_6', 'predict_7', 'predict_8',
             'predict_9']
for s in save_cols:
    test_session[s] = test_session[s].astype(int)
test_session[save_cols].to_csv(exp_dir / f"exp{exp}_sub.csv", index=False)
