# =================================
# libraries
# =================================
import pandas as pd
from pathlib import Path

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


# =============================
# settings
# =============================
exp = "081"
exp_dir = OUTPUT_DIR / "exp" / f"exp{exp}"
exp_dir.mkdir(parents=True, exist_ok=True)


# =============================
# main
# =============================
test = pd.read_csv(TEST_LOG_PATH)
test_last = test.drop_duplicates(
    subset="session_id", keep="last").reset_index(drop=True)
test_last = test_last[["session_id", "yad_no"]].reset_index(drop=True)
test_last.columns = ["session_id", "last_yad_no"]
test_last.to_parquet(exp_dir / f"exp{exp}_test_last.parquet")

test = test.merge(test_last, on="session_id", how="left")
test = test[test["yad_no"] != test["last_yad_no"]].reset_index(drop=True)
test = test.drop_duplicates(
    subset=["session_id", "yad_no"]).reset_index(drop=True)
test["rank"] = test["seq_no"].values
test[["session_id", "yad_no", "rank"]].to_parquet(
    exp_dir / f"exp{exp}_test.parquet")
