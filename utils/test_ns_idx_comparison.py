import torch
import pickle
import os
from tqdm import tqdm
import pandas as pd
import pickle
import dgl
import numpy as np

"""
1-Week(1w) Adressa 데이터에서 test_ns.tsv 를 생성하는 스크립트.
 - train_ns.tsv  : tkg_test_negative_samples_lt36_ns20.tsv 를 news_int 로 매핑 후 가중치·remaining 저장
 - publish_time  : news_publish_times.tsv
 - output        : ./psj/Adressa_1w/test/test_ns.tsv

변경 사항
---------
* 3w → 1w 경로 변경
* LANCER 식(5) 기반 weight 계산 로직 리팩터
* weight 계산 전 remaining_lifetime(시간) 컬럼 추가
"""

# ────────────────────────────────────────────────────────────────────────────────
# 경로 설정 ▶ 1‑Week 데이터셋 ----------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
WEEK = 3
ROOT_DIR            = f"./psj/Adressa_{WEEK}w"   # ✅ 3w → 1w
DATAS_DIR           = f"{ROOT_DIR}/datas"
TEST_DIR            = f"{ROOT_DIR}/test"

train_news_file_path = f"{DATAS_DIR}/all_news_nyheter_splitted.tsv"
train_ns_path        = f"{TEST_DIR}/tkg_test_negative_samples_lt36_ns20.tsv"
train_file_path      = f"{DATAS_DIR}/{WEEK}w_behaviors.tsv"
news2int_file_path   = f"{DATAS_DIR}/news2int.tsv"
user2int_file_path   = f"{DATAS_DIR}/user2int.tsv"
pub_path             = f"{DATAS_DIR}/news_publish_times.tsv"
OUTPUT_PATH         = f"{TEST_DIR}/test_ns_comparison.tsv"

# ────────────────────────────────────────────────────────────────────────────────
# 데이터 로드 -------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
# 뉴스 카테고리 --------------------------------------------------------------
train_news_df = pd.read_csv(train_news_file_path, sep="\t", names=["index_col", "newsId", "category", "subcategory", "title"])
sub_train_news_df = train_news_df[["newsId", "category"]]

# train_ns(neg‑sample) --------------------------------------------------------
train_ns = pd.read_csv(train_ns_path, sep="\t")

# behaviors train set ---------------------------------------------------------
df = pd.read_csv(train_file_path, sep="\t", encoding="utf-8")
df["click_time"] = pd.to_datetime(df["click_time"])
df["clicked_news"] = df["clicked_news"].str.replace(r"-\d+$", "", regex=True)

if WEEK == 1:
    criteria_time1 = pd.Timestamp("2017-01-05 00:00:00")   # 1‑week 범위로 재설정 (예시)
    criteria_time2 = pd.Timestamp("2017-01-12 00:00:00")
elif WEEK == 3:
    criteria_time1 = pd.Timestamp("2017-01-05 00:00:00")   # 1‑week 범위로 재설정 (예시)
    criteria_time2 = pd.Timestamp("2017-01-26 00:00:00")
else:
    criteria_time1 = pd.Timestamp("2017-01-01 00:00:00")   # 1‑week 범위로 재설정 (예시)
    criteria_time2 = pd.Timestamp("2017-02-19 00:00:00")
train_df = df[(criteria_time1 <= df["click_time"]) & (df["click_time"] < criteria_time2)]
train_df = train_df.merge(sub_train_news_df, left_on="clicked_news", right_on="newsId", how="left")
train_df = train_df.dropna(subset=["clicked_news"])

# news2int / user2int ---------------------------------------------------------
news2int = dict(pd.read_csv(news2int_file_path, sep="\t").values)
user2int = dict(pd.read_csv(user2int_file_path, sep="\t").values)

# train_ns 매핑 ---------------------------------------------------------------
train_ns["news_int"] = train_ns["clicked_news"].map(news2int)

def map_negative_samples(ns_str):
    if pd.isna(ns_str):
        return ns_str
    return " ".join([str(news2int.get(nid, -1)) for nid in ns_str.split()])
train_ns["negative_samples"] = train_ns["negative_samples"].apply(map_negative_samples)
train_ns["user_int"] = train_ns["user"].map(user2int)

# ────────────────────────────────────────────────────────────────────────────────
# publish_time 로드 -------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
pub_df = pd.read_csv(pub_path, sep="\t", usecols=["news_id", "publish_time"])
pub_df["publish_time"] = pd.to_datetime(pub_df["publish_time"])
pub_df["news_int"] = pub_df["news_id"].map(news2int)

pub_dict = dict(zip(pub_df["news_int"], pub_df["publish_time"]))

# 각 행에 pub‑time 리스트 매핑 ---------------------------------------------------
train_ns["click_time"] = pd.to_datetime(train_ns["click_time"])
train_ns["publish_times_list"] = (
    train_ns["publish_times"]
    .str.split(',')
    .apply(lambda lst: pd.to_datetime(lst))
)
train_ns["pos_publish_time"] = train_ns["news_int"].map(pub_dict)

# ────────────────────────────────────────────────────────────────────────────────
# LANCER 식(5) 기반 remaining / weight 계산 -------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
ALPHA          = 0.1          # 기울기
LIFETIME_HOUR  = 36           # |ltime| (threshold)

def calc_remaining(click_t: pd.Timestamp, pub_t: pd.Timestamp) -> float:
    """남은 수명(시간) = LIFETIME_HOUR - 경과시간"""
    return LIFETIME_HOUR - (click_t - pub_t).total_seconds() / 3600.0

def weight_from_remaining(remaining: float) -> float:
    """식 (5) : alpha (alpha·remaining)"""
    return 1.0 / (1.0 + np.exp(-ALPHA * remaining))

# per‑row 처리 -----------------------------------------------------------------

def compute_row_values(row):
    click_t = row["click_time"]

    # positive --------------------------------------------------------------
    pos_pub = row["pos_publish_time"]
    remaining_list   = []
    weight_list      = []

    rem_pos = calc_remaining(click_t, pos_pub)
    remaining_list.append(f"{rem_pos:.2f}")
    weight_list.append(f"{weight_from_remaining(rem_pos):.6f}")

    # negatives -------------------------------------------------------------
    for neg_pub_t in row["publish_times_list"]:
        rem_neg = calc_remaining(click_t, neg_pub_t)
        remaining_list.append(f"{rem_neg:.2f}")
        weight_list.append(f"{weight_from_remaining(rem_neg):.6f}")

    return " ".join(weight_list), " ".join(remaining_list)

train_ns[["candidate_weight", "remaining_lifetime"]] = train_ns.apply(
    compute_row_values,
    axis=1,
    result_type="expand",
)

# 필요없는 임시 컬럼 제거 --------------------------------------------------------
train_ns.drop(columns=["pos_publish_time", "publish_times_list"], inplace=True)

# ────────────────────────────────────────────────────────────────────────────────
# 저장 -------------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────────
train_ns.to_csv(OUTPUT_PATH, sep="\t", index=False)

print(f"✅ test_ns.tsv 저장 완료 → {OUTPUT_PATH}")
