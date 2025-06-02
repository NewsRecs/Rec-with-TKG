# -*- coding: utf-8 -*-
"""
7w Adressa 데이터 – 2·4·6-hop 이웃 계산 + 36 h 내 valid 비율
GPU(CUDA) 연산으로 메모리 이슈 해결.
"""

import torch, pandas as pd, numpy as np, time
from dgl.data.utils import load_graphs
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# ----------------------- 0. 설정 -----------------------
DATA_DIR     = Path("./psj/Adressa_4w")     # ← 7-week 경로
MAX_DELTA_H  = 36
MAX_DELTA_S  = MAX_DELTA_H * 3600
device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device)

# ----------------------- 0-1. GPU 안전 곱 -----------------------
def safe_sparse_mm(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    cuSPARSE SpGEMM 자원 부족 시 CPU로 fall-back 후 다시 원래 device로 복귀.
    """
    try:
        return torch.sparse.mm(mat1, mat2).coalesce()
    except RuntimeError as e:
        if "cusparseSpGEMM" in str(e):
            print("⚠️  GPU SpGEMM 실패 → CPU 재계산")
            res = torch.sparse.mm(mat1.cpu(), mat2.cpu()).coalesce()
            return res.to(mat1.device)
        raise

# ----------------------- 1. 그래프 로드 & 2-hop -----------------------
# → load_graphs 에는 str(path) 형태로 넘겨야 에러가 나지 않습니다.
g = load_graphs(str(DATA_DIR / "datas/total_graph_pre_experiment.bin"))[0][0]

# ── bipartite 뉴스→유저 행렬 A 만들기 ──────────────────────────────
news2int = pd.read_csv(DATA_DIR / "history/news2int.tsv", sep="\t")
news_num   = len(news2int)
num_total  = g.number_of_nodes()
num_user   = num_total - news_num
news_start = num_user

src, dst = g.edges()                      # 전역 ID
mask_fwd      = (src >= news_start) & (dst < news_start)   # 뉴스→유저
news_local_id = (src[mask_fwd] - news_start).long()
user_local_id = dst[mask_fwd].long()

# (news_num × user_num) sparse COO 값 = 1.0 (float16 → VRAM 절약)
indices = torch.stack([news_local_id, user_local_id])
values  = torch.ones(indices.shape[1], dtype=torch.float32)
A = torch.sparse_coo_tensor(indices, values,
                            size=(news_num, num_user),
                            device=device).coalesce()

# ── 2-hop(뉴스↔뉴스) ───────────────────────────────────────────────
At = torch.sparse_coo_tensor(
        torch.stack([A.indices()[1], A.indices()[0]]),
        A.values(), (num_user, news_num),
        device=device).coalesce()

adj2_incl = safe_sparse_mm(A, At)                # 포함형 2-hop
# 대각 제거
idx2     = adj2_incl.indices()
mask2    = idx2[0] != idx2[1]
adj2     = torch.sparse_coo_tensor(idx2[:, mask2],
                                   adj2_incl.values()[mask2],
                                   (news_num, news_num),
                                   device=device).coalesce()
print("2-hop nnz =", adj2._nnz())

# ----------------------- 2. 4-hop & 6-hop --------------------------
adj4_incl = safe_sparse_mm(adj2, adj2)           # 포함형 4-hop
adj6_incl = safe_sparse_mm(adj4_incl, adj2)      # 포함형 6-hop

# ── “정확히” 4-hop = 4-hop \ 2-hop ────────────────────────────────
def _exact_diff(big: torch.Tensor, small: torch.Tensor):
    """big − small (둘 다 same shape sparse)."""
    big_idx   = big.indices().cpu()
    small_set = set(zip(small.indices()[0].cpu().tolist(),
                        small.indices()[1].cpu().tolist()))
    keep = [(r, c) not in small_set
            for r, c in zip(big_idx[0], big_idx[1])]
    keep_mask = torch.tensor(keep, dtype=torch.bool)
    return torch.sparse_coo_tensor(
        big_idx[:, keep_mask],
        big.values().cpu()[keep_mask],
        big.shape).to(device).coalesce()

adj4 = _exact_diff(adj4_incl, adj2)              # 정확히 4-hop
union_2_4 = torch.cat([adj2.indices().t(), adj4_incl.indices().t()], dim=0)
union_set = set(map(tuple, union_2_4.cpu().tolist()))
# ── “정확히” 6-hop = 6-hop \ (2 ∪ 4) ──────────────────────────────
idx6 = adj6_incl.indices().cpu()
keep6 = torch.tensor([(r, c) not in union_set
                      for r, c in zip(idx6[0], idx6[1])])
adj6 = torch.sparse_coo_tensor(
    idx6[:, keep6],
    adj6_incl.values().cpu()[keep6],
    adj6_incl.shape).to(device).coalesce()

print("4-hop nnz =", adj4._nnz())
print("6-hop nnz =", adj6._nnz())

# ----------------------- 3. hop별 이웃 딕셔너리 --------------------
def _sparse_to_dict(mat: torch.Tensor):
    r, c = mat.indices()
    d = {}
    for rr, cc in zip(r.tolist(), c.tolist()):
        d.setdefault(rr, []).append(cc)
    return d

two_hop  = _sparse_to_dict(adj2)
four_hop = _sparse_to_dict(adj4)
six_hop  = _sparse_to_dict(adj6)

# ----------------------- 4. 발행 시각 병합 -------------------------
pub_df = pd.read_csv(DATA_DIR / "datas/news_publish_times.tsv", sep="\t")
pub_df["news_id"] = pub_df["news_id"].astype(str).str.strip()
pub_df["publish_time"] = pd.to_datetime(pub_df["publish_time"])

news2int["news_id"] = news2int["news_id"].astype(str).str.strip()
merged = pd.merge(news2int, pub_df, on="news_id", how="left")
merged = merged.dropna(subset=["publish_time"])
int2time = dict(zip(merged["news_int"], merged["publish_time"]))

def _map_to_times(hop_dict):
    return {k: [int2time[n] for n in v if n in int2time]
            for k, v in hop_dict.items()}

neighbors_dicts = {
    2: _map_to_times(two_hop),
    4: _map_to_times(four_hop),
    6: _map_to_times(six_hop)
}

# ----------------------- 5. 36 h valid 비율 ------------------------
ratio_per_hop = {}
for k, nbr_dict in neighbors_dicts.items():
    t0 = time.time()
    tot = val = 0
    for nid, nbr_times in nbr_dict.items():
        main_t = int2time[nid]
        thres  = main_t - timedelta(hours=MAX_DELTA_H)
        v = sum(thres <= t <= main_t for t in nbr_times)
        val += v
        tot += len(nbr_times)
    ratio_per_hop[k] = val / tot if tot else float("nan")
    print(f"{k}-hop: total={tot:,}, valid={val:,}, "
          f"ratio={ratio_per_hop[k]:.4f}, {time.time()-t0:.1f}s")

print(">>> hop별 36 h valid ratio =", ratio_per_hop)
