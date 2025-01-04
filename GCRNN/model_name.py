# model_name.py
"""
모델 실행하는 코드
"""


import torch
import pickle  
import os
from tqdm import tqdm
from GCRNN import GCN  

def main():
    # 1) 저장된 snapshots 불러오기
    # snapshots_embeddings.pkl 로드
    snapshots_path = "./Adressa_4w/history/snapshots.pkl"
    with open(snapshots_path, 'rb') as f:
        snapshots = pickle.load(f)
    
    print(f"Loaded snapshots' embeddings from {snapshots_path}. #={len(snapshots)}")

    # 2) GCN 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn_layer = GCN(batch_size=500).to(device)  
    
    # 3) snapshots에 대해 GCN 실행
    for idx, g in tqdm(enumerate(snapshots), desc='processing GCN'):
        # forward를 통해 업데이트된 임베딩을 얻는다
        updated_user, updated_news = gcn_layer(g)

        # 업데이트된 임베딩을 그래프에 다시 할당
        g.nodes['user'].data['feat'] = updated_user
        g.nodes['news'].data['feat'] = updated_news

        # (추가) 여기서 LSTM(GRNN)이나 Loss 계산 등을 자유롭게 진행 가능
        # 예: GRNN을 돌리려면, 모든 스냅샷별 유저 임베딩들을 모아서
        #     (time, num_users, emb_dim) 텐서로 만든 뒤 lstm에 넣는 방식 등등.

    print("\nAll snapshots GCN update finished.")

if __name__ == "__main__":
    main()
