# model_name.py

import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from GCRNN import GCRNN  

def main():
    # 0) device 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) 저장된 snapshots, wor2int, 사전 학습된 단어 embeddings 및 df 불러오기
    # snapshots_embeddings.pkl 로드
    snapshots_path = "./Adressa_4w/history/snapshots.pkl"
    with open(snapshots_path, 'rb') as f:
        snapshots = pickle.load(f)
    
    print(f"Loaded snapshots' embeddings from {snapshots_path}. #={len(snapshots)}")

    # 사전 학습된 단어 로드
    word2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'word2int.tsv'), sep='\t')
    # word_to_idx, pretrained embedding
    # words = word2int['word'].tolist()
    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('./Adressa_4w/history/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)
    
    # df 로드
    def tokenize_title(title: str) -> list:
        """
        2.2) 타이틀을 공백 기준으로 단순 토크나이징
        """
        return title.split()
    
    file_path = './Adressa_4w/history/history_tkg_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    user_num = len(df['history_user'].unique())
    # 뉴스별 제목 집계
    news_info = df.groupby('clicked_news', as_index=False).agg({'title': 'first'})

    # title -> token -> index
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [word_to_idx[w] if w in word_to_idx else 0 for w in words]
    )

    # news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
    news_info_df = news_info[['clicked_news', 'title_idx']].rename(
        columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
    )

    # news_id_to_info
    news_id_to_info = news_info_df.set_index('news_id')\
        [['title_idx']].to_dict(orient='index')
        # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장
    
    
    all_news_ids = df['clicked_news'].unique()
    # news2int = {nid[1:]: i for i, nid in enumerate(all_news_ids)}

    
    # 필요한 정보 로드 끝 -------------------------------------------------------------------------------------------------
    
    # 2) GCN 초기화
    gcn_layer = GCRNN(pretrained_word_embedding, all_news_ids, news_id_to_info, user_num, emb_dim=128, batch_size=500).to(device)  
    
    # 3) snapshots에 대해 GCN 실행
    # for idx, g in tqdm(enumerate(snapshots), desc='processing GCN'):
    # forward를 통해 업데이트된 임베딩을 얻는다
    updated_user, updated_news = gcn_layer(snapshots)

    # (추가) 여기서  Loss 계산, backpropagation 등을 진행 가능

    print("\nAll snapshots GCN update finished.")

if __name__ == "__main__":
    main()
