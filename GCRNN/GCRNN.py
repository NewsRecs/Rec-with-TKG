from torch import nn
import dgl
import torch
from tqdm import tqdm
import gc
"""
해야 할 일들:
1. GCN 업데이트 처리 (완)
2. GRNN 유저 id로 각 g에 존재하는 유저들 embedding만 호출
3. 뉴스 인코더 추가 (완)
"""
from news_encoder import NewsEncoder
from config import Config



# ------------------------------------------------------
# GCN 클래스 정의
# ------------------------------------------------------
class GCRNN(nn.Module):
    """
    News Encoder
    뉴스 인코더도 학습이 되려면 여기에 추가해줘야 함!
    
    GCN
    유저/뉴스 노드 모두에 대해 다음을 수행:
    1) 1-hop 이웃 노드들의 임베딩과 엣지 임베딩을 element-wise product
    2) 결과를 평균처리 해준 뒤, 자기 임베딩과 더해주기 (residual-like)
    3) batch 단위(500개)로 나눠서 처리 
    
    GRNN
    유저 노드만 수행:
    1) LSTM 정의
    2) LSTM 수행
    
    Loss function
    1) 각 뉴스에 대한 negative samples 불러오기
    2) NLL loss 정의
    3) NLL loss 학습
    
    forward
    1) 그래프 정보 가져오기
    2) GCN 수행
    3) GRNN 수행
    4) Loss 계산
    5) Backpropagation
    """
    def __init__(self, pretrained_word_embedding, all_news_ids, news_id_to_info, user_num, emb_dim=128, batch_size=500):
        super(GCRNN, self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_hn = torch.zeros((user_num, emb_dim)).to(self.device)
        self.prev_cs = torch.zeros((user_num, emb_dim)).to(self.device)
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device) # input dim, hn dim
        # News_Encoder에 필요한 정보들
        self.pretrained_word_embedding = pretrained_word_embedding
        self.all_news_ids = all_news_ids
        self.news_id_to_info = news_id_to_info
        self.config = Config
        
    def News_Encoder(self):
        # NewsEncoder 초기화
        news_encoder = NewsEncoder(self.config, self.pretrained_word_embedding).to(self.device)
        # 뉴스 embeddings 생성
        news_vectors = []
        for nid in tqdm(self.all_news_ids, desc="Making news embeddings"):
            if nid in self.news_id_to_info:
                info = self.news_id_to_info[nid]
                title_idx_list = info['title_idx']
                title_tensor = torch.tensor(title_idx_list, dtype=torch.long, device=self.device)
                nv = news_encoder(title_tensor)  # shape: (1, num_filters) 가 되도록 내부 처리
                news_vectors.append(nv.squeeze(0))
            else:
                news_vectors.append(torch.randn(self.config.num_filters, device=self.device))

        news_embeddings = torch.stack(news_vectors)  # (전체 뉴스 수, num_filters)
        
        return news_embeddings
            
    def batch_GCN(self, g, num_edges, message_func, reduce_func, etype):
        for start in range(0, num_edges, self.batch_size):
            end = start + self.batch_size
            end = min(end, num_edges)
            edge_ids = list(range(start, end))  # 마지막 배치 처리 시 범위 초과 방지
            
            # message passing
            g.send_and_recv(
                edges=edge_ids,
                message_func=message_func,
                reduce_func=reduce_func,
                etype=etype
            )
            # if etype == 'clicked':
            #     updated_feat = g.nodes['news'].data['new_feat']
            # else:   # reverse_click
            #     updated_feat = g.nodes['user'].data['new_feat']
            # g.nodes['user'].data['feat'] = updated_feat
        
        # edge_ids = list(range(num_edges))
        # # message passing
        # g.send_and_recv(
        #     edges=edge_ids,
        #     message_func=message_func,
        #     reduce_func=reduce_func,
        #     etype=etype
        # )
        # if etype == 'clicked':
        #     g.nodes['user'].data['feat'] = g.nodes['news'].data['new_feat']
        # else:   # reverse_click
        #     g.nodes['user'].data['feat'] = g.nodes['user'].data['new_feat']
            
        return
        
    def GRNN(self, g):
        """
        1. 유저 id 순대로 모든 embeddings를 tensor로 초기화 (약 85000개) - shape: (user 수, 128), 값: 0
        *** 1의 tensor는 GRNN의 temporal embeddings
        2. 이전 상태 c, h를 노드마다 갖도록 하고, 이를 불러와서 lstm 생성
        """
        g_user_ids = g.nodes['user'].data['user_ids'].tolist()

        prev_hn = self.prev_hn[g_user_ids].to(self.device)
        prev_cs = self.prev_cs[g_user_ids].to(self.device)
        
        user_input = g.nodes['user'].data['feat']
        user_hn, user_cs = self.user_RNN(user_input, (prev_hn, prev_cs))
        self.prev_hn[g_user_ids] = user_hn
        self.prev_cs[g_user_ids] = user_cs        
        
        return 
        
    # def negative_samples(self):
        
        
    # def NLL_loss(self):
        
    
    def forward(self, snapshots: dgl.DGLHeteroGraph):
        """
        g: DGL 이종 그래프
           - etype = ('user', 'clicked', 'news')
           - g.nodes['user'].data['feat'], g.nodes['news'].data['feat'] 에는
             노드 임베딩이 저장되어 있음.
           - g.edges['clicked'].data['feat'] 에는 user -> news 방향의 엣지 임베딩이 저장되어 있음.
           - g.edges['clicked_reverse'].data['feat'] 에는 news-> user 방향의 엣지 임베딩이 저장되어 있음.
           - g.nodes['user'].data['user_ids']에는 user_ids가 tensor 형태로 저장되어 있음. (user_num, 1)
           - g.nodes['news'].data['news_ids']에는 news_ids가 tensor 형태로 저장되어 있음. (news_num, 1)
        
        snapshots_user_ids: snapshot마다 저장된 유저 id

        return:
            updated_user_feats: (num_users, emb_dim)
            updated_news_feats: (num_news, emb_dim)
        """
        # print(f"[Before] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
        
        news_embeddings = self.News_Encoder()
        for i, g in tqdm(enumerate(snapshots), desc='Processing GCRNN', total=1680):
            # 뉴스 노드 embeddings 할당
            snapshot_news_ids = g.nodes['news'].data['news_ids'].cpu().numpy()
            # snapshot_news_indices = [self.news2int[u] for u in snapshot_news_ids]   # g에 저장된 뉴스 노드 순서대로 뉴스의 고유 idx
            snapshot_news_embeddings = news_embeddings[snapshot_news_ids].to(self.device)
            
            g.nodes['news'].data['feat'] = snapshot_news_embeddings
            
            # 2) GCN
            
            # 유저 임베딩, 뉴스 임베딩, 엣지 임베딩 가져오기
            # user_feats = g.nodes['user'].data['feat']  # (num_users, emb_dim)
            # news_feats = g.nodes['news'].data['feat']  # (num_news, emb_dim)
            edge_feats_news = g.edges['clicked'].data['feat']  # (num_edges, emb_dim)
            # edge_feats_user = g.edges['clicked_reverse'].data['feat']  # (num_edges, emb_dim)
            num_edges = len(edge_feats_news)

            # 최종적으로 업데이트된 임베딩을 저장할 텐서(초기: 기존 임베딩 복사)
            # updated_user_feats = user_feats.clone()
            # updated_news_feats = news_feats.clone()

            # ------------------------------------------------------
            # (A) 유저 노드에 대한 GCN
            # ------------------------------------------------------
            # num_users = user_feats.shape[0]
            
            def message_func(edges):
                # 엣지 메시지는 뉴스 임베딩과 엣지 임베딩의 element-wise product
                return {'msg': edges.src['feat'] * edges.data['feat']}

            def reduce_func(nodes):
                # 메시지를 상수 c로 나눈 뒤, 기존 임베딩과 더해줌 (residual)
                # aggregated = torch.sum(nodes.mailbox['msg'], dim=1) / self.c
                aggregated = nodes.mailbox['msg'].mean(1)
                return {'feat': aggregated + nodes.data['feat']}   # new_feat -> feat: GCN 결과가 바로 update되도록 바꿈
            
            
            # 뉴스 노드에서 유저 노드로의 메시지 전달 및 집계
            etype = 'clicked_reverse'
            self.batch_GCN(g, num_edges, message_func, reduce_func, etype)

            # print(f"[After clicked_reverse] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
                    
            # ------------------------------------------------------
            # (B) 뉴스 노드에 대한 GCN
            # ------------------------------------------------------     
            
            # 유저 노드에서 뉴스 노드로의 메시지 전달 및 집계
            # g.update_all(message_func, reduce_func, etype='clicked')
            etype = 'clicked'
            self.batch_GCN(g, num_edges, message_func, reduce_func, etype)
            
            # print(f"[After clicked] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
            
            # 업데이트된 뉴스 임베딩 추출
            # updated_news_feats = g.nodes['news'].data['new_feat']

            
            # 3) GRNN
            self.GRNN(g)
            
            print(f"[After GRNN] GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device)/1024**2:.2f} MiB")
            
            
        
        return #updated_user_feats, updated_news_feats


