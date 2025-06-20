from torch import nn
import dgl
import torch
from tqdm import tqdm
# from utils import binary_search
import numpy as np
import gc
import random
from utils.nce_loss import NCELoss
from typing import Optional, List

from model.config_3w import Config
if Config.method == 'cnn_attention':
    if Config.no_category:
        from utils.title_news_encoder import NewsEncoder
    else:
        from utils.full_news_encoder import NewsEncoder
else:
    if Config.no_category:
        from utils.title_MSA_news_encoder import NewsEncoder
    else:    
        from utils.MSA_news_encoder import NewsEncoder

torch.cuda.set_device(Config.gpu_num)
random_seed = Config.seed
random.seed(random_seed)
torch.manual_seed(random_seed)

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
    def __init__(self, all_news_ids, news_id_to_info, user_num, cat_num, news_num, pretrained_word_embedding=None, emb_dim=100, batch_size=300, snapshots_num=1680):
        """
        학습 대상
        1. user embeddings
        2. category (edge) embeddings
        3. GRNN parameters
        4. NewsEncoder parameters
        *** GCN은 따로 파라미터가 존재하지 않음
        *** g의 유저, 뉴스 노드 idx는 user2int와 news2int의 순서대로 만들어짐 (edge도 마찬가지)
        """
        super(GCRNN, self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.snapshots_num = snapshots_num
        self.device = torch.device(f"cuda:{Config.gpu_num}" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.user_embedding_layer = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN user 초기값
        self.cat_embedding_layer = nn.Embedding(num_embeddings=cat_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN relation 초기값
        # GRNN에 필요한 변수들
        self.user_num = user_num
        self.cat_num = cat_num
        self.news_num = news_num
        # self.prev_hn = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # for hidden state in LSTM_GCN
        self.c0_embedding_layer_u = nn.Embedding(num_embeddings=user_num+news_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # for cell state in LSTM_GCN
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device)   # input dim, hn dim
        # News_Encoder에 필요한 정보들
        self.config = Config
        self.pretrained_word_embedding = pretrained_word_embedding
        self.news_encoder = NewsEncoder(self.config, self.pretrained_word_embedding).to(self.device)
        self.all_news_ids = all_news_ids   # news_int 순서대로 id 저장됨
        self.news_id_to_info = news_id_to_info
        
        
    # def News_Encoder(self, news_ids):
    #     # print("Processing news embeddings \n")
    #     # 뉴스 embeddings 생성
    #     news_embeddings = torch.zeros((len(news_ids), self.emb_dim)).to(self.device)   # (전체 뉴스 수, 뉴스 embedding 차원)
    #     for i, nid in enumerate(news_ids):
    #         if nid in self.news_id_to_info:
    #             info = self.news_id_to_info[nid]
    #             title_idx_list = info['title_idx']
    #             title_tensor = torch.tensor(title_idx_list, dtype=torch.long, device=self.device)
    #             category_idx = info['category_idx']
    #             subcategory_idx = info['subcategory_idx']
    #             nv = self.news_encoder(title_tensor, category_idx, subcategory_idx)  # shape: (1, num_filters) 가 되도록 내부 처리
    #             news_embeddings[i] = nv.squeeze(0)
    #         else:
    #             news_embeddings[i] = torch.randn(self.emb_dim, device=self.device)
        
    #     return news_embeddings

    def News_Encoder(self, news_ids, max_batch: int = 512):
        """
        news_ids  : 리스트(int) - news_int 순서
        max_batch : 한 배치에 넣을 최대 뉴스 수
        """
        news_embeddings = torch.zeros((len(news_ids), self.emb_dim)).to(self.device)   # (전체 뉴스 수, 뉴스 embedding 차원)

        batch_titles, batch_cats, batch_scats, batch_idx = [], [], [], []

        def _flush():
            # _flush() 시작 부분
            assert all(0 <= idx < len(news_ids) for idx in batch_idx), \
                f"batch_idx 범위 초과! max={max(batch_idx)}, news_ids len={len(news_ids)}"

            if not batch_titles:
                return
            # ① title 텐서 패딩 — 길이가 다른 타이틀을 한 텐서로
            padded = nn.utils.rnn.pad_sequence(
                [torch.tensor(t, dtype=torch.long) for t in batch_titles],
                batch_first=True, padding_value=0
            ).to(self.device)
            # print("DEBUG batch_cats:", batch_cats)
            # print("DEBUG batch_scats:", batch_scats)
            # print("DEBUG padded.shape, cats len, scats len:", padded.shape, len(batch_cats), len(batch_scats))

            cats = torch.tensor(batch_cats, dtype=torch.long, device=self.device)
            scats = torch.tensor(batch_scats, dtype=torch.long, device=self.device)

            # ② MSA 뉴스 인코더 한 번에 호출
            nv = self.news_encoder(padded, cats, scats)   # (B, emb_dim)
            news_embeddings[batch_idx] = nv
            # print(f"DEBUG: news_embeddings.shape = {news_embeddings.shape}, nv.shape = {nv.shape}")
            # print(f"DEBUG: batch_idx min/max = {min(batch_idx)}/{max(batch_idx)}")

            # ③ 다음 배치를 위해 초기화
            batch_titles.clear(); batch_cats.clear()
            batch_scats.clear(); batch_idx.clear()

        for i, nid in enumerate(news_ids):
            if nid in self.news_id_to_info:
                info = self.news_id_to_info[nid]
                batch_titles.append(info['title_idx'])
                batch_cats.append(info['category_idx'])
                batch_scats.append(info['subcategory_idx'])
                batch_idx.append(i)
            else:                                       # OOV 뉴스
                news_embeddings[i] = torch.randn(self.emb_dim, device=self.device)
            # 배치가 max_batch에 도달하면 처리
            if len(batch_titles) == max_batch:

                _flush()
                # print("nv shape:", news_embeddings[0].shape)
                # print("nv:", news_embeddings[0])
                # exit()

        # 마지막에 남은 샘플 처리
        _flush()

        return news_embeddings
    
    # ------------------------------------------------------------------
    # LightGCN 초기화 (K, α 세팅) – 논문 식 (3)(4) 반영
    # ------------------------------------------------------------------
    def _init_lightgcn(self, K: int = 3, alpha: Optional[List[float]] = None):
        """K: propagation 층 수, alpha: layer combination weight list."""
        self.K = K
        # α_k = 1/(K+1) 기본값 – Layer Combination 식 (4) fileciteturn1file2
        if alpha is None:
            self.alpha = torch.tensor([1.0 / (K + 1)] * (K + 1), device=self.device)
        else:
            assert len(alpha) == K + 1
            self.alpha = torch.tensor(alpha, device=self.device)
    
    def lgcn_message_func(self, edges):
        # deg 정규화 1/√(deg_u * deg_v)
        coefficient = torch.rsqrt(edges.src['deg'] * edges.dst['deg']).unsqueeze(1)
        if not self.use_cat_emb:
            cat_embedding = self.rel_embedding[edges.data['cat_idx'].type(torch.LongTensor).to(self.device)]
            msg = edges.src['h'] * cat_embedding * coefficient  # 관계별 가중치(선택)
        else:
            msg = edges.src['h'] * coefficient
        return {'msg': msg}
    
    def lgcn_reduce_func(self, nodes):
        return {'h': nodes.mailbox['msg'].sum(1)}
    
    # def message_func(self, edges):
    #     # 엣지 메시지는 뉴스 임베딩과 엣지 임베딩의 element-wise product
    #     return {'msg': edges.src['node_emb'] * self.rel_embedding[edges.data['cat_idx'].type(torch.LongTensor)]}

    # def reduce_func(self, nodes):
    #     # 메시지를 상수 c로 나눈 뒤, 기존 임베딩과 더해줌 (residual)
    #     # aggregated = torch.sum(nodes.mailbox['msg'], dim=1) / self.c
    #     aggregated = nodes.mailbox['msg'].mean(1)
    #     return {'node_emb2': aggregated}   # new_feat -> feat: GCN 결과가 바로 update되도록 바꿈
    #                                                            # 위 주석처럼 할 경우 batch로 나눠서 GCN을 실행하기 때문에 오류 발생 가능성
    #                                                            # 다시 new_feat으로 생성하는 방식 사용        

    # ------------------------------------------------------------------
    # LightGCN K‑layer propagation + Layer Combination
    # ------------------------------------------------------------------
    # def propagate_lightgcn(self, g, seed_nodes, edges):
    #     """g: DGLGraph, seed_nodes: 업데이트 후 읽어올 노드 index 리스트"""
    #     # (1) 차수 정보 준비 – 0인 노드는 1로 클램프
    #     g.ndata['deg'] = g.in_degrees().float().clamp(min=1).to(self.device)
    #     # (2) 0‑th layer 임베딩을 h0로 저장
    #     g.ndata['h'] = g.ndata['node_emb']
    #     layer_embs = [g.ndata['h']]

    #     # (3) K 번 1‑hop propagation
    #     for _ in range(self.K):
    #         g.send_and_recv(edges = edges)
    #         layer_embs.append(g.ndata['h'])

    #     # (4) Weighted sum → 최종 임베딩 (식 (4))
    #     h_final = torch.stack(layer_embs, dim=0)  # (K+1, N, d)
    #     h_final = (self.alpha.view(-1, 1, 1) * h_final).sum(0)
    #     g.ndata['node_emb'] = h_final  # 다음 모듈(RNN 등)에서 사용
    #     return h_final[seed_nodes]

    ### 메모리 최적화
    def propagate_lightgcn(self, g, seed_nodes, edges):
        g.ndata['deg'] = g.in_degrees().float().clamp(min=1).to(self.device)
        g.ndata['h'] = g.ndata['node_emb']

        # ★ 누적합으로 변경 → (K+1)배 저장 안 함
        h_accum = self.alpha[0] * g.ndata['h']          # α₀·h⁽⁰⁾
        for k in range(self.K):
            g.send_and_recv(edges=edges)
            # h_accum = h_accum + self.alpha[k+1] * g.ndata['h']  # += αₖ₊₁·h⁽ᵏ⁺¹⁾
            h_accum = h_accum.add(self.alpha[k+1] * g.ndata['h'])
            
        g.ndata['node_emb'] = h_accum
        # 메모리 해제
        del g.ndata['deg'], g.ndata['h'];  torch.cuda.empty_cache()
        return h_accum[seed_nodes]


    def seq_GCRNN_batch(self, g, sub_g, latest_train_time, seed_list, history_length):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        gcn_1hopneighbor_per_time = []
        gcn_seed_2hopedge_per_time = []
        gcn_seed_2nd_layer_edge_per_time = []   # 1/2-hop edges를 (src, dst) 형태로 표현하여 담은 리스트
        # for 3-hop
        # gcn_2hopneighbor_per_time = []
        # gcn_seed_3hopedge_per_time = []
        # gcn_seed_3rd_layer_edge_per_time = []   # 1/2/3-hop edges를 (src, dst) 형태로 표현하여 담은 리스트
        
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num + self.news_num)
        for i in range(latest_train_time, -1, -1): # latest -> 0 미래부터 본다.
            # print(len(future_needed_nodes))
            
            # seed_list는 유저와 뉴스의 인덱스가 순서대로 저장돼 있도록 먼저 처리를 해주고, 이 작업을 수행할 수 있도록 해야 함
            # 이때 뉴스 인덱스는 유저 인덱스의 최대치만큼 다 더한 상태로 저장돼야 함 그래야 check_lifetime의 인덱스 형태로 들어갈 수 있으니까!
            check_lifetime[list(seed_list[i])] = history_length # seed_list: time별로 seed user가 들어있음

            # seed list에 들어있는 user들을 future needed nodes에 추가함(과거로 미래를 initialize하기 때문)
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            # 따라서 해당 seed들은 과거에도 계속 seed에 들어가게 되지만, 과거에 edge가 존재하는지 여부는 모름
            # 또한 history length가 full(100)인 현 상황에서는 초반(앞 시간대)구간의 경우 seed수가 계속 같을 수 있음
            # 또한, 데이터 특성상 과거로 갈 수록 edge가 적음

            # # future_needed_nodes가 (g에 대한) global_id인데 sub_g에서 future_needed_nodes를 사용해 발생하는 문제 해결
            # sub_g_user_ids = sub_g[i].nodes['user'].data['user_ids'].tolist()
            # global_to_local_user = {gid: local for local, gid in enumerate(sub_g_user_ids)}
            # valid_future_nodes = [global_to_local_user[gid] for gid in future_needed_nodes if gid in global_to_local_user]

            # -------------- 1‑, 2‑hop(원래 코드) ----------------
            # 1hop edges of seed at i
            hop1_u, hop1_v = sub_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')
            # hop1_u, hop1_v = sub_g[i].in_edges(v = valid_future_nodes, form = 'uv', etype='clicked_reverse')   # u (news) -> v (user) 이다
            ### sub_g를 쓰는 것은 snapshot 시간을 조절하기 위함임 - 해당 시간에 존재하는 edges만 뽑아내려고!!! 
            hop1_neighbors_at_i, _, seed_edges_at_i = sub_g[i].in_edges(v = list(future_needed_nodes), form = 'all')
            # node는 그대로 가져와지지만, splitted에서 추출한 edge id는 g의 edge id와 다를 수 있다.
            # 따라서 'edge id'가 아니라 'node id 쌍'로 edge를 기록해야 한다.
            ### form = 'all': 순서대로 src, output, edge (array)

            # sample한 user들(entity)에 대해 in-edge들 찾는다.
            # hop 1 neighbors는 2layer를 위해 찾아둔것

            # 2번째 layer를 위한 edge
            # check_lifetime[hop1_neighbors_at_i] = history_length
            # hop2_u, hop2_v = sub_g[i].in_edges(v = hop1_neighbors_at_i, form = 'uv') # hop2_edges_at_i = sub_g[i].in_edges(v = hop1_neighbors_at_i, form = 'eid')
            # ---------------------------------------------------

            # -------------- *** 3‑hop 추가 *** ------------------
            hop2_neighbors_at_i, _, seed_edges_at_i_2hop = sub_g[i].in_edges(v = hop1_neighbors_at_i, form = 'all')
            # 3번째 layer를 위한 edge            
            # check_lifetime[hop2_neighbors_at_i] = history_length
            # hop3_u, hop3_v = sub_g[i].in_edges(v=hop2_neighbors_at_i, form='uv')
            # ---------------------------------------------------

            gcn_seed_per_time.append(list(future_needed_nodes)) # Seed            
            ## for 1-hop
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))
            
            ##### 1-hop edge만 사용
            # ### for 2-hop
            # gcn_1hopneighbor_per_time.append(hop1_neighbors_at_i) # Seed's Edge's source node
            # gcn_seed_2hopedge_per_time.append((hop2_u, hop2_v)) # Source node's edge

            # ### for 3-hop
            # gcn_2hopneighbor_per_time.append(hop2_neighbors_at_i)
            # gcn_seed_3hopedge_per_time.append((hop3_u, hop3_v))
            # # ---------------------------------------------------
            
            # # 에지가 전혀 없는 timestamp에서는 스킵
            # if len(hop1_u) == 0 and len(hop2_u) == 0 and len(hop3_u) == 0:
            #     empty = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            #     gcn_seed_2nd_layer_edge_per_time.append(empty)
            #     gcn_seed_3rd_layer_edge_per_time.append(empty)
            #     continue
            
            # ### gcn_seed_2nd_layer_edge_per_time을 만들어야 함 (1hop edge랑 2hop edge를 중복 없이 합쳐야 함)
            # # --- 1-hop, 2-hop 에지를 합쳐서 중복 제거 ---
            # all_hop1_2_u = torch.cat([hop1_u, hop2_u])
            # all_hop1_2_v = torch.cat([hop1_v, hop2_v])

            # # (u, v) 쌍을 [N, 2] 형태로 만든 뒤, 중복 쌍 제거
            # unique_hop1_2_uv = torch.unique(torch.stack([all_hop1_2_u, all_hop1_2_v], dim=1), dim=0)
            # final_hop1_2_u, final_hop1_2_v = unique_hop1_2_uv[:, 0], unique_hop1_2_uv[:, 1]
            
            # gcn_seed_2nd_layer_edge_per_time.append((final_hop1_2_u, final_hop1_2_v))
            
            # ### for 3-hop
            # all_hop_u = torch.cat([hop1_u, hop2_u, hop3_u])
            # all_hop_v = torch.cat([hop1_v, hop2_v, hop3_v])

            # # (u, v) 쌍을 [N, 2] 형태로 만든 뒤, 중복 쌍 제거
            # unique_all_uv = torch.unique(torch.stack([all_hop_u, all_hop_v], dim=1), dim=0)
            # final_all_u, final_all_v = unique_all_uv[:, 0], unique_all_uv[:, 1]
            
            # gcn_seed_3rd_layer_edge_per_time.append((final_all_u, final_all_v))
            
            # ##### check
            # print("hop1_u shape:", hop1_u.shape, "hop1_v shape:", hop1_v.shape)
            # print("hop2_u shape:", hop2_u.shape, "hop2_v shape:", hop2_v.shape)
            # print("all_hop_u shape:", all_hop_u.shape, "all_hop_v shape:", all_hop_v.shape)

            # stacked = torch.stack([all_hop_u, all_hop_v], dim=1)
            # print("stacked shape:", stacked.shape)
            # unique_uv = torch.unique(stacked, dim=0)
            # print("unique_uv shape:", unique_uv.shape)
            
            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - set(np.where(check_lifetime==0)[0]) # seed next
            except:
                pass
        
        self.rel_embedding = self.cat_embedding_layer(torch.tensor(range(self.cat_num)).to(self.device))
        # 초기화해줘야 밑에서 슬라이싱 가능
        # g.ndata['node_emb'] = torch.zeros(g.number_of_nodes(), self.emb_dim, device=self.device)
        user_embeddings = self.user_embedding_layer(torch.tensor(range(self.user_num)).to(self.device))
        news_embeddings = self.News_Encoder(self.all_news_ids)

        # g.ndata['node_emb'][:self.user_num] = self.user_embedding_layer(torch.tensor(range(self.user_num)).to(self.device))
        g.ndata['node_emb'] = torch.cat([user_embeddings, news_embeddings], dim=0)

        # history_index = [nid[1:] for nid in self.all_news_ids]
        g.ndata['node_emb'][self.user_num:] = self.News_Encoder(self.all_news_ids)
        # print(g.device)
        # print()
        # g.ndata['cx'] = self.c0_embedding_layer_u(torch.tensor(range(g.number_of_nodes())).to(self.device))
        g.ndata['cx'] = self.c0_embedding_layer_u(torch.arange(self.user_num+self.news_num, device=self.device))

        entity_embs = []
        entity_index = []
        
        # LightGCN 옵션
        self.use_cat_emb = self.config.unique_category  # 카테고리 임베딩을 쓰려면 True
        self._init_lightgcn(K=self.config.hop)  # K 층, α 기본값
        
        # register함수는 DGL 0.9이상에서는 없어졌다.
        g.register_message_func(self.lgcn_message_func)
        g.register_reduce_func(self.lgcn_reduce_func)
        for i in range(latest_train_time+1): # 0 -> latest
            # g_now = splitted_g[i]
            inverse = latest_train_time - i   # 1680-i
            # gcn_seed_per_time -> 미래부터 들어있다
            if len(gcn_seed_per_time[inverse]) > 0:   # inverse를 했을 때, gcn_seed_per_time이 애초에 시간 역순이라, 다시 시간 순서대로 보겠다는 의미
                changed = sorted(gcn_seed_per_time[inverse])   # 해당 time의 seed user 리스트를 user_id 순으로 정렬

                user_seed_ = changed   # g.nodes['user'].data['user_ids']
                # print(user_seed_)
                user_prev_hn = g.ndata['node_emb'][user_seed_]#.to(self.device1)
                user_prev_cn = g.ndata['cx'][user_seed_]#.to(self.device1)

                # ### 여러 번 더해지는 것을 방지
                # original_node_embs = g.ndata['node_emb']
                # aggregator  = torch.zeros_like(original_node_embs)
                # # 3-hop information aggregated
                # edge_num = len(gcn_seed_3rd_layer_edge_per_time[inverse][0])
                # g.send_and_recv(edges = gcn_seed_3rd_layer_edge_per_time[inverse])
                # if edge_num > 0:
                #     try:
                #         aggregator += g.ndata.pop('node_emb2')   # hop별 결과 누적
                #     except:
                #         pass
                # # 2-hop information aggregated
                # edge_num = len(gcn_seed_2nd_layer_edge_per_time[inverse][0])
                # g.send_and_recv(edges = gcn_seed_2nd_layer_edge_per_time[inverse])
                # if edge_num > 0:
                #     try:
                #         aggregator += g.ndata.pop('node_emb2')   # hop별 결과 누적
                #     except:
                #         pass
                # # 1-hop information aggregated    
                # edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                # g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse])
                # if edge_num > 0:
                #     try:
                #         aggregator += g.ndata.pop('node_emb2')   # hop별 결과 누적
                #     except:
                #         pass
                # # 최종 노드 업데이트 
                # g.ndata['node_emb'] = original_node_embs + aggregator / 3

                user_input = self.propagate_lightgcn(g, user_seed_, gcn_seed_1hopedge_per_time[inverse])

                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))   # RNN이 실행되는 time gap이 유저임베딩마다 다름

                # g.ndata['node_emb'][user_seed_] = user_hn
                # g.ndata['cx'][user_seed_] = user_cn
                # 수정된 코드:
                old_emb = g.ndata['node_emb']                 # 현재 노드 임베딩 (텐서)
                new_emb = old_emb.clone()                     # 전체 복사본 생성 (그래디언트 연결 보존)
                new_emb[user_seed_] = user_hn           # 필요한 인덱스만 갱신
                g.ndata['node_emb'] = new_emb                 # 수정된 전체 텐서 한 번에 할당

                # LSTM cell state도 동일하게 처리
                old_cx = g.ndata['cx']
                new_cx = old_cx.clone()
                new_cx[user_seed_] = user_cn
                g.ndata['cx'] = new_cx
                
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]   # user_id 순으로 정렬되진 않음
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i   # user_id 순으로 index 크기가 정렬되게 함
                # 같은 유저라도 timestamp에 따라 고유한 index를 갖도록 해줌
                # 즉, 각 user_changed_in_global은 유저의 고유한 embedding 순서를 나타냄
                entity_embs.append(seed_emb)   # 아직 user_id 순으로 정렬되지 않은 embeddings
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))

        # for i in range(len(entity_embs)):
        #     print(entity_embs[i].shape)
        # print("finished")
        # for i in range(len(entity_index)):
        #     print(entity_index[i].shape)
        # exit()
        entity_embs = torch.cat(entity_embs).to(self.device)   # (각 snapshot마다 존재하는 유저 수의 총 합, emb_dim)
                                                               # 우리 방법의 경우, 각 유저가 모두 마지막 seed_list(seed_list[1679])에만 존재
                                                               # -> 실제 크기는 (batch_size, emb_dim)
        # print(entity_embs.shape)
        # print(len(entity_index))

        # entity_index torch.cat 전: len=100(각 요소: [0], [0], ..., [500]), 후: shape=(500,)
        entity_index = torch.cat(entity_index)   # entity_index의 값들을 순서대로 정렬할 때, 이들의 indicies를 순서대로 반환
                                                 # shape: (각 snapshot마다 존재하는 user_num * snapshots_num, )
                                                 # 따라서 entity_index는 각 user의 idx를 갖고 있는 것과 같음
        # a4 = time.time()
        ent_embs = entity_embs[entity_index.argsort()]   # 드디어 user_id 순으로 정렬됨
        # entity_index.argsort(): user indicies가 시간 순대로 나열됨
        # ent_embs: GCRNN으로 구한 user embedding이 시간 순대로 나열됨 - 각 유저의 마지막 값이 rnn의 최종 결과값 (마지막 hidden state)
        # ent_embs shape: (500, 128)
        return ent_embs
        
    
    def forward(self, user_batch, news_batch, category_batch, time_batch, g, sub_g, ns_idx, history_length=100): # user_batch, news_batch,
        """
        g: DGL 이종 그래프
           - etype = ('user', 'clicked', 'news'),
                     ('user', 'clicked_reverse', 'news')
           - 
        #    - g.nodes['user'].data['feat'], g.nodes['news'].data['feat'] 에는
        #      노드 임베딩이 저장되어 있음.
           - g.edges['clicked'].data['feat'] 에는 user -> news 방향의 엣지 임베딩이 저장되어 있음.
           - g.edges['clicked_reverse'].data['feat'] 에는 news-> user 방향의 엣지 임베딩이 저장되어 있음.
           - g.nodes['user'].data['user_ids']에는 user_ids가 tensor 형태로 저장되어 있음. (user_num, 1)
           - g.nodes['news'].data['news_ids']에는 news_ids가 tensor 형태로 저장되어 있음. (news_num, 1)
        #    - g.edges['clicked'].data['category_id']에는 category_ids가 tensor 형태로 저장되어 있음. (cat_num, 1)
        #    - g.edges['clicked_reverse'].data['category_id']에는 category_ids가 tensor 형태로 저장되어 있음. (cat_num, 1)
        
        snapshots_user_ids: snapshot마다 저장된 유저 id

        return:
            updated_user_feats: (num_users, emb_dim)
            updated_news_feats: (num_news, emb_dim)
        """
        # print(f"[Before] GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device)/1024**2:.2f} MiB")
        
        seed_list = []
        seed_entid = []
        train_t = []
        latest_train_time = self.snapshots_num - 1
        for i in range(latest_train_time+1):
            seed_list.append(set())
            
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                train_t.append(time)
                
        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), 
                                            sorted = True, return_inverse = True)
        # 각 rnn의 마지막 hidden state, 즉 rnn 결과로 얻은 각 유저의 embedding indicies를 저장한 tensor
        # index_for_ent_emb: unique 값들의 indicies를 모아둔 tensor
        # 즉, seed_entid의 train_click_num만큼 존재하는 user indicies
        
        # for index in index_for_ent_emb:
        #     print(index)
        # print(index_for_ent_emb.shape)
        # print('\n')
        # print(len(train_t))
        # for tt in train_t:
        #     print(tt)
        # print('\n')
        # print(len(seed_entid))
        # for sid in seed_entid:    
        #     print(sid)
        # exit()
        
        user_embs = ent_embs[index_for_ent_emb]   # (train_click_num, 128)
                                                  # 이게 아마 정답) user_embs는 seed_entid 순으로 정렬된 user embeddings
                                                  # 즉, user_embs가 이미 내가 원하는 candidate_user_embs 형태!!!
        # print(entity_index.shape)
        # print(ent_embs.shape)
        # print(index_for_ent_emb.shape)
        # print(user_embs.shape)
        # exit()

        # 각 유저의 GCRNN 후 embeddings
        # *** userid 순으로 정렬됨 ***
        # train_click_num만큼 복사해줘야 함
        
        # u_time_embs = torch.cat([user_emb_0, user_embs]) # (N, emb_dim)   왜 합쳤니???????????? 난 어떻게 해야 하지...

        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch] # (N, emb_dim)
        # 원본: target_c_embs = self.ent_embedding_layer(torch.cat(comp_target_0).to(self.device0) + self.user_id_max + 1) # (N, emb_dim)
        # comp_target_0: 각 회사별 첫 번째 시간대의 news embedding의 indicies를 먼저 각각 하나의 list에 추가하고, 이후 각 회사별 첫 번째 외의 시간대 tensor가 순서대로 쌓여 있음
        
        """
        user_embs: (click 수, emb_dim)
        candidate_n_embs: (click 수, 9, emb_dim)
        내적 후 score: (click 수, 9)
        label: (click 수,) (예: 모든 값이 0, 즉 첫 번째 후보가 정답)
        """
        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch]   # (target_news_num, emb_dim); target_news_num은 batch마다 다름
        # target_score = torch.matmul(user_embs, target_n_embs.transpose(1,0))   # (batch_size=500, target_news_num)
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # g.nodes['news'].data['node_emb']는 news_int 순서대로 embedding 저장한 텐서; shape: (news_num, emb_dim)
        # candidate_n_embs: (train_click_num, (1 + 4), emb_dim); 1: target, 4: ns sample 수
        # ns_idx: (train_click_num, 5)
        candidate_user_embs = user_embs#[user_score_idx]   # user_score_idx: (train_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (train_click_num, 1, 128)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # print("candidate_score shape:", candidate_score.shape)
        # candidate_n_embs: (train_click_num, emb_dim)*(train_click_num, 5, emb_dim)
        # candidate_score: (train_click_num, 5)
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   

        return loss
    
    
    def inference(self, user_batch, news_batch, time_batch, g, sub_g, ns_idx, history_length=100):
        ### test_time = 1679가 input일 때 코드
        # seed_list = []
        # seed_entid = []
        # test_t = []
        # for i in range(test_time+1):
        #     seed_list.append(set())
        # a= 0
        # b=0
        # for user, news_tensor in zip(user_batch, news_batch):   # batch_size(500)만큼 동작
        #     if len(news_tensor) == 0:
        #         pass
        #     else:
        #         test_t.append(test_time)
        #         seed_entid.append(user)
        #         seed_list[test_time].add(user)
        
        
        seed_list = []
        seed_entid = []
        test_t = []
        
        for time_list in time_batch:
            for time in time_list:
                test_t.append(time)
        
        latest_train_time = self.snapshots_num - 1   # train까지 포함한 snapshot 수는 2016개
        seed_entid = []
        test_t = []
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                test_t.append(time)

        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)   # (batch_size, emb_dim)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(test_t), 
                                            sorted = True, return_inverse = True)
        # (batch_size, )
        u_time_embs = ent_embs[index_for_ent_emb]

        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # g.nodes['news'].data['node_emb']는 news_int 순서대로 embedding 저장한 텐서; shape: (news_num, emb_dim)
        # candidate_n_embs: (test_click_num, (1 + 20), emb_dim); 1: target, 20: ns sample 수
        # ns_idx: (test_click_num, 21)
        candidate_user_embs = u_time_embs#[user_score_idx]   # user_score_idx: (test_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (test_click_num, 1, 128)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # candidate_n_embs: (test_click_num, emb_dim)*(test_click_num, 21, emb_dim)
        # candidate_score: (test_click_num, 21)        
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   
        
        return candidate_score, loss
