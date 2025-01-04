from torch import nn
import dgl
import torch


# ------------------------------------------------------
# GCN 클래스 정의
# ------------------------------------------------------
class GCN(nn.Module):
    """
    사용자/뉴스 노드 모두에 대해 다음을 수행:
    1) 1-hop 이웃 노드들의 임베딩과 엣지 임베딩을 element-wise product
    2) 결과를 상수 c로 나눈 뒤, 자기 임베딩과 더해주기 (residual-like)
    3) batch 단위(500개)로 나눠서 처리 
    """
    def __init__(self, batch_size=500):
        super(GCN, self).__init__()
        self.batch_size = batch_size
            
    def batch_GCN(self, g, num_edges, message_func, reduce_func, etype):
        for start in range(0, num_edges, self.batch_size):
            end = start + self.batch_size
            edge_ids = list(range(start, min(end, num_edges)))  # 마지막 배치 처리 시 범위 초과 방지
            
            # message passing
            g.send_and_recv(
                edges=edge_ids,
                message_func=message_func,
                reduce_func=reduce_func,
                etype=etype
            )
            if etype == 'clicked':
                updated_feat = g.nodes['news'].data['new_feat']
            else:
                updated_feat = g.nodes['user'].data['new_feat']
            
            return updated_feat    
    
    def forward(self, g: dgl.DGLHeteroGraph):
        """
        g: DGL 이종 그래프
           - etype = ('user', 'clicked', 'news')
           - g.nodes['user'].data['feat'], g.nodes['news'].data['feat'] 에는
             이미 노드 임베딩이 저장되어 있음.
           - g.edges['clicked'].data['feat'] 에는 엣지 임베딩이 저장되어 있음.

        return:
            updated_user_feats: (num_users, emb_dim)
            updated_news_feats: (num_news, emb_dim)
        """
        device = g.device

        # print(f"[Before] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
        
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
            return {'new_feat': aggregated + nodes.data['feat']}
        
        
        # 뉴스 노드에서 유저 노드로의 메시지 전달 및 집계
        etype = 'clicked_reverse'
        updated_user_feats = self.batch_GCN(g, num_edges, message_func, reduce_func, etype)

        # print(f"[After clicked_reverse] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
                
        # ------------------------------------------------------
        # (B) 뉴스 노드에 대한 GCN
        # ------------------------------------------------------     
        
        # 유저 노드에서 뉴스 노드로의 메시지 전달 및 집계
        # g.update_all(message_func, reduce_func, etype='clicked')
        etype = 'clicked'
        updated_news_feats = self.batch_GCN(g, num_edges, message_func, reduce_func, etype)
        
        # print(f"[After clicked] GPU 메모리 사용량: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
        
        # 업데이트된 뉴스 임베딩 추출
        # updated_news_feats = g.nodes['news'].data['new_feat']

        # updated_user_feats = updated_user_feats.cpu()
        # updated_news_feats = updated_news_feats.cpu()
        
        return updated_user_feats, updated_news_feats

