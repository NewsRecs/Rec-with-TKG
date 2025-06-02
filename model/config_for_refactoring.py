import os, time, argparse, random, datetime
import numpy as np
import torch, torch.distributed as dist

class Config:
    """실행 시점에 CLI·환경설정·시드·디렉터리 등을 한꺼번에 준비한다."""
    # ──────────────────────────────────────────────────────────────
    # 1) CLI 인자 정의
    # ──────────────────────────────────────────────────────────────
    def _parse_args(self):
        p = argparse.ArgumentParser(description='GCRNN 설정')

        # ── 실행 모드 & 일반 ─────────────────────────────
        p.add_argument('--mode',            type=str, default='train',  choices=['train','dev','test'])
        p.add_argument('--device_id',       type=int, default=0)                # GPU 번호
        p.add_argument('--seed',            type=int, default=28)               # 재현성

        # ── 데이터셋 / 경로 ─────────────────────────────
        p.add_argument('--dataset',         type=str, default='Adressa_7w',
                                            choices=['Adressa_7w', 'Adressa_3w', 'Adressa_1w'])
        p.add_argument('--base_root',       type=str, default='/home/user/pyo') # 베이스 경로
        p.add_argument('--pretrained_embedding_npy',
                                            type=str,
                                            default='/home/user/pyo/psj/Adressa_4w/history/pretrained_word_embedding.npy')

        # ── 학습 하이퍼파라미터 ─────────────────────────
        p.add_argument('--batch_size',      type=int, default=150)              # original_batch_size
        p.add_argument('--lr',              type=float, default=1e-4)
        p.add_argument('--num_epochs',      type=int, default=10)
        p.add_argument('--weight_decay',    type=float, default=1e-2)
        p.add_argument('--emb_dim',         type=int, default=300)
        p.add_argument('--history_length',  type=int, default=100)
        
        
        # ── NewsEncoder 파라미터 ───────────────────────
        p.add_argument('--window_size',     type=int, default=3)
        p.add_argument('--word_embedding_dim',     type=int, default=100,
                                            help='사전 학습된 단어 embedding 차원')
        p.add_argument('--num_words',       type=int, default=330899+1,
                                            help='사전 학습된 단어 수 (index = 0: 존재하지 않는 단어들)')
        p.add_argument('--num_filters',     type=int, default=100)
        p.add_argument('--num_categories_for_NewsEncoder',    type=int, default=17)
        p.add_argument('--num_subcategories_for_NewsEncoder', type=int, default=95)
        p.add_argument('--use_batch',                         type=bool, default=True)
        p.add_argument('--dropout_probability',               type=float, default=0.2)
        p.add_argument('--query_vector_dim',                  type=int, default=200)
        ### for MSA NewsEncoder
        p.add_argument('--head_num',                          type=int, default=20)
        p.add_argument('--head_dim',                          type=int, default=15)
        p.add_argument('--dataset_lang',                      type=str, default='norwegian')
        p.add_argument('--attention_hidden_dim',              type=int, default=100)
        
        # ── 스냅샷/그래프 ──────────────────────────────
        p.add_argument('--snapshot_week',   type=int, default=6)                # history + train 주 수
        p.add_argument('--snapshots_num',   type=int, default=None,             # 자동 계산용 placeholder
                                            help='명시하지 않으면 snapshot_week*7*24*2 로 자동 설정')
        p.add_argument('--user_num',        type=int, default=84989)
        p.add_argument('--news_num',        type=int, default=15383)
        p.add_argument('--GCN_cat_num',     type=int, default=35,
                                            help='nyheter의 서브카테고리들을 포함한 카테고리 수')
        
        # ── 모델 구조 ─────────────────────────────────
        p.add_argument('--adjust_score',    type=bool, default=False)
        p.add_argument('--split_nyheter',   type='bool', default=False,                # flag: nyheter split 사용
                                            help='all_news_nyheter_splitted.tsv 사용 여부')
        p.add_argument('--NewsEncoder_method',    type=str, default='multihead_self_attention',
                                                  choices=['cnn_attention', 'multihead_self_attention'])
        p.add_argument('--NewsEncoder_content',   type=str, default='title_category',
                                                  choices=['title', 'title_category'])
        p.add_argument('--TG_window_size',        type=float, default=0.5,
                                                  help='단위는 hour')

        # ── wandb ─────────────────────────────────────
        p.add_argument('--wandb_project',         type=str, default='LifeTGNN')
        p.add_argument('--wandb_run_name',        type=str, default=f'{self.dataset}w_adjust_{self.adjust_score}_split_{self.split_nyheter}')


        # 필요한 인자 추가 시 여기에서 정의
        return p.parse_args()

    # ──────────────────────────────────────────────────────────────
    # 2) CUDA, 시드, DDP 초기화
    # ──────────────────────────────────────────────────────────────
    def _setup_cuda(self):
        assert torch.cuda.is_available(), 'CUDA 사용 불가'
        torch.cuda.set_device(self.device_id)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    # ──────────────────────────────────────────────────────────────
    # 3) 유틸: 경로·스냅숏 계산·디렉터리 생성
    # ──────────────────────────────────────────────────────────────
    def _postprocess(self):
        # 스냅숏 개수 자동 계산 (30분 간격)
        if self.snapshots_num is None:
            self.snapshots_num = int(self.snapshot_week * 7 * 24 * (1 / self.TG_window_size))

        # 경로들
        self.train_root = os.path.join(self.base_root, self.dataset, 'train')
        self.test_root  = os.path.join(self.base_root, self.dataset, 'test')

        # 출력/모델 폴더 생성
        def mkdirs(p): os.makedirs(p, exist_ok=True)
        model_name = 'LifeTGNN'  # 필요시 인자로 확장
        for d in [
            f'configs/{self.dataset}/{model_name}',
            f'bests/{self.dataset}/{model_name}',
            f'results/{self.dataset}/{model_name}',
        ]:
            mkdirs(d)

    # ──────────────────────────────────────────────────────────────
    # 생성자: 위 메서드 순차 실행
    # ──────────────────────────────────────────────────────────────
    def __init__(self):
        # 1) CLI → 속성
        args = self._parse_args()
        self.__dict__.update(vars(args))   # argparse 결과를 곧바로 멤버로

        # 2) CUDA/시드
        self._setup_cuda()

        # 3) 후처리
        self._postprocess()

        # 4) 가장 쉬운 확인 로그
        print('\n' + '*'*30 + ' 설정 값 ' + '*'*30)
        for k, v in sorted(self.__dict__.items()):
            print(f'{k:20}: {v}')
        print('*'*68 + '\n')
