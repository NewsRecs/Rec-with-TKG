import torch
import os

import torch
import os  

class EarlyStopping:
    """
    Validation 시 AUC, MRR, nDCG@5, nDCG@10의 평균값(=val_score) 기준 Early Stopping 클래스.

    - patience: 개선 없이 몇 epoch를 기다릴지
    - min_delta: 이전 best_score에서 얼마나 증가해야 개선이라고 보나
    - ckpt_dir: 체크포인트를 저장할 디렉토리
    - verbose: 메시지 출력 여부
    - save_all: 모든 epoch마다 모델 저장할지 여부
    """
    def __init__(self, emb_dim, patience=3, min_delta=1e-4, ckpt_dir='checkpoints', verbose=True, save_all=False):
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose
        self.save_all = save_all
        self.emb_dim = emb_dim

        self.counter = 0
        self.best_score = None  # 4개 지표의 평균값 중 최고
        self.early_stop = False
        self.best_epoch = -1
        self.best_ckpt_path = None
        self.lr = None  # 나중에 저장할 때 필요

        if self.ckpt_dir is not None:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def __call__(self, val_score, model, epoch, lr):
        """
        val_score: (AUC + MRR + nDCG@5 + nDCG@10) / 4
        model   : 현재 학습 중인 모델
        epoch   : 현재 epoch
        lr      : 현재 learning rate
        """
        self.lr = lr

        # (A) 모든 epoch 마다 모델 저장 (save_all=True일 경우)
        if self.save_all:
            ckpt_name = f"epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_avgscore_{val_score:.4f}.pth"
            save_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)
            if self.verbose:
                print(f"[EarlyStopping] Saved checkpoint (all epochs): {save_path}")

        # (B) EarlyStopping 로직 (val_score가 클수록 좋다고 가정)
        if self.best_score is None:
            # 초기 설정
            self.best_score = val_score
            self.best_epoch = epoch
            if self.verbose:
                print(f"[EarlyStopping] Initialize best_score = {val_score:.6f}")
            self.save_best_model(val_score, model, epoch, lr)

        elif (val_score - self.best_score) >= self.min_delta:
            # score 개선됨
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] score improved to {val_score:.6f} at epoch={epoch}. Reset counter.")
            self.save_best_model(val_score, model, epoch, lr)

        else:
            # score 개선 안 됨
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement. counter={self.counter}/{self.patience} (score={val_score:.6f}).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Stop training.")

    def save_best_model(self, val_score, model, epoch, lr):
        """현재까지 최고의 모델(가장 높은 val_score) 저장."""
        ckpt_name = f"best_epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_avgscore_{val_score:.4f}.pth"
        save_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save(model.state_dict(), save_path)
        self.best_ckpt_path = save_path
        if self.verbose:
            print(f"[EarlyStopping] Best model saved: {save_path}")



############# validation ndcg@5 기준
# class EarlyStopping:
#     """
#     Validation nDCG 기준 Early Stopping 클래스.

#     - patience: 개선 없이 몇 epoch를 기다릴지
#     - min_delta: 이전 best_ndcg에서 얼마나 증가해야 개선으로 보나
#     - ckpt_dir: 체크포인트를 저장할 디렉토리
#     - verbose: 메시지 출력 여부
#     - save_all: 모든 epoch마다 모델 저장할지 여부

#     *기존 best_loss → best_ndcg로 교체하고,
#      기존 val_loss → val_ndcg로 교체하였습니다.*
#     """
#     def __init__(self, emb_dim, patience=5, min_delta=1e-4, ckpt_dir='checkpoints', verbose=True, save_all=False):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.ckpt_dir = ckpt_dir
#         self.verbose = verbose
#         self.save_all = save_all
#         self.emb_dim = emb_dim

#         self.counter = 0
#         self.best_ndcg = None  # nDCG 최고값
#         self.early_stop = False
#         self.best_epoch = -1
#         self.best_ckpt_path = None

#         if self.ckpt_dir is not None:
#             os.makedirs(self.ckpt_dir, exist_ok=True)

#     def __call__(self, val_ndcg, model, epoch, lr):
#         """
#         val_ndcg: 현재 epoch에서 계산한 validation nDCG
#         model   : 현재 학습 중인 모델
#         epoch   : 현재 epoch
#         lr      : 현재 learning rate
#         """
#         # (A) 모든 epoch 마다 모델 저장 (save_all=True일 경우)
#         if self.save_all:
#             ckpt_name = f"epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_ndcg_{val_ndcg:.4f}.pth"
#             save_path = os.path.join(self.ckpt_dir, ckpt_name)
#             torch.save(model.state_dict(), save_path)
#             if self.verbose:
#                 print(f"[EarlyStopping] Saved checkpoint (all epochs): {save_path}")

#         # (B) EarlyStopping 로직 (nDCG가 클수록 좋다고 가정)
#         if self.best_ndcg is None:
#             # 초기 설정
#             self.best_ndcg = val_ndcg
#             self.best_epoch = epoch
#             if self.verbose:
#                 print(f"[EarlyStopping] Initialize best_ndcg = {val_ndcg:.6f}")
#             self.save_best_model(val_ndcg, model, epoch, lr)

#         elif (val_ndcg - self.best_ndcg) >= self.min_delta:
#             # nDCG 개선됨
#             self.best_ndcg = val_ndcg
#             self.best_epoch = epoch
#             self.counter = 0
#             if self.verbose:
#                 print(f"[EarlyStopping] nDCG improved to {val_ndcg:.6f} at epoch={epoch}. Reset counter.")
#             self.save_best_model(val_ndcg, model, epoch, lr)

#         else:
#             # nDCG 개선 안 됨
#             self.counter += 1
#             if self.verbose:
#                 print(f"[EarlyStopping] No improvement. counter={self.counter}/{self.patience} (nDCG={val_ndcg:.6f}).")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.verbose:
#                     print("[EarlyStopping] Stop training.")

#     def save_best_model(self, val_ndcg, model, epoch, lr):
#         """현재까지 최고의 모델(가장 높은 nDCG) 저장."""
#         ckpt_name = f"best_epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_ndcg_{val_ndcg:.4f}.pth"
#         save_path = os.path.join(self.ckpt_dir, ckpt_name)
#         torch.save(model.state_dict(), save_path)
#         self.best_ckpt_path = save_path
#         if self.verbose:
#             print(f"[EarlyStopping] Best model saved: {save_path}")


################ validation loss 버전
# import torch
# import os

# class EarlyStopping:
#     """
#     Validation Loss 기준 Early Stopping 클래스.
    
#     - patience: 개선 없이 몇 epoch를 기다릴지
#     - min_delta: 이전 best_loss에서 얼마나 감소해야 개선으로 보나
#     - ckpt_dir: 체크포인트를 저장할 디렉토리
#     - verbose: 메시지 출력 여부
#     - save_all: 모든 epoch마다 모델 저장할지 여부
#     """
#     def __init__(self, emb_dim, patience=3, min_delta=1e-4, ckpt_dir='checkpoints', verbose=True, save_all=False):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.ckpt_dir = ckpt_dir
#         self.verbose = verbose
#         self.save_all = save_all
#         self.emb_dim = emb_dim

#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#         self.best_epoch = -1
#         self.best_ckpt_path = None

#         if self.ckpt_dir is not None:
#             os.makedirs(self.ckpt_dir, exist_ok=True)

#     def __call__(self, val_loss, model, epoch, lr):
#         # (A) 모든 epoch마다 저장 (save_all=True일 경우)
#         if self.save_all:
#             self.lr = lr
#             ckpt_name = f"epoch_{epoch}_lr_{lr}_embdim_{self.emb_dim}_loss_{val_loss:.4f}.pth"
#             save_path = os.path.join(self.ckpt_dir, ckpt_name)
#             torch.save(model.state_dict(), save_path)
#             if self.verbose:
#                 print(f"[EarlyStopping] Saved checkpoint: {save_path}")

#         # (B) EarlyStopping 체크
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.best_epoch = epoch
#             if self.verbose:
#                 print(f"[EarlyStopping] Initialize best_loss = {val_loss:.6f}")
#             self.save_best_model(val_loss, model, epoch)

#         elif (self.best_loss - val_loss) >= self.min_delta:
#             # 개선됨
#             self.best_loss = val_loss
#             self.best_epoch = epoch
#             self.counter = 0
#             if self.verbose:
#                 print(f"[EarlyStopping] Loss improved to {val_loss:.6f} at epoch={epoch}. Reset counter.")
#             self.save_best_model(val_loss, model, epoch)

#         else:
#             # 개선 안 됨
#             self.counter += 1
#             if self.verbose:
#                 print(f"[EarlyStopping] No improvement. counter={self.counter}/{self.patience} (loss={val_loss:.6f}).")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.verbose:
#                     print("[EarlyStopping] Stop training.")

#     def save_best_model(self, val_loss, model, epoch):
#         """현재까지 최고의 모델 저장 (best_epoch_xx_loss_xxxx.pth)."""
#         ckpt_name = f"best_epoch_{epoch}_lr_{self.lr}_embdim_{self.emb_dim}_loss_{val_loss:.4f}.pth"
#         save_path = os.path.join(self.ckpt_dir, ckpt_name)
#         torch.save(model.state_dict(), save_path)
#         self.best_ckpt_path = save_path
#         if self.verbose:
#             print(f"[EarlyStopping] Best model saved: {save_path}")
