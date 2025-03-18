import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, score, label):
        """
        
        Args:
            score: (batch_size, candidate_num)   # 후보별 점수를 담고 있음
            label: (batch_size, candidate_num)   # 각 행 (배치)마다 어떤 후보가 정답인지 index를 가짐. 실제 shape: (batch_size,)

        Returns:

        """
        # (batch_size)
        result = F.log_softmax(score, dim=1)
        loss = F.nll_loss(result, label)
        return loss