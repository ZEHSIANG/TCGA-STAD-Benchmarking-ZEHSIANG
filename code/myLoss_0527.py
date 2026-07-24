import torch
import torch.nn as nn
import numpy as np


def get_rank_mat(time, event):
    """
    動態計算 rank_mat（N×N 矩陣）
    rank_mat[i,j] = 1 表示：病患 i 死得比 j 早且 i 確實死亡
    """
    time_i = time.unsqueeze(1)
    time_j = time.unsqueeze(0)
    rank_mat = ((time_i < time_j) & (event.unsqueeze(1) == 1)).float()
    return rank_mat


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss
    來源：Cox (1972), Journal of Royal Statistical Society
    假設風險比例恆定（Proportional Hazards）
    最經典的生存分析 Loss，作為本研究基準線
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_pred, time, event):
        risk_pred = risk_pred.view(-1)
        time = time.view(-1)
        event = event.view(-1)
        sort_idx = torch.argsort(time, descending=True)
        risk_pred_sorted = risk_pred[sort_idx]
        event_sorted = event[sort_idx]
        risk_exp = torch.exp(risk_pred_sorted)
        risk_cumsum = torch.cumsum(risk_exp, dim=0)
        log_risk_cumsum = torch.log(risk_cumsum + 1e-8)
        loss_vector = risk_pred_sorted - log_risk_cumsum
        loss = -torch.sum(event_sorted * loss_vector) / (torch.sum(event_sorted) + 1e-8)
        return loss


class RankingLoss(nn.Module):
    """
    Pairwise Ranking Loss（Hinge Loss）
    來源：Burges et al. (2005), RankNet
    公式：L = Σ max(0, margin - (θᵢ - θⱼ)) for pairs where tᵢ<tⱼ, δᵢ=1
    選用原因：直接優化排序，和 C-Index 評估指標直接對應
    不需要假設 Proportional Hazards
    """
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, risk_pred, time, event):
        risk_pred = risk_pred.view(-1)
        time = time.view(-1)
        event = event.view(-1)
        time_i = time.unsqueeze(1)
        time_j = time.unsqueeze(0)
        valid_pairs = (time_i < time_j) & (event.unsqueeze(1) == 1)
        risk_i = risk_pred.unsqueeze(1)
        risk_j = risk_pred.unsqueeze(0)
        diff = risk_i - risk_j
        pair_loss = torch.clamp(self.margin - diff, min=0)
        n_pairs = valid_pairs.float().sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=risk_pred.device, requires_grad=True)
        result = (pair_loss * valid_pairs.float()).sum() / n_pairs
        if not result.requires_grad:
            result = result + 0.0 * risk_pred.sum()
        return result


class DeepHitLoss(nn.Module):
    """
    DeepHit Loss（自行實作，基於論文公式）
    來源：Lee et al. (2018), NeurIPS
    "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks"
    公式：L = alpha * L_NLL + (1-alpha) * L_Rank
    L_NLL = -log p(t|x) for dead, -log S(t|x) for censored
    L_Rank = Σ exp(-(F(tᵢ|xᵢ) - F(tᵢ|xⱼ)) / sigma) for concordant pairs
    選用原因：不假設 Proportional Hazards，同時學習死亡時間分布和排序
    注意：需要模型輸出 num_bins 維，rank_mat 在訓練時動態計算
    """
    def __init__(self, num_bins=10, alpha=0.2, sigma=0.1):
        super(DeepHitLoss, self).__init__()
        self.num_bins = num_bins
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, phi, time_bin, event):
        import torch.nn.functional as F
        time_bin = time_bin.long()
        event = event.float()
        batch_size = phi.size(0)

        # 確保 time_bin 在有效範圍內
        time_bin = torch.clamp(time_bin, 0, self.num_bins - 1)

        # 計算概率分布
        log_prob = F.log_softmax(phi, dim=1)  # 數值穩定
        prob = torch.exp(log_prob)

        # === L_NLL ===
        nll = torch.tensor(0.0, device=phi.device)
        for i in range(batch_size):
            t = time_bin[i].item()
            if event[i] == 1:
                # 死亡：最大化在時間 t 死亡的概率
                nll -= log_prob[i, t]
            else:
                # 截尾：最大化在時間 t 之後存活的概率
                surv = prob[i, t:].sum()
                nll -= torch.log(surv + 1e-8)
        nll = nll / batch_size

        # === L_Rank ===
        cum_prob = torch.cumsum(prob, dim=1)  # 累積死亡概率 F(t|x)
        rank_loss = torch.tensor(0.0, device=phi.device)
        n_pairs = 0

        for i in range(batch_size):
            if event[i] == 0:
                continue
            t_i = time_bin[i].item()
            for j in range(batch_size):
                if i == j:
                    continue
                t_j = time_bin[j].item()
                if t_i >= t_j:
                    continue
                # i 死得比 j 早，i 的累積死亡概率應該高於 j
                diff = cum_prob[i, t_i] - cum_prob[j, t_i]
                rank_loss += torch.exp(-diff / self.sigma)
                n_pairs += 1

        if n_pairs > 0:
            rank_loss = rank_loss / n_pairs

        return self.alpha * nll + (1 - self.alpha) * rank_loss

    def get_risk_score(self, phi):
        """從 num_bins 維輸出計算 1 維風險分數
        用期望死亡時間的倒數：risk = 1/E[T]
        """
        import torch.nn.functional as F
        prob = F.softmax(phi, dim=1)
        # 時間區間權重（0~9）
        bins = torch.arange(self.num_bins, dtype=torch.float32, device=phi.device)
        # 期望死亡時間 E[T] = Σ t * p(T=t)
        expected_time = (prob * bins).sum(dim=1)
        # 風險 = 1/E[T]
        risk = 1.0 / (expected_time + 1e-8)
        return risk


class MTLRLoss(nn.Module):
    """
    MTLR Loss（自行實作，基於論文公式）
    來源：Fotso (2018)
    "Deep Neural Networks for Survival Analysis Based on a Multi-Task Framework"
    公式：在每個時間點做二元分類，存活概率保證單調遞減
    選用原因：不假設 Proportional Hazards，比 DeepHit 更穩定
    存活概率保證單調遞減（符合生物學直覺）
    注意：需要模型輸出 num_bins 維
    """
    def __init__(self, num_bins=10):
        super(MTLRLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, phi, time_bin, event):
        time_bin = time_bin.long()
        event = event.float()
        batch_size = phi.size(0)

        # 確保 time_bin 在有效範圍內
        time_bin = torch.clamp(time_bin, 0, self.num_bins - 1)

        # 累積 logits（確保存活概率單調遞減）
        cum_phi = torch.flip(
            torch.cumsum(torch.flip(phi, [1]), dim=1), [1]
        )

        # 數值穩定的 log partition function
        log_z = torch.logsumexp(
            torch.cat([
                cum_phi,
                torch.zeros(batch_size, 1, device=phi.device)
            ], dim=1),
            dim=1
        )

        loss = torch.tensor(0.0, device=phi.device)
        for i in range(batch_size):
            t = time_bin[i].item()
            t = max(0, min(int(t), self.num_bins - 1))

            if event[i] == 1:
                log_prob = cum_phi[i, t] - log_z[i]
            else:
                if t + 1 < self.num_bins:
                    log_prob = cum_phi[i, t + 1] - log_z[i]
                else:
                    log_prob = -log_z[i]
            loss -= log_prob

        return loss / batch_size

    def get_risk_score(self, phi):
        """從 num_bins 維輸出計算 1 維風險分數
        用期望死亡時間的倒數：risk = 1/E[T]，存活越短風險越高
        """
        import torch.nn.functional as F
        # 計算每個時間區間的概率
        cum_phi = torch.flip(
            torch.cumsum(torch.flip(phi, [1]), dim=1), [1]
        )
        log_z = torch.logsumexp(
            torch.cat([cum_phi, torch.zeros(phi.size(0), 1, device=phi.device)], dim=1),
            dim=1, keepdim=True
        )
        # 存活概率 S(t) = P(T > t)
        surv_prob = torch.exp(cum_phi - log_z)
        # 期望死亡時間 E[T] ≈ Σ S(t)（離散情況）
        expected_time = surv_prob.sum(dim=1)
        # 風險 = 1/E[T]（存活時間越短 → 風險越高）
        risk = 1.0 / (expected_time + 1e-8)
        return risk

class LogisticHazardLoss(nn.Module):
    """
    Logistic Hazard Loss（Discrete-Time Hazard Model）
    來源：Brown et al. (1975), 深度學習版本由 Gensheimer & Narasimhan (2019) 提出
    "A scalable discrete-time survival model for neural networks"
    實作：pycox 套件（havakv/pycox），nll_logistic_hazard
    
    原理：在每個時間區間預測危險率 h(t|x) = sigmoid(phi_t)
    公式：L = -Σ [δᵢ * log(hᵢ(tᵢ)) + Σ_{t<tᵢ} log(1 - hᵢ(t))]
    
    選用原因：
    - 不假設 Proportional Hazards
    - 比 DeepHit 更簡單穩定（不需要 rank_mat）
    - 小資料集也能正常訓練
    - pycox 官方實作，結果可信
    注意：需要模型輸出 num_bins 維
    """
    def __init__(self, num_bins=10):
        super(LogisticHazardLoss, self).__init__()
        self.num_bins = num_bins
        from pycox.models.loss import nll_logistic_hazard
        self.loss_fn = nll_logistic_hazard

    def forward(self, phi, time_bin, event):
        time_bin = time_bin.long()
        event = event.float()
        time_bin = torch.clamp(time_bin, 0, self.num_bins - 1)
        return self.loss_fn(phi, time_bin, event)

    def get_risk_score(self, phi):
        """
        從 num_bins 維輸出計算 1 維風險分數
        危險率的累積乘積 = 1 - 存活概率
        """
        hazard = torch.sigmoid(phi)
        surv = torch.cumprod(1 - hazard, dim=1)
        # 用期望存活時間的倒數作為風險
        expected_surv = surv.sum(dim=1)
        return 1.0 / (expected_surv + 1e-8)
