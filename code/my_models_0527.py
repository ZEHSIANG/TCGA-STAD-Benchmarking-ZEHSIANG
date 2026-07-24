import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. SNN (Self-Normalizing Network) - Clinical
# ==========================================
class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(SNN, self).__init__()
        # SNN 標準結構: Linear -> ELU -> AlphaDropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout)
        )
        # 最終輸出一個 risk score (1維)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.classifier(x)
        return logits

# ==========================================
# 2. Gated Attention (ABMIL 的核心元件)
# ==========================================
class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention_a = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_c = nn.Linear(self.D, self.K)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # (N, K)
        return A

# ==========================================
# 3. ABMIL (Attention-Based MIL)
# ==========================================
class ABMIL(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super(ABMIL, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = GatedAttention()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x: (1, N_patches, input_dim) -> 我們先壓扁成 (N, input_dim)
        x = x.squeeze(0) 
        
        h = self.fc(x)  # (N, 512)
        
        # 計算 Attention 分數
        A = self.attention_net(h)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = F.softmax(A, dim=1)  # 歸一化
        
        # 加權平均 (Aggregation)
        M = torch.mm(A, h)  # (1, 512)
        
        # 分類
        logits = self.classifier(M)
        return logits
        
    # 用於 Fusion 時提取特徵用
    def get_embedding(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        A = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h) # (1, 512)
        return M

# ==========================================
# 4. CLAM_SB (單分支版本)
# ==========================================
class CLAM_SB(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super(CLAM_SB, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = GatedAttention()
        
        # CLAM 特有的 Instance Classifiers (雖然 Cox 用不太到，但保留結構)
        self.classifiers = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        
        A = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        return logits

    def get_embedding(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        A = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        return M

# ==========================================
# 5. TransMIL (Transformer-based MIL)
# ==========================================
class TransMIL(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super(TransMIL, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        
        # PPEG (位置編碼)
        self.pos_layer = PPEG(dim=512)
        
        # Transformer Layer
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = x.squeeze(0) # (N, dim)
        h = self.fc(x)   # (N, 512)
        
        # TransMIL 需要 batch 維度
        h = h.unsqueeze(0) # (1, N, 512)
        
        # PPEG & Layers
        h = self.pos_layer(h)
        h = self.layer1(h)
        h = self.layer2(h)
        
        h = self.norm(h)[:, 0] # 取出 Class Token (或是 Mean Pooling)
        
        logits = self.classifier(h)
        return logits

    def get_embedding(self, x):
        x = x.squeeze(0)
        h = self.fc(x).unsqueeze(0)
        h = self.pos_layer(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)[:, 0]
        return h

# TransMIL 的輔助層
import math 

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        # PPEG 使用卷積來捕捉局部位置特徵
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, Dim]
        B, _, C = x.shape
        
        # 1. 計算目前的長度
        N = x.shape[1]
        
        # 2. 找出「下一個」完美平方數
        # 例如 N=1000 -> ceil(31.6) = 32 -> H=32 -> Needed=1024
        H = int(math.ceil(N ** 0.5))
        needed_N = H * H
        
        # 3. 如果不夠長，就補零 (Padding)
        if needed_N > N:
            pad_len = needed_N - N
            # 建立全 0 的 padding [B, pad_len, C]
            padding = torch.zeros(B, pad_len, C).to(x.device)
            # 接起來 -> [B, 1024, C]
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x
            
        # 4. 變形成 2D 圖片 [B, C, H, H] 以便進行卷積
        # transpose(1, 2) 把 Dim 換到 Channel 位置
        feat_token = x_padded.transpose(1, 2).view(B, C, H, H)
        
        # 5. 執行卷積 (PPEG 核心)
        cnn_feat = self.proj(feat_token) + self.proj1(feat_token) + self.proj2(feat_token)
        
        # 6. 變形回來 [B, H*H, C]
        x_out = cnn_feat.flatten(2).transpose(1, 2)
        
        # 7. 切掉剛剛補的零，還原成原始長度 N
        x = x + x_out[:, :N, :]
        
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(dim)
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class NystromAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = 8
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # 簡化版 Attention，實際 TransMIL 使用 Nystrom 加速，這裡用標準 Attention 模擬
        # 為了避免缺少套件，我們先用 PyTorch 內建的 MultiheadAttention
        # 注意：若要完整復現 TransMIL 需安裝 nystrom-attention 套件
        # 這裡為了方便您運行，我們使用標準 Self-Attention
        B, N, C = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = self.out(x)
        return x

# ==========================================
# 6. Fusion Model (Multimodal)
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, wsi_backbone, clin_dim=3, hidden_dim=256):
        super(FusionModel, self).__init__()
        self.wsi_backbone = wsi_backbone
        
        # Clinical Embedding Net
        self.clin_net = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion Classifier (WSI 512 + Clin 256 = 768)
        self.classifier = nn.Sequential(
            nn.Linear(512 + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Risk Score
        )

    def forward(self, wsi_x, clin_x):
        # 1. 取得 WSI Embedding (不是 Logits)
        wsi_emb = self.wsi_backbone.get_embedding(wsi_x) # (1, 512)
        
        # 2. 取得 Clinical Embedding
        clin_emb = self.clin_net(clin_x) # (Batch, 256)
        
        # 3. Concatenation
        if wsi_emb.shape[0] != clin_emb.shape[0]:
             # 如果 Physical Batch Size > 1，這邊要注意維度對齊
             # 但目前 Physical BS=1，可以直接 cat
             pass
             
        fusion_emb = torch.cat((wsi_emb, clin_emb), dim=1) # (1, 768)
        
        # 4. Final Prediction
        logits = self.classifier(fusion_emb)
        return logits
# ==========================================
# 6. MeanMIL (Baseline: Average Pooling)
# ==========================================
class MeanMIL(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super(MeanMIL, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = x.squeeze(0) # (N, dim)
        h = self.fc(x)   # (N, 512)
        
        # 直接算平均，沒有 Attention
        M = torch.mean(h, dim=0, keepdim=True) # (1, 512)
        
        logits = self.classifier(M)
        return logits

    def get_embedding(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        M = torch.mean(h, dim=0, keepdim=True)
        return M

# ==========================================
# 7. MaxMIL (Baseline: Max Pooling)
# ==========================================
class MaxMIL(nn.Module):
    def __init__(self, input_dim, dropout=0.25):
        super(MaxMIL, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        
        # 直接取最大值
        M, _ = torch.max(h, dim=0, keepdim=True) # (1, 512)
        
        logits = self.classifier(M)
        return logits

    def get_embedding(self, x):
        x = x.squeeze(0)
        h = self.fc(x)
        M, _ = torch.max(h, dim=0, keepdim=True)
        return M


# ==========================================
# [修正版] NaiveFusionModel
# 修正重點：在 __init__ 中將 backbone 的分類器移除，
# 確保 backbone(x) 回傳的是 512 維特徵，而不是 1 維分數。
# ==========================================
class NaiveFusionModel(nn.Module):
    def __init__(self, backbone, clin_dim=3, hidden_dim=512):
        super(NaiveFusionModel, self).__init__()
        
        # 1. 影像處理部分 (Backbone)
        self.backbone = backbone
        
        # === [關鍵修正] ===
        # 檢查 backbone 是否有 'classifier' 或 'fc' 層，將其替換為 Identity (不做任何事)
        # 這樣 backbone 就會直接輸出 512 維的特徵向量，而不是被壓縮成 1 維的預測分數
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        # =================
        
        # 2. 融合後的維度
        # 512 (WSI Feature) + 3 (Raw Clinical) = 515
        self.fusion_dim = 512 + clin_dim 
        
        # 3. 分類器
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, x_wsi, x_clin):
        # Step A: 影像特徵提取
        # 因為上面把 classifier 變成 Identity 了，這裡會回傳 [Batch, 512]
        wsi_feat = self.backbone(x_wsi)
        
        # Step B: 臨床數據直接拿來用 [Batch, 3]
        
        # Step C: 串接 [Batch, 512] + [Batch, 3] -> [Batch, 515]
        combined = torch.cat((wsi_feat, x_clin), dim=1)
        
        # Step D: 預測
        logits = self.classifier(combined)
        return logits
