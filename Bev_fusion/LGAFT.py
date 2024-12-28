import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFusionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(BEVFusionModule, self).__init__()
        
        # 1x1卷积
        self.conv1x1_lidar = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.conv1x1_camera = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.conv1x1_weight = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)  # 调整为 hidden_dim
        
        # Query, Key, Value权重矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        
        # 多层感知机（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 归一化层
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, lidar_bev, camera_bev):
        # Step 1: 1x1卷积处理LiDAR和Camera BEV特征
        F_lidar = self.conv1x1_lidar(lidar_bev)
        F_camera = self.conv1x1_camera(camera_bev)
        
        # Step 2: 计算特征权重 W_F
        concat_features = torch.cat([F_lidar, F_camera], dim=1)
        W_f = torch.sigmoid(self.conv1x1_weight(concat_features))  # 使用1x1卷积降维
        
        # Step 3: 计算Query, Key, Value
        Q = torch.cat([(1 - W_f) * F_lidar, W_f * (F_camera + self.position_encoding)], dim=1)
        Q = Q.view(Q.size(0), Q.size(1), -1).mean(-1)
        Q = self.W_q(Q)
        
        K = W_f * (F_camera + self.position_encoding)
        K = self.W_k(K.view(K.size(0), -1))
        
        V = F_camera + self.position_encoding
        V = self.W_v(V.view(V.size(0), -1))
        
        # Step 4: 自注意力机制
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Step 5: 添加和归一化
        fusion_output = self.norm(attn_output + Q)
        
        # Step 6: 多层感知机（MLP）
        output = self.mlp(fusion_output)
        
        return output
    
if __name__ == '__main__':
    # 假设输入的特征维度和隐藏层维度
    feature_dim = 128
    hidden_dim = 256

    # 初始化模型
    model = BEVFusionModule(feature_dim, hidden_dim)

    # 输入LiDAR和Camera BEV特征 (batch_size, channels, height, width)
    lidar_bev = torch.randn(1, feature_dim, 32, 32)
    camera_bev = torch.randn(1, feature_dim, 32, 32)

    # 前向传播
    output = model(lidar_bev, camera_bev)
    print(output.shape)
