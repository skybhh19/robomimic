import torch.nn as nn
import torch
import numpy as np


class MlpExtractor(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout):
        super(MlpExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, obs):
        features = self.feature(obs)
        return features


class MlpSepExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, dropout):
        super(MlpSepExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.object_feature = nn.Sequential(
            nn.Linear(object_dim, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, robot_obs, object_obs):
        object_features = self.object_feature(object_obs)
        features = torch.cat((object_features, robot_obs), dim=-1)
        return features


def scaled_dot_product_attention(q, k, v):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.shape[-1]
    scaled_qk = matmul_qk / np.sqrt(dk)
    attention_weights = nn.functional.softmax(scaled_qk, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class SelfAttentionBase(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads):
        super(SelfAttentionBase, self).__init__()
        self.n_heads = n_heads
        self.q_linear = nn.Linear(input_dim, feature_dim)
        self.k_linear = nn.Linear(input_dim, feature_dim)
        self.v_linear = nn.Linear(input_dim, feature_dim)
        self.dense = nn.Linear(feature_dim, feature_dim)

    def split_head(self, x):
        x_size = x.size()
        assert isinstance(x_size[2] // self.n_heads, int)
        x = torch.reshape(x, [-1, x_size[1], self.n_heads, x_size[2] // self.n_heads])
        x = torch.transpose(x, 1, 2) # (batch_size, n_heads, seq_len, depth)
        return x

    def forward(self, q, k, v):
        assert len(q.size()) == 3
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q_heads = self.split_head(q)
        k_heads = self.split_head(k)
        v_heads = self.split_head(v)
        attention_out, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads)
        attention_out = torch.transpose(attention_out, 1, 2)  # (batch_size, seq_len_q, n_heads, depth)
        out_size = attention_out.size()
        attention_out = torch.reshape(attention_out, [-1, out_size[1], out_size[2] * out_size[3]])
        attention_out = self.dense(attention_out)
        return attention_out


class SelfAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads):
        super(SelfAttentionExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size)
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )
        self.dropout = nn.ModuleList(
            [nn.Dropout(0.2) for _ in range(n_attention_blocks)]
        )
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )

    def forward(self, robot_obs, object_obs):
        n_object = object_obs.shape[1]
        objects_obs = torch.cat([robot_obs.unsqueeze(dim=1).repeat(1, n_object, 1), object_obs], dim=-1)
        features = self.embed(objects_obs)
        for i in range(self.n_attention_blocks):
            features = self.dropout[i](features)
            attn_output = self.attention_blocks[i](features, features, features)
            out1 = self.layer_norm1[i](features + attn_output)
            ffn_out = self.feed_forward_network[i](out1)
            features = self.layer_norm2[i](ffn_out)
        features = torch.mean(features, dim=1)
        return features


class CrossAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size):
        super(CrossAttentionExtractor, self).__init__()
        self.robot_embed = nn.Sequential(
            nn.Linear(robot_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU()
        )
        self.object_embed = nn.Sequential(
            nn.Linear(object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU()
        )
        self.ln = nn.LayerNorm(hidden_size // 2)

    def forward(self, robot_obs, objects_obs):
        robot_embedding = self.robot_embed(robot_obs)
        objects_embedding = self.object_embed(objects_obs)
        weights = torch.matmul(robot_embedding.unsqueeze(dim=1), objects_embedding.transpose(1, 2)) / np.sqrt(objects_embedding.size()[2])
        weights = nn.functional.softmax(weights, dim=-1)
        weighted_feature = torch.matmul(weights, objects_embedding).squeeze(dim=1)
        weighted_feature = nn.functional.relu(self.ln(weighted_feature))
        return torch.cat([robot_embedding, weighted_feature], dim=-1)

class SelfAttentionExtractor2(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads, dropout_prob=0.0):
        super(SelfAttentionExtractor2, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )
        self.dropout = nn.ModuleList(
            [nn.Dropout(self.dropout_prob) for _ in range(n_attention_blocks)]
        )
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(self.dropout_prob),
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )

    def forward(self, robot_obs, object_obs):
        n_object = object_obs.shape[1]
        objects_obs = torch.cat([robot_obs.unsqueeze(dim=1).repeat(1, n_object, 1),
                                 object_obs,], dim=-1)
        features = self.embed(objects_obs)
        for i in range(self.n_attention_blocks):
            features = self.dropout[i](features)
            attn_output = self.dropout[i](self.attention_blocks[i](features, features, features))
            out1 = self.layer_norm1[i](features + attn_output)
            features = out1
            # ffn_out = self.feed_forward_network[i](out1)
            # features = self.layer_norm2[i](out1 + ffn_out)
        features = torch.mean(features, dim=1)
        return features

class GoalConditionedSelfAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads, dropout_prob=0.0):
        super(GoalConditionedSelfAttentionExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim + robot_dim + object_dim, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )
        self.dropout = nn.ModuleList(
            [nn.Dropout(self.dropout_prob) for _ in range(n_attention_blocks)]
        )
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(self.dropout_prob),
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)]
        )

    def forward(self, robot_obs, object_obs, robot_goals, object_goals):
        n_object = object_obs.shape[1]
        objects_obs = torch.cat([robot_obs.unsqueeze(dim=1).repeat(1, n_object, 1),
                                 object_obs,
                                 robot_goals.unsqueeze(dim=1).repeat(1, n_object, 1),
                                 object_goals], dim=-1)
        features = self.embed(objects_obs)
        for i in range(self.n_attention_blocks):
            features = self.dropout[i](features)
            attn_output = self.dropout[i](self.attention_blocks[i](features, features, features))
            out1 = self.layer_norm1[i](features + attn_output)
            ffn_out = self.feed_forward_network[i](out1)
            features = self.layer_norm2[i](out1 + ffn_out)
        features = torch.mean(features, dim=1)
        return features

class DeepSetExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size):
        super(DeepSetExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.object_feature = nn.Sequential(
            nn.Linear(2*(robot_dim + object_dim), hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

    def forward(self, robot_obs, object_obs, robot_goals, object_goals):
        n_object = object_obs.shape[1]
        objects_obs = torch.cat([robot_obs.unsqueeze(dim=1).repeat(1, n_object, 1),
                                 object_obs,
                                 robot_goals.unsqueeze(dim=1).repeat(1, n_object, 1),
                                 object_goals], dim=-1)
        objects_features = self.object_feature(objects_obs)
        features = torch.mean(objects_features, dim=1)
        return features