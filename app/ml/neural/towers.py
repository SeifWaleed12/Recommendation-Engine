"""
Two-Tower Neural Network Architectures

Defines UserTower, ItemTower, and the combined TwoTowerModel for
learning user and item embeddings via BPR loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    User embedding tower.

    Architecture:
        Embedding(n_users, 64) → concat user_features(16)
        → Linear(80, 256) → ReLU → Dropout(0.2)
        → Linear(256, 128) → L2 normalize
    """

    def __init__(self, n_users: int, feature_dim: int = 16, embed_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.fc1 = nn.Linear(embed_dim + feature_dim, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, user_idx: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx: (batch,) int tensor
            user_features: (batch, 16) float tensor

        Returns:
            L2-normalized embeddings of shape (batch, 128)
        """
        embed = self.user_embedding(user_idx)  # (batch, 64)
        x = torch.cat([embed, user_features], dim=1)  # (batch, 80)
        x = F.relu(self.fc1(x))  # (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 128)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


class ItemTower(nn.Module):
    """
    Item embedding tower.

    Architecture:
        Embedding(n_items, 64) → concat item_features(16)
        → Linear(80, 256) → ReLU → Dropout(0.2)
        → Linear(256, 128) → L2 normalize
    """

    def __init__(self, n_items: int, feature_dim: int = 16, embed_dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.fc1 = nn.Linear(embed_dim + feature_dim, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, item_idx: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_idx: (batch,) int tensor
            item_features: (batch, 16) float tensor

        Returns:
            L2-normalized embeddings of shape (batch, 128)
        """
        embed = self.item_embedding(item_idx)  # (batch, 64)
        x = torch.cat([embed, item_features], dim=1)  # (batch, 80)
        x = F.relu(self.fc1(x))  # (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 128)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


class TwoTowerModel(nn.Module):
    """
    Two-Tower model wrapping UserTower and ItemTower.

    forward() computes the dot product between user and item embeddings,
    producing a scalar relevance score per pair.
    """

    def __init__(self, n_users: int, n_items: int, feature_dim: int = 16):
        super().__init__()
        self.user_tower = UserTower(n_users, feature_dim=feature_dim)
        self.item_tower = ItemTower(n_items, feature_dim=feature_dim)

    def forward(
        self,
        user_idx: torch.Tensor,
        user_feats: torch.Tensor,
        item_idx: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dot-product scores between user and item embeddings.

        Returns:
            Scalar scores of shape (batch,)
        """
        user_embed = self.user_tower(user_idx, user_feats)  # (batch, 128)
        item_embed = self.item_tower(item_idx, item_feats)  # (batch, 128)
        scores = (user_embed * item_embed).sum(dim=1)  # (batch,)
        return scores

    def get_user_embedding(
        self, user_idx: torch.Tensor, user_feats: torch.Tensor
    ) -> torch.Tensor:
        """Get user embedding without computing scores."""
        return self.user_tower(user_idx, user_feats)

    def get_item_embedding(
        self, item_idx: torch.Tensor, item_feats: torch.Tensor
    ) -> torch.Tensor:
        """Get item embedding without computing scores."""
        return self.item_tower(item_idx, item_feats)
