import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models.video as video_models

class GradientMonitoringMixin:
    def monitor_gradients(self, name=""):
        for param_name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 10:
                    print(f"Large gradient norm in {name} - {param_name}: {grad_norm}")

class HARViT(nn.Module, GradientMonitoringMixin):
    def __init__(self, img_size=224, patch_size=32, in_channels=3, num_classes=101,
                 embed_dim=128, depth=2, num_heads=4, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.2, attn_drop_rate=0.2, num_frames=8):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Use pretrained R3D_18 as backbone with mixed precision
        self.backbone = video_models.r3d_18(pretrained=True)
        # Remove the last fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-2]:  # Freeze more layers
            param.requires_grad = False
        
        # Simplified patch embedding with batch norm
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, 
                     kernel_size=(1, patch_size, patch_size),
                     stride=(1, patch_size, patch_size),
                     padding=(0, 0, 0)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()  # Changed to GELU for better gradient flow
        )
        
        # Calculate number of patches
        self.n_patches = (img_size // patch_size) ** 2
        num_patches = self.n_patches * num_frames

        # Positional embedding and CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with batch norm
        self.blocks = nn.ModuleList([
            TransformerBlockWithBN(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Simplified classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 512, 512),  # Reduced dimension
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]
        
        # Get backbone features
        with torch.amp.autocast(device_type='cpu' if x.device.type == 'cpu' else 'cuda'):
            backbone_features = self.backbone(x)  # Shape: (B, 512)
        
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B, E, T, H/P, W/P)
        x = rearrange(x, 'b e t h w -> b (t h w) e')  # Shape: (B, T*H*W, E)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        # Get transformer features
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        
        # Concatenate backbone and transformer features
        x = torch.cat([x, backbone_features], dim=1)
        
        # Classification head
        x = self.head(x)
        
        # Monitor gradients during backward pass
        if self.training:
            self.monitor_gradients("HARViT")
        
        return x

class TransformerBlockWithBN(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x + self.attn(self.norm1(x))
        # Reshape for batch norm
        x2 = self.norm2(x)
        x2 = x2.reshape(-1, x2.size(-1))  # (B*N, C)
        x2 = self.mlp(x2)
        x2 = x2.reshape(B, N, C)  # (B, N, C)
        x = x + x2
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 