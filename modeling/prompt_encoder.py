import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Type, Optional
import math
from segment_anything.modeling.common import MLPBlock
from transformers import AutoTokenizer, AutoModel


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim, hidden_dim=mid_dim, output_dim=input_dim, num_layers=2
        )

    def forward(self, features):
        out = features + self.model(features)
        return out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        num_classes: int = 3 # Include the background class
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        self.num_classes = num_classes # Include the background class
        
        # Update the label_proj to match our dimensions
        # If coords is 256 (point_embedding dimension) and we add 3 normalized labels
        # self.label_proj = nn.Linear(256+num_classes, embedding_dim)  # Fixed dimension

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_coord,
        #point_labels
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        #print(f"point_coord shape: {point_coord.shape}")

        # It is probably better if we first grid_sample the points and then add the label information

        # Create pseudo-embedding through projection

        # Label information is already incorporated in image_embedding
        point_embedding = F.grid_sample(image_embedding, point_coord, align_corners=False).squeeze(2).squeeze(2)
        point_embedding = point_embedding.permute(0, 2, 1)

        #print(f"point_embedding shape: {point_embedding.shape}")
        #print(f"image_pe shape: {image_pe.shape}")
        #print(f"Here we want to add 3 more in the 3rd dimension representing the class label")
        point_pe = F.grid_sample(image_pe, point_coord, align_corners=False).squeeze(2).squeeze(2)
        point_pe = point_pe.permute(0, 2, 1)
        #print(f"point_pe shape: {point_pe.shape}")
        	
        # def _add_label_to_coordinates(coords, labels, num_classes):
        #     # Convert labels to one-hot encoding
        #     #print(f"labels: {labels}")
        #     #print(f"labels shape: {labels.shape}")
        #     #print(f"Coords shape: {coords.shape}")
        #     #print(f"num_classes: {num_classes}")
        #     #print(f"Unique labels: {torch.unique(labels)}")
            
        #     # Instead of using one-hot encoding, simply normalize the label values
        #     # This keeps the same shape [batch_size, num_points, 3]
        #     normalized_labels = labels.float() / (num_classes - 1)  # Normalize to [0,1]
        #     #print(f"normalized_labels shape: {normalized_labels.shape}, dtype: {normalized_labels.dtype} \n squeezed: {normalized_labels.squeeze(0).shape}")
        #     # Concatenate with coordinates
        #     augmented_coords = torch.cat([coords, normalized_labels.squeeze(0)], dim=-1)
            
        #     return augmented_coords

        # augmented_coords = _add_label_to_coordinates(point_embedding.squeeze(), point_labels, self.num_classes)
        # #print(f"augmented_coords shape: {augmented_coords.shape}")
        # adjusted_point_embedding = self.label_proj(augmented_coords).unsqueeze(0)
        # #print(f"adjusted_point_embedding shape: {adjusted_point_embedding.shape}")
        original_shape = image_embedding.shape
        #print(f"original_shape: {original_shape}")
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        #print(f"image_embedding shape: {image_embedding.shape}")
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        #print(f"image_pe shape: {image_pe.shape}")
        ##print(f"point_labels shape: {point_labels.shape}")
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            image_embedding, point_embedding = layer(
                image_embedding,
                point_embedding,
# Labels are useless when I am not concatenating stuff at the forward pass
                #adjusted_point_embedding,
                image_pe,
                point_pe,
            )
        return image_embedding


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(1, 10, embedding_dim))

    def forward(self, img_embed, point_embed, img_pe, point_pe) -> Tuple[Tensor, Tensor]:
        
        #print(f"img_embed shape: {img_embed.shape}")
        #print(f"point_embed shape: {point_embed.shape}")
        q = torch.cat([self.global_query, point_embed], dim=1)
        self_out = self.self_attn(q=q, k=q, v=q)
        self_out = self.norm1(self_out)

        # Cross attention block, tokens attending to image embedding
        queries = q + self_out
        queries = self.norm2(queries)
        point_embed = queries[:, 10:, :]
        queries = queries[:, :10, :]

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_token(q=img_embed, k=queries, v=queries)
        keys = img_embed + attn_out
        keys = self.norm4(keys)

        return keys, point_embed


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class PromptEncoder(nn.Module): # Changed to include label embedding
    def __init__(
        self,
        *,
        transformer: nn.Module,
        num_pos_feats: int = 128,
        mask_prompt = False,
        num_classes = 2
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_classes = num_classes
        # Add label embedding layer
        # self.label_embedding = nn.Embedding(num_classes + 1, num_pos_feats)  # +1 for background

        self.register_buffer(
            "positional_encoding_gaussian_matrix",
             torch.randn((3, num_pos_feats)),
        )
        self.mask_prompt = mask_prompt
        if mask_prompt:
            self.default_prompt = nn.parameter.Parameter(torch.randn(1, 256, 32, 32, 32))
            self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 256 // 4, kernel_size=3, stride=3),
            LayerNorm3d(256 // 4),
            nn.GELU(),
            nn.Conv3d(256 // 4, 256, kernel_size=3, padding = 1, stride=1),
            LayerNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 256, kernel_size=1),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coord,
        #point_labels,
        img_size = [512, 512, 32],
        feat_size = [32, 32, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # Generate the positional encoding after the label embeddings are added to the image embeddings
        # image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        '''
        if self.mask_prompt:
            if masks == None:
                image_embeddings += self.default_prompt
            else:
                image_embeddings += self.mask_encoder(masks)
        '''
        
        
        point_coord[:, :, 0] = (point_coord[:, :, 0]+0.5) * 2 / img_size[2] - 1
        point_coord[:, :, 1] = (point_coord[:, :, 1]+0.5) * 2 / img_size[1] - 1
        point_coord[:, :, 2] = (point_coord[:, :, 2]+0.5) * 2 / img_size[0] - 1
        point_coord = point_coord.reshape(1,1,1,-1,3)
        #print(f"point_coord shape: {point_coord.shape}")
        # Get label embeddings and combine with coordinate features
        #label_emb = self.label_embedding(point_labels.long())  # [B, N_points, C]
        ##print(f"label_emb shape: {label_emb.shape}")
        #print(f"image_embeddings shape: {image_embeddings.shape}")
        #label_emb = label_emb.permute(0, 3, 1, 2).unsqueeze(-1)  # [B, C, N_points, 1, 1]

        # Combine with image embeddings
        image_embeddings = image_embeddings #+ label_emb

        # Generate the positional encoding after the label embeddings are added to the image embeddings
        image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        features = self.transformer(image_embeddings, image_pe, point_coord)#, point_labels)
        features = features.transpose(1,2).reshape([1, -1] + feat_size)

        return features

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords * 3 / 2
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def get_img_pe(self, size: Tuple[int, int], device) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2).unsqueeze(0)  # C x D X H x W


############################  GPT2  ####################################
class CrossAttentionModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, tensor_5d, tensor_2d):
        # Reshape 5D tensor to 2D for attention
        b, c, d, h, w = tensor_5d.shape
        tensor_5d_flat = tensor_5d.flatten(2).permute(0, 2, 1)  # [B, D*H*W, C]
        # Get a 3D tensor at this point for LLAMA
        if tensor_2d.ndim == 3: 
            expanded_tensor = tensor_2d.expand(-1, tensor_5d_flat.size(1), -1) # [b, 32768, 256] 32^3 = 32768
        elif tensor_2d.ndim == 2:
            expanded_tensor = tensor_2d.unsqueeze(1).expand(-1, tensor_5d_flat.size(1), -1) # [b, 32768, 256] 32^3 = 32768
        else:
             print(f"Tensor has {tensor_2d.ndim} dimensions, which is not handled by this logic for expansion.")
        
        # Process in chunks to save memory
        chunk_size = 1024  # 1024 originally
        num_chunks = (tensor_5d_flat.shape[1] + chunk_size - 1) // chunk_size
        
        outputs = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, tensor_5d_flat.shape[1])
            chunk = tensor_5d_flat[:, start_idx:end_idx, :]
            
            # Process this chunk
            chunk_output, _ = self.cross_attention(query=expanded_tensor, key=chunk, value=chunk)
            outputs.append(chunk_output)
        
        # Combine results if processing in chunks
        if num_chunks > 1:
            attn_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        else:
            attn_output = outputs[0]
        return attn_output.permute(0, 2, 1).view(b, c, d, h, w) 

class PromptEncoder_Text(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_classes=3,
        device = "cuda:0"
    ) -> None:
        from transformers import GPT2Model, GPT2Tokenizer
        super().__init__()
        self.device = device
        # Load GPT-2 tokenizer and model
        model_name="gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model_gpt = GPT2Model.from_pretrained(model_name)
        # Freeze GPT-2/LLAMA-3.2-1b parameters
        for param in self.model_gpt.parameters():
            param.requires_grad = False

        # Class-specific embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))

        hidden_dim = self.model_gpt.config.hidden_size
        # Will print 768 for GPT2 and 2048 for LLAMA
        # These are the outputs that the linear projectiong will project to 256 so that it matches the previous model dimensions.
        print(f"hidden_dim: {hidden_dim}") 

        self.projector = nn.Linear(768, 256)

        # Fuse image and text features to find how they are related
        self.cross_attention = CrossAttentionModel(embed_dim=embed_dim, num_heads=8)   

    def forward(
        self,
        image_embeddings: torch.Tensor,
        inputs_prompt,
        class_idx = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Tokenize and convert to tensor
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # Extract hidden states (text embeddings)
        if isinstance(inputs_prompt, str):
            # Create proper input format for GPT2
            inputs_prompt = {"input_ids": self.tokenizer(inputs_prompt, return_tensors="pt").input_ids.to(self.device)}
        with torch.no_grad():
            outputs = self.model_gpt(**inputs_prompt)
        # Get last hidden state (shape: [batch_size, sequence_length, hidden_dim])
        hidden_states = outputs.last_hidden_state
        pooled_features = hidden_states.mean(dim=1)
        projected_features = self.projector(pooled_features)

        # Class-specific embeddings
        
        # Add class embedding during training 
        if class_idx is not None: # During inference class_idx is None.
            projected_features = projected_features + self.class_embeddings[class_idx].unsqueeze(0)
        
        fused_features = self.cross_attention(image_embeddings, projected_features)
        return fused_features


############################  LLAMA3.2  ####################################

############################## Shared Text Encoder ##############################
# class PromptEncoderTextBase(nn.Module):
#     """
#     Identical logic for GPT-2 and LLaMA, differing only by model_name
#     and the resulting hidden_size used in the projector.
#     """
#     def __init__(
#         self,
#         model_name,
#         embed_dim = 256,
#         num_classes = 3,
#         device = "cuda:0"
#     ) -> None:
#         super().__init__()
#         self.device = device

#         # Tokenizer and model
#         if model_name == 'gpt2':
#             from transformers import GPT2Model, GPT2Tokenizer
#             self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#             self.model_gpt = GPT2Model.from_pretrained(model_name)
#         elif "llama" in model_name:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModel.from_pretrained(model_name)

#         # Ensure pad token exists for batch collation, but pooling is simple mean
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         # Freeze
#         for p in self.model.parameters():
#             p.requires_grad = False

#         # Class embeddings
#         self.class_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))

#         # Project to the target embed_dim using the model hidden size
#         hidden_dim = self.model.config.hidden_size
#         print(f"hidden_dim: {hidden_dim}")
#         self.projector = nn.Linear(hidden_dim, embed_dim)

#         # Cross attention
#         self.cross_attention = CrossAttentionModel(embed_dim=embed_dim, num_heads=8)

#         # Move backbone to device
#         self.model.to(self.device)

#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         inputs_prompt,
#         class_idx = None
#     ) -> torch.Tensor:
        
#         # Tokenize and convert to tensor
#         # inputs = self.tokenizer(prompt, return_tensors="pt")
#         # Extract hidden states (text embeddings)
#         if isinstance(inputs_prompt, str):
#             # Create proper input format for GPT2
#             inputs_prompt = {"input_ids": self.tokenizer(inputs_prompt, return_tensors="pt").input_ids.to(self.device)}
#         with torch.no_grad():
#             outputs = self.model(**inputs_prompt)
#         # Get last hidden state (shape: [batch_size, sequence_length, hidden_dim])
#         hidden_states = outputs.last_hidden_state #[1, 768] or [1, 2048]
#         pooled_features = hidden_states.mean(dim=1)
#         projected_features = self.projector(pooled_features) # [256] or [1, 256]

#         # Class-specific embeddings
        
#         # Add class embedding during training 
#         if class_idx is not None: # During inference class_idx is None.
#             projected_features = projected_features + self.class_embeddings[class_idx].unsqueeze(0)
        
#         fused_features = self.cross_attention(image_embeddings, projected_features)
#         return fused_features

############################ Specific Variants ##################################
# class PromptEncoder_Text_GPT2(PromptEncoderTextBase):
#     def __init__(self, embed_dim=256, num_classes=3, device="cuda:0"):
#         super().__init__(
#             model_name="gpt2",
#             embed_dim=embed_dim,
#             num_classes=num_classes,
#             device=device,
#         )

class PromptEncoder_Text_Llama(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_classes=3,
        device="cuda:0",
        model_name="meta-llama/llama-3.2-1b"
    ):
        super().__init__()
        self.device = device
        # Load LLAMA tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze GPT-2/LLAMA-3.2-1b parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Class-specific embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))

        hidden_dim = self.model.config.hidden_size
        # Will print 768 for GPT2 and 2048 for LLAMA
        # These are the outputs that the linear projectiong will project to 256 so that it matches the previous model dimensions.
        print(f"hidden_dim: {hidden_dim}") 

        self.projector = nn.Linear(2048, 256)

        # Fuse image and text features to find how they are related
        self.cross_attention = CrossAttentionModel(embed_dim=embed_dim, num_heads=8)   

    def forward(
        self,
        image_embeddings: torch.Tensor,
        inputs_prompt,
        class_idx = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Tokenize and convert to tensor
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # Extract hidden states (text embeddings)
        if isinstance(inputs_prompt, str):
            # Create proper input format for GPT2
            inputs_prompt = {"input_ids": self.tokenizer(inputs_prompt, return_tensors="pt").input_ids.to(self.device)}
        with torch.no_grad():
            outputs = self.model(**inputs_prompt)
        # Get last hidden state (shape: [batch_size, sequence_length, hidden_dim])
        hidden_states = outputs.last_hidden_state
        pooled_features = hidden_states.mean(dim=1)
        projected_features = self.projector(pooled_features)

        # Class-specific embeddings
        
        # Add class embedding during training 
        if class_idx is not None: # During inference class_idx is None.
            projected_features = projected_features + self.class_embeddings[class_idx].unsqueeze(0)
        
        fused_features = self.cross_attention(image_embeddings, projected_features)
        return fused_features

