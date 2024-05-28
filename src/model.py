import numpy as np
import scipy
import math
from typing import List, Mapping, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.relational_memory import RelationalMemory
from src.utils import accuracy, autoregressive_generate, cross_entropy, decode_return, encode_return, encode_reward, sample_from_logits


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        seq_len,
        attn_drop,
        resid_drop,
        use_gw: bool = False,
        memory=None,
        mem_slots=1290,
        use_topk=False,
        topk=3,
        num_steps=5,
        null_attention=False,
        shared_memory_percentage: float = 1.,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.gw = use_gw
        self.memory = memory
        self.shared_memory_percentage = shared_memory_percentage

        # memory module
        if self.gw:
            if self.memory is None:
                self.relational_memory = RelationalMemory(
                    mem_slots=mem_slots,
                    head_size=n_embd,
                    attn_drop=attn_drop,
                    num_heads=n_head,
                    num_blocks=64,
                    forget_bias=1,
                    input_bias=0,
                    gate_style="unit",
                    attention_mlp_layers=1,
                    return_all_outputs=False,
                    use_topk=use_topk,
                    topk=topk,
                    num_steps=num_steps,
                    null_attention=null_attention
                )
        else:
            self.key = nn.Linear(n_embd, n_embd)
            self.value = nn.Linear(n_embd, n_embd)
            self.query = nn.Linear(n_embd, n_embd)

            self.n_head = n_head
            self.register_buffer("mask", torch.tril(torch.ones(
                seq_len, seq_len)).view(1, 1, seq_len, seq_len))

            self.attn_drop = nn.Dropout(attn_drop)
            self.resid_drop = nn.Dropout(resid_drop)

            self.proj = nn.Linear(n_embd, n_embd)

    def forward(
        self,
        query,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        custom_causal_mask: Optional[torch.Tensor] = None,
        memory=None,
    ):

        # used for memory
        if self.gw:
            memory, out_with_mem = self.relational_memory(
                ipts=key,
                memory=memory
            )

            return out_with_mem, memory

        else:
            B, T, C = query.size()
            key = key if key is not None else query
            value = value if value is not None else query

            k = self.key(key).view(B, T, self.n_head, C //
                                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = self.query(query).view(B, T, self.n_head, C //
                                       self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.value(value).view(B, T, self.n_head, C //
                                       self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            causal_mask = custom_causal_mask
            if causal_mask is None:
                causal_mask = self.mask
            causal_mask = causal_mask[None, None, :, :]

            att = (q @ k.transpose(-2, -1)) * \
                (1.0/math.sqrt(k.size(-1)))  # (B, hn, T, T)
            mask = mask * causal_mask if mask is not None else causal_mask
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, hn, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            output = self.resid_drop(self.proj(y))
            return output

    def init_memory(self, bs, device=None):
        self.memory = self.relational_memory.initial_state(bs).to(device)


class DenseBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        seq_len,
        attn_drop,
        resid_drop,
        widening_factor=4,
        use_gw: bool = False,
    ):
        super().__init__()
        self.gw = use_gw

        self.attn_net = CausalSelfAttention(
            n_embd=n_embd, n_head=n_head, seq_len=seq_len,
            attn_drop=attn_drop, resid_drop=resid_drop, use_gw=use_gw)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, widening_factor * n_embd),
            nn.GELU(),
            nn.Linear(widening_factor * n_embd, n_embd),
            nn.Dropout(resid_drop)
        )

    def forward(self, x, mask=None, custom_causal_mask=None, memory=None):
        ipt = self.ln1(x)
        if self.gw:
            res_x, memory = self.attn_net(
                query=ipt,
                key=ipt,
                value=ipt,
                mask=mask,
                custom_causal_mask=custom_causal_mask,
                memory=memory,
            )
            x = x + res_x
        else:
            x = x + self.attn_net(ipt, mask=mask,
                                  custom_causal_mask=custom_causal_mask)
        x = x + self.mlp(self.ln2(x))
        if self.gw:
            return x, memory
        else:
            return x


class GPT2(nn.Module):
    def __init__(
        self,
        n_layers,
        n_embd,
        n_head,
        seq_len,
        attn_drop,
        resid_drop,
        use_gw=False,
        use_topk=False,
        topk=3,
        num_steps=5,
        null_attention=False,
        shared_memory_percentage: float = 1.0,
        memory=None,
    ):
        super().__init__()

        self.use_gw = use_gw
        self.memory = memory
        self.shared_memory_percentage = shared_memory_percentage

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(DenseBlock(n_embd=n_embd, n_head=n_head, seq_len=seq_len,
                                          attn_drop=attn_drop, resid_drop=resid_drop, use_gw=use_gw))
        if self.use_gw:
            if self.memory is None:
                self.relational_memory = RelationalMemory(
                    mem_slots=1290,
                    head_size=n_embd,
                    attn_drop=attn_drop,
                    num_heads=n_head,
                    num_blocks=64,
                    forget_bias=1,
                    input_bias=0,
                    gate_style="unit",
                    attention_mlp_layers=1,
                    return_all_outputs=False,
                    use_topk=use_topk,
                    topk=topk,
                    num_steps=num_steps,
                    null_attention=null_attention
                )

    def forward(
        self,
        x,
        mask=None,
        custom_causal_mask=None,
        is_training: bool = False,
        memory=None,
    ):
        if self.use_gw:
            # x size (64, 312, 512)
            memory_size = int(self.shared_memory_percentage * x.size(0))
            memory = torch.randn(memory_size, 1, x.size(2)).repeat(
                1, x.size(1), 1).to(x.device)
            if self.memory is not None:
                self.relational_memory.initial_state(
                    batch_size=x.size(0)
                ).to(x.device)
        """
        Args:
            x: Inputs, (B, T, C)
        """
        if mask is not None:
            x = x*mask[:, :, None]
            mask = mask[:, None, None, :]
        for block in self.blocks:
            if self.use_gw:
                x, memory = block(x, mask, custom_causal_mask, memory)
            else:
                x = block(x, mask=mask, custom_causal_mask=custom_causal_mask)
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_rewards: int,
        return_range: Tuple[int],
        n_layer: int,
        n_embd: int,
        n_head: int,
        seq_len: int,
        attn_drop: int,
        resid_drop: int,
        predict_reward: bool,
        single_return_token: bool,
        patch_size: Optional[Tuple[int, int]] = (14, 14),
        mnets_arch: List[int] = [256, 256],
        num_cond_embs: Optional[int] = 1,
        device: str = 'cpu',
        use_gw=False,
    ):
        super().__init__()

        self.return_range = return_range
        self.predict_reward = predict_reward
        self.num_returns = return_range[1] - return_range[0]
        self.spatial_tokens = True
        self.n_embd = n_embd
        self.single_return_token = single_return_token
        self.seq_len = seq_len

        self.transformer = GPT2(n_layers=n_layer, n_embd=n_embd, n_head=n_head,
                                seq_len=seq_len, attn_drop=attn_drop,
                                resid_drop=resid_drop, use_gw=use_gw)

        patch_height, patch_width = patch_size[0], patch_size[1]
        self.conv_net = nn.Conv2d(
            4, n_embd, (patch_height, patch_width), (patch_height, patch_width), 'valid')

        self.ret_encoder = nn.Embedding(self.num_returns+1, n_embd)
        self.act_encoder = nn.Embedding(num_actions, n_embd)
        self.rew_encoder = nn.Embedding(num_rewards, n_embd)
        self.ret_mlp = nn.Linear(n_embd, self.num_returns+1)
        self.device = device
        if self.predict_reward:
            self.rew_mlp = nn.Linear(n_embd, num_rewards)
        self.act_mlp = nn.Linear(n_embd, num_actions)

    def embed_inputs(
        self,
        obs: torch.Tensor,
        ret: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
    ):
        """
        Args:
            obs: (B, T, W, H, C)
        """
        # Embed only prefix_frames first observations.
        assert len(obs.shape) == 5

        image_dims = obs.shape[-3:]
        batch_dims = obs.shape[:2]

        # obs are [B*T, W, H, C]
        obs = obs.reshape(-1, *image_dims)
        obs = obs.to(dtype=torch.float32) / 255.0
        obs_emb = self.conv_net(obs)

        # Reshape to (B, T, P*P, D)
        obs_emb = obs_emb.reshape(*batch_dims, -1, obs_emb.shape[1])
        pos_emb = nn.Parameter(torch.normal(mean=torch.zeros(
            1, 1, obs_emb.shape[2], obs_emb.shape[3]), std=0.02)).to(device=self.device)
        obs_emb += pos_emb
        # Encode returns
        ret = encode_return(ret, self.return_range)
        rew = encode_reward(rew)
        ret_emb = self.ret_encoder(ret)
        act_emb = self.act_encoder(act)
        if self.predict_reward:
            rew_emb = self.rew_encoder(rew)
        else:
            rew_emb = None
        # obs_emb = obs_emb.reshape(*ret_emb.shape)
        return obs_emb, ret_emb, act_emb, rew_emb

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        num_batch = inputs['actions'].shape[0]
        num_steps = inputs['actions'].shape[1]
        # Embed inputs
        obs_emb, ret_emb, act_emb, rew_emb = self.embed_inputs(
            inputs['observations'], inputs['returns-to-go'],
            inputs['actions'], inputs['rewards'])

        if self.spatial_tokens:
            # obs is (B, T, W, D)
            num_obs_tokens = obs_emb.shape[2]
            # obs_emb = obs_emb.reshape(*obs_emb.shape[:2], -1)  # (B, T, W*D)

        else:
            num_obs_tokens = 1

        # Collect sequence
        if self.predict_reward:
            # 64, 28,36,256 64,28,1,256
            if len(act_emb.shape) == 3:
                ret_emb = ret_emb.unsqueeze(1)
                act_emb = act_emb.unsqueeze(1)
                rew_emb = rew_emb.unsqueeze(1)
            token_emb = torch.cat((obs_emb, ret_emb, act_emb, rew_emb), dim=2)
            tokens_per_step = num_obs_tokens + ret_emb.shape[2]*3
        else:
            token_emb = torch.cat((obs_emb, ret_emb, act_emb), dim=2)
            tokens_per_step = num_obs_tokens + 2

        token_emb = token_emb.reshape(
            (num_batch, tokens_per_step*num_steps, self.n_embd))
        # token_emb = token_emb.reshape(
        #     (num_batch, -1, self.n_embd))
        # Create position embeddings
        pos_emb = nn.Parameter(torch.zeros(
            1, token_emb.shape[1], token_emb.shape[2])).to(device=self.device)
        token_emb = token_emb + pos_emb
        # Run the transformer over the inputs
        # Token dropout
        batch_size = token_emb.shape[0]
        obs_mask = torch.ones((batch_size, num_steps, num_obs_tokens))
        ret_mask = torch.ones((batch_size, num_steps, 1))
        act_mask = torch.ones((batch_size, num_steps, 1))
        rew_mask = torch.ones((batch_size, num_steps, 1))
        if self.single_return_token:
            # Masks out all return tokens except the first one
            ret_mask[:, 1:] = 0
        if self.predict_reward:
            mask = [obs_mask, ret_mask, act_mask, rew_mask]
        else:
            mask = [obs_mask, ret_mask, act_mask]
        mask = torch.cat(mask, dim=-1)
        mask = mask.reshape((batch_size, tokens_per_step*num_steps))
        # mask = mask.reshape((batch_size, -1))

        custom_causal_mask = None
        if self.spatial_tokens:
            seq_len = token_emb.shape[1]
            sequential_causal_mask = np.tril(
                np.ones((seq_len, seq_len)))
            num_timesteps = seq_len // tokens_per_step
            num_non_obs_tokens = tokens_per_step - num_obs_tokens
            diag = [
                np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros(
                    (num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timesteps * 2)
            ]
            block_diag = scipy.linalg.block_diag(*diag)
            custom_causal_mask = np.logical_or(
                sequential_causal_mask, block_diag)
            custom_causal_mask = custom_causal_mask.astype(np.float64)

        custom_causal_mask = torch.from_numpy(
            custom_causal_mask).to(device=self.device)
        mask = mask.to(device=self.device)
        # Perception Module
        output_emb = self.transformer(
            token_emb, mask=mask, custom_causal_mask=custom_causal_mask)

        # Output_embeddings are (B, 3T, D)
        # Next token predictions (tokens one before their actual place)
        ret_pred = output_emb[:, (num_obs_tokens-1)::tokens_per_step, :]
        act_pred = output_emb[:, num_obs_tokens::tokens_per_step, :]
        embeds = torch.cat([ret_pred, act_pred], -1)
        # Project to appropriate dimensionality

        act_pred = self.act_mlp(act_pred)
        ret_pred = self.ret_mlp(ret_pred)
        # Return logits as well as pre-logits embedding.
        result_dict = {
            'embeds': embeds,
            'action_logits': act_pred,
            'return_logits': ret_pred,
        }
        if self.predict_reward:
            rew_pred = output_emb[:, (num_obs_tokens+1)::tokens_per_step, :]
            rew_pred = self.rew_mlp(rew_pred)
            result_dict['reward_logits'] = rew_pred

        # Return evaluation metrics
        result_dict['loss'] = self.sequence_loss(inputs, result_dict)
        result_dict['accuracy'] = self.sequence_accuracy(inputs, result_dict)
        return result_dict

    def _objective_pairs(
        self,
        inputs: Mapping[str, torch.Tensor],
        model_outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get logit-target paris for the model objective terms"""
        act_target = inputs['actions']
        ret_target = encode_return(inputs['returns-to-go'], self.return_range)
        act_logits = model_outputs['action_logits']
        ret_logits = model_outputs['return_logits']
        if self.single_return_token:
            ret_target = ret_target[:, :1]
            ret_logits = ret_logits[:, :1, :]
        obj_pairs = [(act_logits, act_target), (ret_logits, ret_target)]
        if self.predict_reward:
            rew_target = encode_reward(inputs['rewards'])
            rew_logits = model_outputs['reward_logits']
            obj_pairs.append((rew_logits, rew_target))
        return obj_pairs

    def sequence_loss(
        self,
        inputs: Mapping[str, torch.Tensor],
        model_outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the loss on data wrt model outputs"""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def sequence_accuracy(
        self,
        inputs: Mapping[str, torch.Tensor],
        model_outputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the accuracy on data wrt model outputs"""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [accuracy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)
