import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupLinearLayer(nn.Module):
    def __init__(
            self,
            in_dim,  # 2560
            out_dim,  # 5120
            num_blocks,  # 1
            bias=True,
            a=None,
    ) -> None:
        super().__init__()
        # self.num_blocks = num_blocks
        # self.out_dim = out_dim
        if a is None:
            a = 1. / math.sqrt(out_dim)
        # self.weight = nn.Parameter(torch.FloatTensor(
        #     num_blocks, in_dim, out_dim).uniform_(-a, a))
        # self.bias = bias
        # if self.bias:
        #     self.bias = nn.Parameter(torch.FloatTensor(
        #         num_blocks, out_dim).uniform_(-a, a)).unsqueeze(1)
        # else:
        #     self.bias = None
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.data.uniform_(-a, a)
        self.linear.bias.data.uniform_(-a, a)

    def forward(self, x):
        x = self.linear(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=300) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pe = torch.zeros(max_seq_len, hidden_dim)
        for pos in range(max_seq_len):
            for i in range(0, hidden_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/hidden_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/hidden_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.pos_embed_weight = nn.Parameter(torch.ones_like(pe))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        seq_len = x.size(1)

        # (bs,pos,nhid) * (bs, nhid, pos) = (bs, pos, nhid)
        pe_use = self.pe[:, :seq_len] * \
            F.sigmoid(self.pos_embed_weight[:, :seq_len])
        x = x+pe_use
        x = x.permute(1, 0, 2)
        return x


class RepeatLinear(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_steps,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.w = nn.Parameter(torch.randn(in_dim).cuda())
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        w = self.w.unsqueeze(0).repeat(self.num_steps, 1)
        w = self.w.unsqueeze(0).repeat(x.size(0), 1, 1)

        x = torch.relu(w * x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)

        return x


class RelationalMemory(nn.Module):
    def __init__(
            self,
            mem_slots,
            head_size,
            attn_drop: float = 0.9,
            num_heads: int = 1,
            num_blocks: int = 1,
            forget_bias=1.,
            input_bias=0.,
            gate_style="unit",
            attention_mlp_layers=2,
            return_all_outputs=False,
            use_topk=False,
            topk: int = 3,
            num_steps: int = 5,
            null_attention=False,
    ) -> None:
        super().__init__()

        self.mem_slots = mem_slots
        self.head_size = head_size
        self.n_heads = num_heads
        self.use_topk = use_topk
        self.topk = topk
        self.attn_drop = nn.Dropout(attn_drop)

        assert num_blocks >= 1, (f"num blocks mush be >= 1. Got: {num_blocks}")

        self.num_blocks = num_blocks
        self.gate_style = gate_style

        self.num_atten_mlp_layers = attention_mlp_layers

        # value size is same as head_size
        # self.value_size = self.head_size
        # total size for query-key-value

        # self.query_proj = nn.Linear(
        #     self.mem_size, self.key_size*self.n_heads)

        self.query_proj = nn.Linear(self.head_size, self.head_size)
        count_parameters(self.query_proj, "query")
        # self.key_proj = nn.Linear(self.mem_size, self.key_size*self.n_heads)
        self.key_proj = nn.Linear(self.head_size, self.head_size)
        count_parameters(self.key_proj, "key")
        # self.value_proj = nn.Linear(self.mem_size, self.key_size*self.n_heads)
        self.value_proj = nn.Linear(self.head_size, self.head_size)
        count_parameters(self.value_proj, "value")

        self.attention_mlp = nn.ModuleList(
            [nn.Linear(self.head_size, self.head_size)]*self.num_atten_mlp_layers)
        count_parameters(self.attention_mlp[0], "attention_mlp")
        self.attended_memory_layernorm = nn.LayerNorm(self.head_size)
        count_parameters(self.attended_memory_layernorm, "layer_norm1")
        self.attended_memory_layernorm2 = nn.LayerNorm(self.head_size)
        count_parameters(self.attended_memory_layernorm2, "layer_norm2")

        # params for initial embedding function
        self.input_projector = nn.Linear(self.head_size, self.head_size)
        count_parameters(self.input_projector, "input_projector")

        # params for gating
        self.num_gates = 2 * self.calculate_gate_size()
        print("input projector:"+str(self.head_size))
        if gate_style in ['unit', 'memory']:
            self.input_gate_projector = RepeatLinear(
                in_dim=self.head_size, out_dim=self.num_gates, num_steps=num_steps)
            count_parameters(self.input_gate_projector, "input_gate_projector")
            self.memory_gate_projector = GroupLinearLayer(
                in_dim=self.head_size, out_dim=self.num_gates, num_blocks=self.num_blocks)
            count_parameters(self.memory_gate_projector,
                             "memory_gate_projector")

        self.forget_bias = nn.Parameter(
            torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(
            torch.tensor(input_bias, dtype=torch.float32))

        # number of outputs returned
        self.return_all_outputs = return_all_outputs
        self.null_attention = null_attention

    def initial_state(self, batch_size):
        """Create an initial memory"""
        init_state = torch.stack([torch.eye(self.mem_slots)
                                 for _ in range(batch_size)])
        # pad the matrix with zeros
        if self.head_size > self.mem_slots:
            difference = self.head_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)
        elif self.head_size < self.mem_slots:
            init_state = init_state[:, :, :self.head_size]

        return init_state 

    def multi_head_attention(self, ipts, memory, use_topk_=True):
        """Perform multi-head attention"""
        q = self.query_proj(memory)
        k = self.key_proj(ipts)
        v = self.value_proj(ipts)

        B, T, C = q.size()

        q = q.reshape(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.n_heads, -1).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.n_heads, -1).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * \
            (1.0/math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)

        if not self.null_attention:
            if self.use_topk and use_topk_:
                topk = torch.topk(att, dim=-1, k=self.topk)
                mask = torch.zeros_like(att).to(att.device)
                mask.scatter_(3, topk.indices, 1)
                att = att * mask
        else:
            raise NotImplementedError

        output = att @ v
        # output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = output.transpose(
            1, 2).contiguous().view(output.size(0), T, -1)
        return output

    def attend_over_memory(self, inputs, memory):
        """
        Perform multi-head attention over memory
        """
        # memory = memory.view_as(inputs)
        for _ in range(self.num_blocks):
            attended_memory = self.multi_head_attention(inputs, memory)
            memory = self.attended_memory_layernorm(memory + attended_memory)

            attention_mlp = memory
            for i, _ in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
        return memory

    def forward_step(self, ipts, memory, treat_input_as_mtx=False):
        """Forward step of the relational memory core"""
        if treat_input_as_mtx:
            # keep (Batch, Seq, ...) dim (0, 1), flatten starting from dim 2
            ipts = ipts.view(ipts.shape[0], ipts.shape[1], -1)
            inputs_reshape = self.input_projector(ipts)
        else:
            ipts = ipts.view(ipts.shape[0], -1)
            ipts = self.input_projector(ipts)
            # unsqueeze the time step to dim 1
            inputs_reshape = ipts.unsqueeze(dim=1)

        next_memory = self.attend_over_memory(inputs_reshape, memory)

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory
            # self.attn_log[:, :, 1] = input_gate[0].cpu()

        # output = next_memory.reshape(next_memory.shape[0], -1)
        hx = self.multi_head_attention(
            next_memory, inputs_reshape, use_topk_=False)

        return next_memory, hx

    def forward(self, ipts, memory, parallel=True):
        memory, hx = self.forward_step(ipts, memory, True)

        return memory, hx

    def calculate_gate_size(self):
        if self.gate_style == "unit":
            return self.head_size
        elif self.gate_style == "memory":
            return 1
        else:
            return 0

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using inputs and memory
        """
        memory = torch.tanh(memory)
        if len(inputs.shape) == 3:
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(1)
            gate_memory = self.memory_gate_projector(
                memory) 
        else:
            raise ValueError(
                f"input shape of create_gate function is {inputs.shape}, expects 3")

        gates = gate_memory + gate_inputs
        #self.attn_log = gates[0]
        gates = torch.split(gates, split_size_or_sections=int(
            gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        # to be used for equation 7
        self.attn_log = torch.zeros(
            input_gate.shape[1], input_gate.shape[2], 2)
        self.attn_log[:, :, 0] = input_gate[0].cpu()

        input_gate = torch.sigmoid(input_gate+self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    @classmethod
    def from_linear(cls, layer, n_head):
        n_embd, fan_in = layer.weight.shape
        return cls(
            mem_slots=1290,  
            head_size=n_embd,
            num_heads=n_head,
            num_blocks=64,
            forget_bias=1,
            input_bias=0,
            gate_style="unit",
            attention_mlp_layers=1,
            return_all_outputs=False,
        )


def count_parameters(model, name):
    k = 0
    for p in model.parameters():
        k += p.numel()
    print(name, end=':')
    print(k)
