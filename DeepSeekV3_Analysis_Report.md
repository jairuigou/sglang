# DeepSeek V3 Model Analysis Report (Detailed & Corrected)

## Executive Summary

This report provides a comprehensive analysis of the DeepSeek V3 model implementation in SGLang, correcting issues found in the previous version and adding missing details about the initialization process and computation flow.

**Key Finding**: DeepSeekV3ForCausalLM is an empty subclass that inherits all functionality from DeepseekV2ForCausalLM. The model is configured differently through the model config (architecture name "DeepseekV3ForCausalLM") rather than through code differences.

## 1. Model Class Hierarchy

```
DeepseekV3ForCausalLM (empty class)
    └── DeepseekV2ForCausalLM
            ├── DeepseekV2Model
            │     ├── VocabParallelEmbedding (embed_tokens)
            │     ├── nn.ModuleList[DeepseekV2DecoderLayer] (layers)
            │     └── RMSNorm (norm)
            ├── ParallelLMHead (lm_head)
            └── LogitsProcessor (logits_processor)
```

**Location**: `python/sglang/srt/models/deepseek_v2.py:2967-2968`

```python
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass
```

## 2. Model Initialization Process

### 2.1 Top-Level Initialization (`DeepseekV2ForCausalLM.__init__`)

Located at `deepseek_v2.py:2774-2836`, the initialization process:

1. **Configuration Determination**:
   - `fuse_qkv_a_proj`: Set based on `config.q_lora_rank` existence
   - `num_fused_shared_experts`: Calculated based on hardware and config constraints (see Section 2.1.1)
   - `use_nsa`: Determined by `is_deepseek_nsa(config)` for DeepSeek V3.2

2. **Pipeline Parallelism Setup**:
   - `pp_group`: Pipeline parallel group for distributed training/inference
   - Only the last rank has the `lm_head` layer
   - Other ranks use `PPMissingLayer` as placeholder

3. **Main Model Initialization**:
   ```python
   self.model = DeepseekV2Model(
       config, quant_config, prefix=add_prefix("model", prefix)
   )
   ```

4. **LM Head Configuration**:
   - If `tie_word_embeddings`: shares weight with `embed_tokens`
   - Otherwise: Creates `ParallelLMHead` with vocabulary size × hidden size
   - Uses attention TP group when `enable_dp_lm_head` is set

5. **Lazy Weight Loading**:
   - `_routed_experts_weights_of_layer`: Lazy initialization for MoE weights
   - `capture_aux_hidden_states`: Flag for Eagle speculative decoding

6. **NSA Context Parallelism Setup**:
   - `nsa_enable_prefill_cp`: Enable prefill context parallelism
   - `cp_rank` and `cp_size`: Context parallel rank and size

7. **Attention TP Context Initialization**:
   - Initializes attention tensor parallel context with q_lora_rank and NSA flag

#### 2.1.1 Shared Experts Fusion Determination (`determine_num_fused_shared_experts`)

This method determines whether to fuse shared experts with routed experts for optimization:

**Requirements for Fusion** (all must be met):
- Architecture must be "DeepseekV3ForCausalLM"
- `n_routed_experts = 256` and `n_shared_experts = 1`
- CUDA platform with capability >= 8.0 (SM80+) or AMD with capability >= gfx942 (MI300X)
- Not using DeepEP expert parallelism (unless on AMD MI300X)
- Not using W4AFP8 quantization

If conditions are met, `num_fused_shared_experts = 1` (fusing 1 shared expert), otherwise 0.

### 2.2 DeepseekV2Model Initialization

Located at `deepseek_v2.py:2499-2628`:

1. **Embedding Layer** (`embed_tokens`):
   - Type: `VocabParallelEmbedding`
   - Shape: `[vocab_size, hidden_size]`
   - For V3: `vocab_size=102400`, `hidden_size=7168`
   - Uses attention TP group when DP attention is enabled

2. **Alternative Stream** (`alt_stream`):
   - Created for CUDA or NPU multi-stream support
   - Used for overlapping computations (e.g., shared experts with routing)

3. **Decoder Layers** (`layers`):
   - Type: `nn.ModuleList` of `DeepseekV2DecoderLayer`
   - Number: `config.num_hidden_layers` (typically 61 for V3)
   - Created via `make_layers()` utility for pipeline parallelism
   - Supports offloading for MoE layers via `offloader_kwargs`

4. **Final Normalization** (`norm`):
   - Type: `RMSNorm`
   - Only on last PP rank, placeholder on others

5. **GEMM Output Zero Allocator** (AMD optimization):
   - Computes buffer size for zero-initialized GEMM outputs
   - Used on AMD GFX95 platforms with specific model configurations

6. **A2A MoE Flag**:
   - `enable_a2a_moe`: True when using DeepEP or Mooncake backends

### 2.3 DeepseekV2DecoderLayer Initialization

Located at `deepseek_v2.py:2218-2331`:

Each decoder layer contains:

1. **Self-Attention** (`self_attn`):
   - Type: `DeepseekV2AttentionMLA` (Multi-head Latent Attention)
   - Initialized with full config including rope parameters

2. **MLP** (`mlp`):
   - Condition: Sparse layer (MoE) or Dense layer
   - Sparse condition: `layer_id >= first_k_dense_replace AND layer_id % moe_layer_freq == 0`
   - For V3: First 4 layers are dense, then MoE every 1 layer

3. **Layer Normalizations**:
   - `input_layernorm`: RMSNorm before attention
   - `post_attention_layernorm`: RMSNorm before MLP

4. **Layer Communicator** (`layer_communicator`):
   - Manages data distribution/communication for distributed inference
   - Type: `NSACPLayerCommunicator` (with NSA CP) or `LayerCommunicator`
   - Handles:
     - Attention preparation (RMSNorm, TP all-reduce)
     - MLP preparation (post-attention RMSNorm)
     - Layer postprocessing (residual connections, reduce-scatter)
     - QKV latent preparation for attention

### 2.4 DeepseekV2AttentionMLA Initialization

Located at `deepseek_v2.py:1060-1316`:

**Key Dimensions for DeepSeek V3**:
- `qk_nope_head_dim`: 128 (non-rotary portion)
- `qk_rope_head_dim`: 64 (rotary position encoding portion)
- `v_head_dim`: 128
- `q_lora_rank`: 1536 (Q latent dimension)
- `kv_lora_rank`: 512 (KV latent dimension)
- `num_heads`: 128 (total), divided by TP size per GPU
- `hidden_size`: 7168

**MLA Components**:

1. **Q Projection** (when `q_lora_rank` is not None):
   ```
   fused_qkv_a_proj_with_mqa: [hidden_size] → [q_lora_rank + kv_lora_rank + qk_rope_head_dim]
                                = [7168] → [1536 + 512 + 64] = [7168] → [2112]
   ```

2. **Q Latent Normalization**:
   ```
   q_a_layernorm: RMSNorm(q_lora_rank) = RMSNorm(1536)
   ```

3. **Q Down Projection**:
   ```
   q_b_proj: [q_lora_rank] → [num_heads * qk_head_dim]
           = [1536] → [128 * (128 + 64)] = [1536] → [24384]
   ```

4. **KV Latent Normalization**:
   ```
   kv_a_layernorm: RMSNorm(kv_lora_rank) = RMSNorm(512)
   ```

5. **KV B Projection**:
   ```
   kv_b_proj: [kv_lora_rank] → [num_heads * (qk_nope_head_dim + v_head_dim)]
            = [512] → [128 * (128 + 128)] = [512] → [32768]
   ```

6. **Output Projection**:
   ```
   o_proj: [num_heads * v_head_dim] → [hidden_size]
         = [128 * 128] → [7168] = [16384] → [7168]
   ```

7. **Weight Absorption Matrices** (initialized as None, loaded later):
   ```python
   self.w_kc = None  # [num_local_heads, kv_lora_rank, qk_nope_head_dim]
   self.w_vc = None  # [num_local_heads, kv_lora_rank, v_head_dim]
   self.w_scale = 1.0  # Scale factor for quantization
   ```

8. **Attention Modules**:
   - `attn_mqa`: RadixAttention for MQA (multi-query attention)
     - `num_kv_heads=1` (single KV head)
     - `v_head_dim=kv_lora_rank` (512)
   - `attn_mha`: RadixAttention for MHA (multi-head attention)
     - `num_kv_heads=num_local_heads`
     - `v_head_dim=v_head_dim` (128)

9. **Rotary Embedding** (`rotary_emb`):
   - Dimension: `qk_rope_head_dim` = 64
   - Base: `rope_theta` = 1000000
   - Supports DeepSeek YARN scaling for long contexts

### 2.5 DeepseekV2MoE Initialization

Located at `deepseek_v2.py:360-550`:

**MoE Components**:

1. **Gate Router** (`gate`):
   - Type: `MoEGate`
   - Weight shape: `[n_routed_experts, hidden_size] = [256, 7168]`
   - Includes `e_score_correction_bias` for load balancing

2. **Experts** (`experts`):
   - Type: `FusedMoE` (or other MoE implementation based on quantization)
   - `num_experts`: `n_routed_experts + num_fused_shared_experts` = 256 + 1 = 257
   - Each expert has:
     - `gate_proj`: [hidden_size, moe_intermediate_size] = [7168, 2048]
     - `up_proj`: [hidden_size, moe_intermediate_size] = [7168, 2048]
     - `down_proj`: [moe_intermediate_size, hidden_size] = [2048, 7168]

3. **Shared Experts** (`shared_experts`, when not fused):
   - Type: `DeepseekV2MLP`
   - `intermediate_size`: `moe_intermediate_size * n_shared_experts` = 2048 * 1
   - Disabled TP when using DeepEP/Mooncake for communication efficiency

4. **TopK Selector** (`topk`):
   - Type: `TopK`
   - `top_k`: 8 experts per token + fused shared experts
   - Uses grouped top-k for load balancing
   - Supports `norm_topk_prob` for probability normalization
   - Applies `routed_scaling_factor` for output scaling

### 2.6 DeepseekV2MLP Initialization (Dense Layers)

Located at `deepseek_v2.py:212-284`:

```
gate_up_proj: [hidden_size, 2 * intermediate_size] = [7168, 2 * 7168] = [7168, 14336]
down_proj: [intermediate_size, hidden_size] = [7168, 7168]
act_fn: SiluAndMul activation
```

## 3. Weight Loading Process

### 3.1 Weight Loading Flow (`do_load_weights`)

Located at `deepseek_common/deepseek_weight_loader.py:96`:

1. **NextN Configuration**:
   - Initialize NextN speculative decoding configuration if enabled
   - Filter weights for specific layer if loading NextN model

2. **Quantization Preprocessing**:
   - Optionally quantize weights to FP8 UE8M0 format for NVFP4 checkpoints
   - Apply to specified attention modules for GEMM efficiency

3. **Parameter Mappings**:
   - `stacked_params_mapping`: Maps split parameters (gate_proj, up_proj) to stacked (gate_up_proj)
   - `expert_params_mapping`: Maps expert-specific weights with proper naming
   - Special handling for mixed-precision models (W4AFP8)

4. **Shared Experts Fusion**:
   - If `num_fused_shared_experts > 0`, remap `mlp.shared_experts` to `mlp.experts.{n_routed_experts}`

5. **QKV Fusion**:
   - When `q_lora_rank` exists, fuse `q_a_proj` and `kv_a_proj_with_mqa` into `fused_qkv_a_proj_with_mqa`
   - Cache weights for delayed binding

### 3.2 Weight Absorption Preparation

During weight loading (`deepseek_weight_loader.py:556-610`):

1. **Split kv_b_proj**:
   ```python
   w_kc, w_vc = w.unflatten(
       0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
   ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
   ```

2. **Process and Store**:
   - For standard path: Transpose and store in `w_kc` and `w_vc`
   - For DeepGEMM: Additional block scale processing for FP8
   - For quantization: Apply weight scales and convert to bfloat16

3. **Shape**:
   - `w_kc`: `[num_local_heads, kv_lora_rank, qk_nope_head_dim]`
   - `w_vc`: `[num_local_heads, kv_lora_rank, v_head_dim]`

## 4. Forward Computation Flow

### 4.1 Top-Level Forward (`DeepseekV2ForCausalLM.forward`)

Located at `deepseek_v2.py:2890-2920`:

```
Input:
    input_ids: [num_tokens] - Token IDs
    positions: [num_tokens] - Position indices
    forward_batch: ForwardBatch - Batch metadata
    input_embeds: [num_tokens, hidden_size] or None - Pre-computed embeddings

Steps:
    1. NSA CP preparation (if enabled):
       - Check if CP split is possible
       - Prepare NSACP metadata for data parallelism

    2. Attention TP Context:
       - Enter scattered input context if needed
       - Call model.forward()

    3. Aux Hidden States:
       - Extract aux_hidden_states if capture_aux_hidden_states is True
       - Used for Eagle speculative decoding

    4. Logits Computation:
       - If last PP rank: call logits_processor()
       - Else: return hidden_states for pipeline

Output:
    logits: [num_tokens, vocab_size] or [num_tokens, 1, vocab_size] (speculative)
```

### 4.2 Model Forward (`DeepseekV2Model.forward`)

Located at `deepseek_v2.py:2632-2767`:

```
Input:
    input_ids: [num_tokens]
    positions: [num_tokens]
    forward_batch: ForwardBatch
    input_embeds: [num_tokens, hidden_size] or None

Steps:
    1. Allocator Initialization:
       - Create zero_allocator for intermediate buffers
       - Create gemm_output_zero_allocator for MoE (if needed)

    2. [First PP Rank] Token Embedding:
       if input_embeds is None:
           hidden_states = embed_tokens(input_ids)
           # Shape: [num_tokens, 7168]
       else:
           hidden_states = input_embeds
       residual = None

    3. (Optional) NSACP Data Splitting:
       - If CP enabled and first rank: split hidden_states
       - Split positions for parallel processing

    4. (Optional) Llama 4 Scaling Computation:
       - Compute scaling factors for Mistral-Large-3 support

    5. Layer Iteration (61 layers):
       - Normal layers: Standard sequential processing
       - TBO layers: Two-batch overlap for sparse layers
       
       for i in range(start_layer, end_layer):
           hidden_states, residual = layers[i](
               positions, hidden_states, forward_batch, residual,
               zero_allocator, gemm_output_zero_allocator, llama_4_scaling
           )
           # Shape remains [num_tokens, 7168]

    6. [Last PP Rank] Final Normalization:
       hidden_states, _ = norm(hidden_states, residual)
       # Shape: [num_tokens, 7168]

    7. (Optional) CP Output Reconstruction:
       - All-gather and rearrange output if CP was used

    8. (Optional) Aux Hidden States:
       - Return captured layer outputs if requested

Output:
    hidden_states: [num_tokens, 7168]
    OR (hidden_states, aux_hidden_states) if capture_aux_hidden_states
```

### 4.3 Decoder Layer Forward (`DeepseekV2DecoderLayer.forward`)

Located at `deepseek_v2.py:2340-2426`:

```
Input:
    positions: [num_tokens]
    hidden_states: [num_tokens, 7168]
    forward_batch: ForwardBatch
    residual: [num_tokens, 7168] or None
    zero_allocator: BumpAllocator
    gemm_output_zero_allocator: BumpAllocator or None
    llama_4_scaling: Optional[torch.Tensor]

Steps:
    1. Determine Quant Format:
       - Check for MXFP4 or FP8 quantization based on weight dtype

    2. Prepare Attention (via layer_communicator):
       hidden_states, residual = prepare_attn(
           hidden_states, residual, forward_batch, quant_format
       )
       # - Apply input_layernorm (in-place)
       # - TP all-reduce if needed
       # - Quantize if using MXFP4/FP8

    3. Self-Attention:
       hidden_states = self_attn(
           positions, hidden_states, forward_batch, zero_allocator, llama_4_scaling
       )
       # Shape: [num_tokens, 7168]

    4. Prepare MLP (via layer_communicator):
       hidden_states, residual = prepare_mlp(hidden_states, residual, forward_batch)
       # - Apply post_attention_layernorm (in-place)

    5. All-reduce Fusion Check:
       should_allreduce_fusion = layer_communicator.should_fuse_mlp_allreduce_with_next_layer()
       use_reduce_scatter = layer_communicator.should_use_reduce_scatter()

    6. MLP (MoE or Dense):
       hidden_states = mlp(
           hidden_states, forward_batch,
           should_allreduce_fusion, use_reduce_scatter,
           gemm_output_zero_allocator
       )
       # Shape: [num_tokens, 7168]

    7. Postprocess Layer (via layer_communicator):
       if not should_allreduce_fusion:
           hidden_states, residual = postprocess_layer(hidden_states, residual, forward_batch)
       # - Apply reduce-scatter if DP attention enabled
       # - Update residual for next layer

Output:
    hidden_states: [num_tokens, 7168]
    residual: [num_tokens, 7168]
```

### 4.4 Attention Forward (`DeepseekV2AttentionMLA.forward`)

Located at `deepseek_v2.py:1353-1368`:

```
Input:
    positions: [num_tokens]
    hidden_states: [num_tokens, 7168]
    forward_batch: ForwardBatch
    zero_allocator: BumpAllocator
    llama_4_scaling: Optional[torch.Tensor]

Steps:
    1. forward_prepare():
       - Dispatch attention backend based on forward mode and config
       - Supported backends: MHA, MHA_CHUNKED_KV, MHA_ONE_SHOT, MLA, 
         MLA_FUSED_ROPE, MLA_FUSED_ROPE_CPU, MHA_NPU, MLA_NPU, DSA_NPU
       - Call appropriate prepare method

    2. forward_core():
       - Call appropriate core method for the selected backend
       - Returns final hidden states

Output:
    hidden_states: [num_tokens, 7168]
```

### 4.5 MLA Attention Prepare (`forward_absorb_prepare`)

Located at `deepseek_v2.py:1514-1736`:

```
Input:
    positions: [num_tokens]
    hidden_states: [num_tokens, 7168]
    forward_batch: ForwardBatch
    zero_allocator: BumpAllocator
    llama_4_scaling: Optional[torch.Tensor]

Steps:
    1. Retrieve QKV Latent (from layer_communicator):
       qkv_latent = get_attn_tp_context().fetch_qkv_latent()
       # Shape: [num_tokens, 2112]

    2. Split QKV Latent:
       q, latent_cache = qkv_latent.split(
           [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=-1
       )
       q: [num_tokens, 1536]
       latent_cache: [num_tokens, 512 + 64]

    3. Extract Components:
       k_nope = latent_cache[..., :kv_lora_rank]  # [num_tokens, 512]
       k_pe = latent_cache[..., kv_lora_rank:]    # [num_tokens, 64]

    4. Q Normalization (with optional stream overlap):
       if alt_stream and capture_mode:
           # Overlap q norm and k norm on different streams
           q = q_a_layernorm(q)  # [num_tokens, 1536]
           (on alt_stream) k_nope = kv_a_layernorm(k_nope)
       else:
           if MXFP4 quantization:
               q, k_nope = fused_rms_mxfp4_quant(q, k_nope)
           elif FP8 quantization:
               q, k_nope = fused_rms_fp8_group_quant(q, k_nope)
           else:
               q = q_a_layernorm(q)
               k_nope = kv_a_layernorm(k_nope)

    5. Q Down Projection:
       q = q_b_proj(q)[0]  # [num_tokens, 24384]
       q = q.view(-1, num_local_heads, qk_head_dim)
       # Shape: [num_tokens, num_local_heads, 192]

    6. Split Q:
       q_nope = q[..., :qk_nope_head_dim]  # [num_tokens, num_local_heads, 128]
       q_pe = q[..., qk_nope_head_dim:]    # [num_tokens, num_local_heads, 64]

    7. Q Projection with Weight Absorption (w_kc):
       if use_deep_gemm_bmm:
           # FP8 grouped GEMM with masking
           q_nope_out = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(...)
       elif ROCm/AITER:
           # AITER optimized batched GEMM
           q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(...)
       elif FP8:
           q_nope_out = bmm_fp8(q_nope, w_kc, ...)
       else:
           q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
       # Shape: [num_tokens, num_local_heads, kv_lora_rank] = [num_tokens, num_local_heads, 512]

    8. Rotary Position Embedding:
       if not fused_rope and not AITER:
           q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)

    9. (Optional) NSA Indexer:
       if use_nsa:
           topk_indices = indexer(x, q_lora, positions, forward_batch, layer_id)

    10. CP KV Cache Reconstruction:
        if nsa_use_prefill_cp:
            k_nope, k_pe = rebuild_cp_kv_cache(latent_cache, forward_batch, k_nope, k_pe)

Output:
    (q_pe, k_pe, q_nope_out, k_nope, forward_batch, zero_allocator,
     positions, topk_indices, llama_4_scaling)
```

### 4.6 MLA Attention Core (`forward_absorb_core`)

Located at `deepseek_v2.py:1738-1922`:

```
Input:
    q_pe: [num_tokens, num_local_heads, 64]
    k_pe: [num_tokens, 1, 64]
    q_nope_out: [num_tokens, num_local_heads, 192]
    k_nope: [num_tokens, 1, 512]
    forward_batch: ForwardBatch
    zero_allocator: BumpAllocator
    positions: [num_tokens]
    topk_indices: Optional[Tensor]
    llama_4_scaling: Optional[Tensor]

Steps:
    1. Attention Backend Selection:
       if backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
           # Standard MLA with fused RoPE
           attn_output = attn_mqa(
               q_nope_out, k_nope, k_nope, forward_batch,
               q_rope=q_pe, k_rope=k_pe, topk_indices=topk_indices
           )
       elif AITER_GFX95:
           # AITER optimized path with fused QK rope and cache
           q, k = fused_qk_rope_cat_and_cache_mla(
               q_nope_out, q_pe, k_nope, k_pe,
               kv_buffer, cache_loc, positions, cos, sin, k_scale
           )
           attn_output = attn_mqa(q, k, k_nope, forward_batch)
       else:
           # Standard path
           q = cat([q_nope_out, q_pe], dim=-1)
           k = cat([k_nope, k_pe], dim=-1)
           if llama_4_scaling:
               q *= llama_4_scaling
           attn_output = attn_mqa(q, k, k_nope, forward_batch, save_kv_cache=True)

       # attn_output shape: [num_tokens, num_local_heads, kv_lora_rank]

    2. Reshape for V Projection:
       attn_output = attn_output.view(-1, num_local_heads, kv_lora_rank)
       # [num_tokens, num_local_heads, 512]

    3. V Projection with Weight Absorption (w_vc):
       if use_deep_gemm_bmm:
           attn_bmm_output = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
               attn_output, w_vc, ...
           )
       elif ROCm/AITER and MXFP4:
           attn_bmm_output = batched_gemm_afp4wfp4_pre_quant(...)
       elif ROCm/AITER and FP8:
           attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(...)
       elif FP8:
           attn_bmm_output = bmm_fp8(attn_output, w_vc, ...)
           attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
       else:
           if piecewise_cuda_graph:
               attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), w_vc)
                   .transpose(0, 1).flatten(1, 2)
           else:
               attn_bmm_output = torch.empty([num_tokens, num_local_heads * v_head_dim])
               torch.bmm(attn_output.transpose(0, 1), w_vc,
                   out=attn_bmm_output.view(-1, num_local_heads, v_head_dim).transpose(0, 1))
       # attn_bmm_output shape: [num_tokens, num_local_heads * 128]

    4. Output Projection:
       output, _ = o_proj(attn_bmm_output)
       # [num_tokens, 7168]

Output:
    hidden_states: [num_tokens, 7168]
```

### 4.7 MoE Forward (`DeepseekV2MoE.forward`)

Located at `deepseek_v2.py:551-582`:

```
Input:
    hidden_states: [num_tokens, 7168]
    forward_batch: Optional[ForwardBatch]
    should_allreduce_fusion: bool
    use_reduce_scatter: bool
    gemm_output_zero_allocator: BumpAllocator

Mode Selection:
    if not enable_a2a_moe:
        if alt_stream and capture_mode and num_fused_shared_experts == 0:
            return forward_normal_dual_stream(...)
        else:
            return forward_normal(...)
    else:
        return forward_deepep(...)

A. Normal Mode (forward_normal):
    1. Shared Experts (if not fused inside SBO):
       if hasattr(shared_experts):
           shared_output = _forward_shared_experts(hidden_states, gemm_output_zero_allocator)
           # [num_tokens, 7168]

    2. Router:
       router_logits = gate(hidden_states, gemm_output_zero_allocator)
       # [num_tokens, 256]

    3. TopK Selection:
       topk_output = topk(hidden_states, router_logits)
       # Contains: topk_indices, topk_weights, sorted_token_ids

    4. Experts Computation:
       final_hidden_states = experts(hidden_states, topk_output)
       # [num_tokens, 7168]

    5. Scaling and Combination:
       if not CUDA or KTEPWrapper:
           final_hidden_states *= routed_scaling_factor
       if shared_output is not None:
           final_hidden_states += shared_output
       if tp_size > 1 and not should_allreduce_fusion and not reduce_scatter:
           final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

B. Dual Stream Mode (forward_normal_dual_stream):
    - Uses alt_stream for overlapping shared experts computation with routing
    - Current stream computes router + experts
    - Alt stream computes shared experts
    - Synchronize and combine results

C. DeepEP Mode (forward_deepep):
    1. Router and TopK (same as normal mode)
    2. Dispatch (dispatch_a and dispatch_b):
       - Send tokens to expert GPUs based on topk_indices
       - Each GPU receives tokens for its assigned experts
    3. Expert Computation:
       - Each GPU computes its local experts
       - combine_input = experts.run_moe_core()
    4. Combine (combine_a and combine_b):
       - All-gather results from all expert GPUs
       - Return final_hidden_states

Output:
    final_hidden_states: [num_tokens, 7168]
```

### 4.8 Dense MLP Forward (`DeepseekV2MLP.forward`)

Located at `deepseek_v2.py:257-284`:

```
Input:
    x: [num_tokens, 7168]
    forward_batch: ForwardBatch

Steps:
    1. Gate and Up Projection:
       gate_up, _ = gate_up_proj(x)  # [num_tokens, 14336]

    2. Activation (SiLU + Mul):
       x = act_fn(gate_up)  # [num_tokens, 7168]

    3. Down Projection:
       x, _ = down_proj(x)  # [num_tokens, 7168]
       (with optional all-reduce fusion)

Output:
    x: [num_tokens, 7168]
```

## 5. Data Shape Summary

### 5.1 Input Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | [num_tokens] | Token indices |
| `positions` | [num_tokens] | Position indices |
| `input_embeds` | [num_tokens, 7168] | Pre-computed embeddings |

### 5.2 Intermediate Shapes (Per Token)

| Component | Tensor | Shape | Notes |
|-----------|--------|-------|-------|
| Embedding | `hidden_states` | [num_tokens, 7168] | After embed_tokens |
| MLA QKV latent | `qkv_latent` | [num_tokens, 2112] | After fused projection |
| Q latent | `q` | [num_tokens, 1536] | Split from latent |
| KV latent (nope) | `k_nope` | [num_tokens, 512] | Non-rotary KV |
| KV latent (rope) | `k_pe` | [num_tokens, 64] | Rotary KV |
| Q after norm | `q` | [num_tokens, 1536] | After q_a_layernorm |
| Q down projection | `q_nope_out` | [num_tokens, 24384] | After q_b_proj |
| Q reshaped | `q_nope_out` | [num_tokens, num_local_heads, 192] | View for attention |
| Q (nope) | `q_nope` | [num_tokens, num_local_heads, 128] | Non-rotary Q |
| Q (rope) | `q_pe` | [num_tokens, num_local_heads, 64] | Rotary Q |
| Q after absorption | `q_nope_out` | [num_tokens, num_local_heads, 512] | After w_kc projection |
| Attention output | `attn_output` | [num_tokens, num_local_heads, 512] | After attention |
| After V proj | `attn_bmm_output` | [num_tokens, num_local_heads * 128] | After w_vc projection |
| After O proj | `hidden_states` | [num_tokens, 7168] | After output projection |
| Router logits | `router_logits` | [num_tokens, 256] | Per-expert scores |
| MoE output | `final_hidden_states` | [num_tokens, 7168] | After experts |
| Final output | `hidden_states` | [num_tokens, 7168] | After each layer |
| Logits | `logits` | [num_tokens, 102400] | After lm_head |

### 5.3 Batch Processing

All shapes shown are per-batch. During inference:
- **Prefill**: `num_tokens` = total tokens in the batch (can be 1000s)
- **Decode**: `num_tokens` = batch size (typically 1-100s)

### 5.4 Tensor Parallelism Effects

When TP > 1:
- `num_local_heads = num_heads // tp_size` (e.g., 128 // 8 = 16 per GPU)
- Weights are sharded along the output dimension for projections
- All-reduce operations synchronize after attention and MLP

### 5.5 Weight Absorption Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| `w_kc` | [num_local_heads, kv_lora_rank, qk_nope_head_dim] | Key compression (loaded from kv_b_proj) |
| `w_vc` | [num_local_heads, kv_lora_rank, v_head_dim] | Value compression (loaded from kv_b_proj) |
| `w_scale` | scalar | Quantization scale factor |

## 6. Key Architectural Innovations

### 6.1 Multi-head Latent Attention (MLA)

MLA reduces KV cache memory by compressing keys and values into a lower-dimensional latent space:

1. **Key Compression**: `[7168] → [512]` (14x compression)
2. **Value Compression**: Values are reconstructed from latent cache via `w_vc`
3. **Q Compression**: `[7168] → [1536] → [24384]` (latent then expand to heads)

### 6.2 Weight Absorption

The weight absorption technique is critical for MLA efficiency:

**Purpose**: Eliminates the need to store full key and value tensors in the KV cache.

**Mechanism**:
- Instead of computing `Q × K^T` with full K, we pre-multiply Q with `w_kc` (absorbed key weights)
- Instead of computing `attn × V` with full V, we post-multiply attn output with `w_vc`
- This allows storing only the compressed latent representation

**Benefits**:
- ~14x reduction in KV cache memory
- No loss in model quality
- Enables longer context lengths

### 6.3 Mixture of Experts (MoE)

1. **256 Routed Experts**: Each token is routed to top-8 experts
2. **1 Shared Expert**: Applied to all tokens for stability
3. **Load Balancing**: Grouped top-k for expert distribution

### 6.4 Hybrid Dense-Sparse Architecture

- **First 4 layers**: Dense MLP (no routing)
- **Layers 4-60**: MoE (every layer)

### 6.5 Expert Parallelism (DeepEP)

For multi-GPU deployment:
- **Expert Sharding**: Different experts on different GPUs
- **Token Dispatch**: Send tokens to GPUs hosting target experts
- **All-Gather**: Combine results from all expert GPUs
- **DeepEP Library**: Optimized communication primitives for EP

### 6.6 Native Sparse Attention (NSA) - V3.2

For extremely long contexts:
- **Hierarchical Sparse Attention**: Selective attention based on importance
- **Context Parallelism**: Split long sequences across multiple GPUs
- **Indexer**: Computes attention scores for sparse token selection

## 7. Critical Data Transitions

### 7.1 Between Embedding and First Layer
```
embed_tokens(input_ids) → [num_tokens, 7168]
```

### 7.2 Between Attention and MLP in Each Layer
```
attn_output → [num_tokens, 7168] → post_attention_layernorm → mlp_input
```

### 7.3 Between Layers
```
layer_output → [num_tokens, 7168] → next_layer_input
```

### 7.4 From Model to Logits
```
final_hidden_states → [num_tokens, 7168] → lm_head → [num_tokens, vocab_size]
```

### 7.5 Attention Computation Flow
```
hidden_states [num_tokens, 7168]
    ↓ fused_qkv_a_proj_with_mqa
qkv_latent [num_tokens, 2112]
    ↓ split
q [num_tokens, 1536], latent_cache [num_tokens, 576]
    ↓ q_a_layernorm, kv_a_layernorm
q [num_tokens, 1536], k_nope [num_tokens, 512], k_pe [num_tokens, 64]
    ↓ q_b_proj + view
q_nope [num_tokens, num_heads, 128], q_pe [num_tokens, num_heads, 64]
    ↓ torch.bmm with w_kc
q_nope_out [num_tokens, num_heads, 512]
    ↓ attn_mqa
attn_output [num_tokens, num_heads, 512]
    ↓ torch.bmm with w_vc
attn_bmm_output [num_tokens, num_heads * 128]
    ↓ o_proj
hidden_states [num_tokens, 7168]
```

## 8. Configuration Parameters (DeepSeek V3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 102400 | Vocabulary size |
| `hidden_size` | 7168 | Model hidden dimension |
| `num_hidden_layers` | 61 | Number of transformer layers |
| `num_attention_heads` | 128 | Number of attention heads |
| `qk_nope_head_dim` | 128 | Non-rotary QK dimension per head |
| `qk_rope_head_dim` | 64 | Rotary QK dimension per head |
| `v_head_dim` | 128 | Value dimension per head |
| `q_lora_rank` | 1536 | Q latent compression rank |
| `kv_lora_rank` | 512 | KV latent compression rank |
| `intermediate_size` | 7168 | Dense MLP intermediate size |
| `moe_intermediate_size` | 2048 | MoE expert intermediate size |
| `n_routed_experts` | 256 | Number of routed experts |
| `n_shared_experts` | 1 | Number of shared experts |
| `num_experts_per_tok` | 8 | Top-K for expert selection |
| `first_k_dense_replace` | 4 | First MoE layer index |
| `moe_layer_freq` | 1 | Frequency of MoE layers |
| `rope_theta` | 1000000 | RoPE base frequency |
| `routed_scaling_factor` | 2.5 | Scaling factor for routed experts output |
| `norm_topk_prob` | True | Whether to normalize top-k probabilities |
| `n_group` | 8 | Number of groups for grouped top-k |
| `topk_group` | 4 | Number of groups to select from |

## 9. Corrections and Additions to Previous Report

### 9.1 Missing Details Added

1. **Weight Absorption Mechanism**: Added detailed explanation of `w_kc` and `w_vc` matrices and their role in MLA
2. **Shared Experts Fusion**: Added conditions and logic for fusing shared experts with routed experts
3. **NSA Support**: Added information about Native Sparse Attention for DeepSeek V3.2
4. **DeepEP Mode**: Added detailed explanation of expert parallelism mode
5. **Dual Stream Execution**: Added explanation of alternative CUDA stream for overlapping computations
6. **Context Parallelism**: Added details about prefill CP for long sequences
7. **Quantization Paths**: Added information about FP8, MXFP4, and INT8 quantization support
8. **Weight Loading Details**: Added explanation of packed_modules_mapping, stacked_params_mapping, and expert_params_mapping

### 9.2 Corrections

1. **MLA Attention Output Shape**: Corrected to show `kv_lora_rank` dimension (512) after attention
2. **V Projection**: Clarified that V projection uses `w_vc` matrix loaded from `kv_b_proj`
3. **Q Projection**: Added missing step of Q projection with `w_kc` before attention
4. **Layer Communicator**: Clarified its role in normalization and communication
5. **MoE Forward**: Added distinction between normal, dual-stream, and DeepEP modes

### 9.3 Clarifications

1. **Empty V3 Class**: Emphasized that DeepSeek V3 uses the same code as V2, only config differs
2. **Backend Dispatch**: Clarified that multiple attention backends are supported (MHA, MLA, etc.)
3. **Hardware Support**: Noted AMD/ROCm specific optimizations via AITER
4. **TBO (Two-Batch Overlap)**: Mentioned this optimization for sparse layers

## 10. Conclusion

DeepSeek V3 represents a highly optimized architecture for efficient large-scale inference:

1. **Memory Efficiency**: MLA with weight absorption reduces KV cache by ~14x
2. **Compute Efficiency**: MoE activates only ~3% of parameters per forward pass
3. **Communication Efficiency**: Expert parallelism (DeepEP) enables multi-GPU scaling
4. **SGLang Integration**: Supports tensor parallelism, pipeline parallelism, expert parallelism, NSA, and various quantization schemes

The `DeepseekV3ForCausalLM` class is intentionally empty, with all functionality inherited from `DeepseekV2ForCausalLM`. This demonstrates the architectural similarity between V2 and V3, with differences primarily in configuration rather than implementation.

The weight absorption technique is particularly noteworthy as it enables the dramatic KV cache reduction that makes long-context inference practical at scale.
