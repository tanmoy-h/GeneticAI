import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from typing import Optional, List, Dict, Any, Union, Tuple

from genomorph.utils.dna_utils import DNAInput
from genomorph.models.dl.processing_dl import DLProcessor
from genomorph.models.dl.chat_template_dl import CHAT_TEMPLATE
from genomorph.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
from genomorph.models.latent_reasoning import LatentReasoningMixin

register_evo2_tokenizer()


class CrossAttentionFusion(nn.Module):
    """
    STRAND-inspired cross-attention DNA-text fusion.

    DNA token embeddings act as queries; they attend over the text prompt
    embeddings (keys / values).  This produces question-aware DNA token
    representations that are injected into the LLM at <|dna_pad|> positions.

    When ``text_context`` is not supplied (e.g. during CLIP-loss computation
    that only needs DNA embeddings) the module degrades gracefully to a two-layer
    linear projection: ``out_proj(q_proj(dna_hidden))``.

    Args:
        dna_hidden_size: Dimensionality of the DNA encoder output.
        text_hidden_size: Dimensionality of the LLM embedding space.
        num_heads: Number of attention heads (default 8).
        dropout: Attention dropout rate (default 0.0).
    """

    def __init__(
        self,
        dna_hidden_size: int,
        text_hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if text_hidden_size % num_heads != 0:
            # Fall back to 1 head if the hidden size is not divisible
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = text_hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Q: project DNA token embeddings into the text attention space
        self.q_proj = nn.Linear(dna_hidden_size, text_hidden_size, bias=False)
        # K, V: text prompt embeddings (already in text space)
        self.k_proj = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.v_proj = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.out_proj = nn.Linear(text_hidden_size, text_hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dna_hidden: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dna_hidden:    ``[N, S, dna_hidden_size]`` — raw DNA encoder output.
            text_context:  ``[N, T, text_hidden_size]`` — text prompt embeddings.
                           If *None*, falls back to a linear projection.
            text_mask:     ``[N, T]`` bool, *True* = valid (non-padding) token.

        Returns:
            ``[N, S, text_hidden_size]`` — text-conditioned DNA token embeddings.
        """
        N, S, _ = dna_hidden.shape
        Q = self.q_proj(dna_hidden)  # [N, S, H]

        if text_context is None:
            # Fallback: no text query available → linear projection only
            return self.out_proj(Q)

        T = text_context.shape[1]
        K = self.k_proj(text_context)  # [N, T, H]
        V = self.v_proj(text_context)  # [N, T, H]

        # Reshape for multi-head attention
        def _split(x: torch.Tensor, L: int) -> torch.Tensor:
            return x.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)

        Q = _split(Q, S)   # [N, heads, S, head_dim]
        K = _split(K, T)   # [N, heads, T, head_dim]
        V = _split(V, T)   # [N, heads, T, head_dim]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [N, heads, S, T]

        if text_mask is not None:
            # text_mask: [N, T]; False / 0 = padding → fill with -inf
            pad = (~text_mask.bool()).unsqueeze(1).unsqueeze(2)  # [N, 1, 1, T]
            attn = attn.masked_fill(pad, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(N, S, -1)  # [N, S, H]
        return self.out_proj(out)


def get_target_modules(model):
    # Apply LoRA to all linear layers in the text model
    target_modules = []

    # Get all unique linear layer names
    seen_names = set()
    for name, module in model.text_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            target_name = names[-1]  # Use the last part of the name

            # Skip output head but include all other linear layers
            if target_name != "lm_head" and target_name not in seen_names:
                target_modules.append(target_name)
                seen_names.add(target_name)

    # Add attention-specific layers
    attention_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]
    for pattern in attention_patterns:
        if pattern not in seen_names:
            target_modules.append(pattern)

    # Return all unique layer names to apply LoRA to all layers
    return list(target_modules)


class DNALLMModel(LatentReasoningMixin, nn.Module):
    """
    A combined model that processes both DNA sequences and text inputs.

    The model uses a DNA encoder (like NucleotideTransformer) to extract features from DNA sequences
    and a text model (LLM) to process text inputs and generate responses. The DNA features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        dna_model_name: str,
        cache_dir: Optional[str] = None,
        max_length_dna: int = 2048,
        max_length_text: int = 512,
        text_model_finetune: bool = True,
        dna_model_finetune: bool = True,
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None,
        use_cross_attention: bool = False,
        use_hrpo_gate: bool = False,
        gate_hidden: int = 128,
        use_dna_gate: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize the DNALLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            dna_model_name: Name of the DNA model to be used.
            cache_dir: Directory to cache the models.
            max_length_dna: Maximum length of DNA sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 512.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            dna_model_finetune: Whether to finetune the DNA model. Defaults to True.
            dna_is_evo2: Whether the DNA model is Evo2. Defaults to False.
            dna_embedding_layer: Name of the layer to use for the Evo2 model. Defaults to None.
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer
        self.use_cross_attention = use_cross_attention
        self.use_hrpo_gate = use_hrpo_gate
        self.use_dna_gate = use_dna_gate

        # Load the text model and tokenizer
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map=device
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            trust_remote_code=True
        )
        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = CHAT_TEMPLATE
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        # Truncate from the LEFT so long sequences lose early context (system/user),
        # not the assistant response at the end (where the labels are).
        self.text_tokenizer.truncation_side = "left"

        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))


        # Load the DNA model and tokenizer
        # Skip entirely when dna_model_name is empty/None (LLM-only mode).
        if not dna_model_name:
            self.dna_model = None
            self.dna_tokenizer = None
            self.dna_config = None
            self.dna_hidden_size = 0

        elif not self.dna_is_evo2:
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_config = self.dna_model.config

        else:
            from evo2 import Evo2
            evo2_short_name = dna_model_name.split("/")[-1]
            self.dna_model = Evo2(evo2_short_name)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = self.dna_model.model.config
            self.dna_embedding_layer = self.dna_embedding_layer

        # Get model dimensions
        self.text_hidden_size = self.text_config.hidden_size
        if self.dna_config is not None:
            self.dna_hidden_size = self.dna_config.hidden_size

        # DNA projection: cross-attention fusion or plain linear depending on flag.
        # Kept as `dna_projection` so existing save/load utilities work unchanged.
        # When dna_model_name is empty, create a zero-param stub so downstream
        # code that references dna_projection doesn't crash.
        if self.dna_hidden_size > 0:
            if self.use_cross_attention:
                self.dna_projection = CrossAttentionFusion(
                    dna_hidden_size=self.dna_hidden_size,
                    text_hidden_size=self.text_hidden_size,
                )
            else:
                self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)
        else:
            self.dna_projection = nn.Identity()  # never called in LLM-only mode

        # HRPO gate: adaptive latent reasoning depth (Week 3+4)
        # use_dna_gate=True: GateNet sees [h; u_dna] for DNA-conditioned gating.
        if use_hrpo_gate:
            from genomorph.models.latent_reasoning import GateNet
            gate_dna_size = self.dna_hidden_size if use_dna_gate else 0
            self.gate_net = GateNet(
                hidden_size=self.text_hidden_size,
                gate_hidden=gate_hidden,
                dna_size=gate_dna_size,
            )

        # Create processor for handling inputs
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)

    
    def _evo2_embed(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Extract embeddings from a specific Evo2 layer.

        Tries the original ``return_embeddings=True`` wrapper API first (matches
        benchmark training conditions from April 2026).  Falls back to a forward
        hook on ``self.dna_model.model`` if the wrapper raises (StripedHyena
        does not support ``output_hidden_states=True`` in some configurations).

        Args:
            input_ids:  ``[1, seq_len]`` tensor of DNA token IDs.
            layer_name: Module name, e.g. ``"blocks.28.mlp.l3"``.

        Returns:
            ``[seq_len, hidden_size]`` tensor of activations (detached).
        """
        # --- Primary path: wrapper API (benchmark-compatible) ---
        if not getattr(self, "_evo2_api_checked", False):
            self._evo2_api_checked = True
            try:
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[layer_name],
                )
                print(f"[Evo2] return_embeddings=True OK — using wrapper API")
                return embeddings[layer_name].squeeze(0).detach()
            except Exception as e:
                print(f"[Evo2] return_embeddings=True failed ({e}) — falling back to hook")
        else:
            try:
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[layer_name],
                )
                return embeddings[layer_name].squeeze(0).detach()
            except Exception:
                pass  # fall through to hook

        # --- Fallback path: forward hook on inner model ---
        captured: Dict[str, torch.Tensor] = {}

        def _hook(module, inp, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        named = dict(self.dna_model.model.named_modules())
        if layer_name not in named:
            raise ValueError(f"Evo2 layer '{layer_name}' not found.")

        dna_device = next(self.dna_model.model.parameters()).device
        input_ids = input_ids.to(dna_device)

        handle = named[layer_name].register_forward_hook(_hook)
        try:
            self.dna_model.model(input_ids)
        finally:
            handle.remove()

        return captured["h"].squeeze(0)

    @torch.no_grad()
    def get_dna_summary(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> torch.Tensor:
        """
        Returns a mean-pooled DNA encoder summary vector for each batch item.

        Used to condition the HRPO GateNet on DNA content (Week 4).

        Args:
            dna_tokenized:  Tokenized DNA sequences (input_ids, attention_mask).
            batch_idx_map:  Mapping from each DNA sequence to its batch index.
            batch_size:     Number of examples in the batch.

        Returns:
            u_dna: [batch_size, dna_hidden_size]  mean-pooled DNA embeddings.
        """
        input_ids     = dna_tokenized["input_ids"]
        attention_mask = dna_tokenized["attention_mask"]

        if self.dna_is_evo2:
            # Use specified layer or last layer by default
            evo2_layer = self.dna_embedding_layer or list(
                self.dna_model.model.layers.keys())[-1]
            hidden_list = []
            for i in range(input_ids.shape[0]):
                hidden_list.append(self._evo2_embed(input_ids[i:i+1], evo2_layer))
            hidden = torch.stack(hidden_list)                           # [N, L, H]
        else:
            outputs = self.dna_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]                          # [N, L, H]

        # Mean-pool over sequence length (masked)
        mask = attention_mask.unsqueeze(-1).float()                    # [N, L, 1]
        u_per_seq = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1) # [N, H]

        # Aggregate per batch item
        u_dna  = torch.zeros(batch_size, self.dna_hidden_size,
                             device=u_per_seq.device, dtype=u_per_seq.dtype)
        counts = torch.zeros(batch_size, device=u_per_seq.device)
        for seq_i, batch_i in enumerate(batch_idx_map):
            u_dna[batch_i]  += u_per_seq[seq_i]
            counts[batch_i] += 1
        u_dna = u_dna / counts.clamp(min=1).unsqueeze(-1)

        return u_dna  # [batch_size, dna_hidden_size]

    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
        text_context: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process DNA sequences to obtain embeddings.

        Args:
            dna_tokenized:  Tokenized DNA sequences.
            batch_idx_map:  Mapping of each sequence to its batch item
                            (values in ``[0, 2*batch_size)``).
            batch_size:     Number of items in the batch.
            text_context:   Optional ``[batch_size, T, text_hidden_size]`` tensor of
                            text prompt embeddings.  When provided, each DNA sequence
                            attends over the prompt of its batch item, producing
                            question-aware DNA token representations.
            text_mask:      Optional ``[batch_size, T]`` bool mask for ``text_context``
                            (True = valid token).

        Returns:
            List of ``[seq_len, text_hidden_size]`` tensors, one per slot in
            ``[0, 2*batch_size)``.
        """
        # Get the device of the DNA model
        # Handle Evo2 wrapper vs HuggingFace models
        if self.dna_is_evo2:
            dna_device = next(self.dna_model.model.parameters()).device
        else:
            dna_device = next(self.dna_model.parameters()).device
        
        # Move DNA tokenized inputs to the same device as the DNA model
        dna_tokenized = {
            k: v.to(dna_device) if isinstance(v, torch.Tensor) else v
            for k, v in dna_tokenized.items()
        }
        
        # Process all sequences to get DNA representations
        with torch.no_grad():
            # Handle different model types based on dna_is_evo2 attribute
            if self.dna_is_evo2:  # Evo2 model
                # Get embeddings from the specific layer in Evo2
                # Fall back to last layer if dna_embedding_layer not specified
                evo2_layer = self.dna_embedding_layer or list(
                    self.dna_model.model.layers.keys())[-1]
                hidden_states_list = []

                for seq_idx in range(len(dna_tokenized["input_ids"])):
                    seq_ids  = dna_tokenized["input_ids"][seq_idx:seq_idx+1]       # [1, L]
                    seq_mask = dna_tokenized["attention_mask"][seq_idx]             # [L]
                    actual_len = int(seq_mask.sum().item())
                    if actual_len < seq_ids.shape[1]:
                        seq_ids = seq_ids[:, :actual_len]                           # strip batch pad
                    seq_embeddings = self._evo2_embed(seq_ids, evo2_layer)
                    if torch.isnan(seq_embeddings).any() or torch.isinf(seq_embeddings).any():
                        print(f"[DEBUG] Evo2 NaN/Inf in seq {seq_idx}: min={seq_embeddings.min().item():.3f} max={seq_embeddings.max().item():.3f} dtype={seq_embeddings.dtype}")
                    hidden_states_list.append(seq_embeddings)
                
                # Stack to get same format as non-Evo2 output.
                # Sequences may have different L after padding trim — pad to batch max.
                if hidden_states_list:
                    max_L = max(e.shape[0] for e in hidden_states_list)
                    hidden_states_list = [
                        torch.nn.functional.pad(e, (0, 0, 0, max_L - e.shape[0]))
                        if e.shape[0] < max_L else e
                        for e in hidden_states_list
                    ]
                    hidden_states = torch.stack(hidden_states_list)
                else:
                    # Return empty tensors on the correct device
                    _p = next(self.dna_projection.parameters())
                    return [torch.zeros((0, self.text_hidden_size),
                                       device=_p.device, dtype=_p.dtype)
                            for _ in range(2 * batch_size)]
                    
            else:  # Standard HuggingFace model
                # Use existing code path for HF models
                outputs = self.dna_model(
                    input_ids=dna_tokenized["input_ids"],
                    attention_mask=dna_tokenized["attention_mask"],
                    output_hidden_states=True,
                )
                # Get the last hidden state
                hidden_states = outputs.hidden_states[-1]  # shape: [n_seqs, seq_len, hidden_dim]

        # Move hidden states to the fusion module's device/dtype
        _p = next(self.dna_projection.parameters())
        hidden_states = hidden_states.to(device=_p.device, dtype=_p.dtype)
        n_seqs = hidden_states.shape[0]

        # ── Cross-attention fusion ───────────────────────────────────────────
        # If text_context is provided, each DNA sequence attends over the text
        # embeddings of its corresponding batch item.
        if self.use_cross_attention and text_context is not None:
            # batch_idx_map values are in [0, 2*batch_size); map to [0, batch_size)
            ctx_indices = [batch_idx_map[i] % batch_size for i in range(n_seqs)]
            text_ctx = text_context[ctx_indices].to(device=_p.device, dtype=_p.dtype)
            text_msk = (
                text_mask[ctx_indices].to(device=_p.device)
                if text_mask is not None else None
            )
            projected_states = self.dna_projection(hidden_states, text_ctx, text_msk)
        else:
            # Linear projection only (cross-attention disabled or no text context)
            projected_states = self.dna_projection(hidden_states)
        if torch.isnan(projected_states).any() or torch.isinf(projected_states).any():
            print(f"[DEBUG] Projection NaN/Inf: min={projected_states.min().item():.3f} max={projected_states.max().item():.3f} dtype={projected_states.dtype}")

        # Group embeddings by batch item
        result = [[] for _ in range(2 * batch_size)]

        # For each sequence, get its embeddings and add to appropriate batch result
        for seq_idx, batch_idx in enumerate(batch_idx_map):
            # Get only the valid (non-padding) tokens
            valid_length = dna_tokenized["attention_mask"][seq_idx].sum().item()
            seq_embedding = projected_states[seq_idx, :valid_length]
            result[batch_idx].append(seq_embedding)

        # Concatenate embeddings for each batch item
        for i in range(2 * batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                result[i] = torch.zeros((0, self.text_hidden_size),
                                        device=_p.device, dtype=_p.dtype)

        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text based on DNA and text inputs.

        Args:
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            labels: Labels for supervised fine-tuning (used if provided directly)
            **kwargs: Additional arguments for generation

        Returns:
            Outputs from the text model
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        embed_layer = self.text_model.get_input_embeddings()
        text_inputs_embeds = embed_layer(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            # Process DNA sequences; pass text embeddings so cross-attention can
            # condition each DNA token on the corresponding text prompt.
            batch_dna_embeds = self.process_dna_embeddings(
                dna_tokenized, batch_idx_map, batch_size,
                text_context=text_inputs_embeds,
                text_mask=attention_mask,
            )

            mask = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)

            # Ensure DNA embeddings have the same dtype and device as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Handle labels if provided (for training)
        if labels is not None:
            # TODO: Implement this
            pass

        # Forward pass through the text model (loss is computed if labels is provided)
        outputs = self.text_model(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on DNA and text inputs.

        Args:
            inputs: The preprocessed inputs from the processor (preferred method)
            batch_dna_sequences: List of lists of DNA sequences per batch item (legacy method)
            input_texts: List of input texts (legacy method)
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs which can be decoded using the processor
        """
        text_inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map
        )

        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        # Generation parameters may need adjustment based on model type
        # do not set use_cache = True, things break
        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        return outputs

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.text_model, "is_gradient_checkpointing", False)

    def gradient_checkpointing_disable(self):
        self.text_model.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):

        self.text_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        use_reentrant = (
            gradient_checkpointing_kwargs["use_reentrant"]
        )

        print("use_reentrant:", use_reentrant)

        if use_reentrant:
            self.text_model.enable_input_require_grads()
        
        print("gradient_checkpointing_enable for model:", self.text_model.is_gradient_checkpointing)
    
    def train(self, mode: bool = True):
        nn.Module.train(self, False)
        self.text_model.train(mode)
        self.dna_projection.train(mode)

        if hasattr(self, "lm_head"):
            self.lm_head.train(mode)
        self.training = self.text_model.training
        return self

    def get_prompt_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None
    ):
        """
        Get prompt embeddings for the model.
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")
    
        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            dna_tokenized = {k: v.to(self.device) for k, v in dna_tokenized.items()}
            batch_dna_embeds = self.process_dna_embeddings(
                dna_tokenized, batch_idx_map, batch_size,
                text_context=text_inputs_embeds,
                text_mask=attention_mask,
            )

            mask = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)

            # Ensure DNA embeddings have the same dtype and device as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device)
            text_inputs_embeds[mask] = dna_embeds_flat
        
        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        return text_inputs_embeds, attention_mask