from dataclasses import replace
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from trace_structured.config import Config
from trace_structured.data import Example
from trace_structured.model import StructuredModel


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 2
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<eos>"}

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 61 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids if not skip_special_tokens or i != 2)

    def get_vocab(self):
        return {str(i): i for i in range(64)}


def tiny_config(**changes):
    base = Config(solve_roles=2, action_dim=4, policy_width=16, role_embedding_dim=4,
                  semantic_dim=8, max_new_tokens=3, group_size=2, global_batch_size=1,
                  stage0_epochs=1, stage1_epochs=1, stage2_epochs=1,
                  stage2_samples_per_epoch=2, save_every=1, warmup_steps=0)
    return replace(base, **changes)


def tiny_model(config=None, seed=23, lora=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    llm = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, eos_token_id=2, attention_dropout=0.2))
    if lora:
        from peft import LoraConfig, get_peft_model
        llm = get_peft_model(llm, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                                          target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    return StructuredModel(llm, TinyTokenizer(), config or tiny_config())


def examples(split="train"):
    return [Example(f"{split}:0", "One plus one?", ("1+1=2", "Total is 2"), "2"),
            Example(f"{split}:1", "Two plus one?", ("2+1=3",), "3")]
