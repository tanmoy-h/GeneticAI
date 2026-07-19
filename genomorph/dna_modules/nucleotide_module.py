from typing import Dict, Any, Union, List, Optional, Callable, Type
from trl.data_utils import maybe_apply_chat_template
import torch

import re

from genomorph.dna_modules.dna_module import DNABaseModule
from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.dl.processing_dl import DLProcessor


class NucleotideDNAModule(DNABaseModule):
    """
    DNA module implementation for NucleotideTransformer-based models.

    This module provides the interface between DNA-LLM models and the training
    infrastructure, handling model loading, processing setup, and reward functions.
    """

    def __init__(self):
        """Initialize the NucleotideDNAModule."""
        super().__init__()

    def post_model_init(self, model: Any, processing_class: Any) -> None:
        """
        Perform any post-initialization setup on the model.

        Args:
            model: The initialized model
            processing_class: The processor for the model
        """
        # No post-init needed for this implementation
        pass

    def get_processing_class(self) -> Type:
        """
        Get the processing class to use with this DNA-LLM model.

        Returns:
            The processing class
        """
        return DLProcessor

    def get_dnallm_modules_keywords(self) -> List[str]:
        """
        Get keywords to identify DNA-specific modules in the model.

        Used to exclude DNA modules from LoRA adaptation during training.

        Returns:
            List of keywords that identify DNA modules
        """
        return ["dna"]

    def get_custom_multimodal_keywords(self) -> List[str]:
        """
        Get keywords for multimodal inputs that should be passed to the model.

        Returns:
            List of input keywords for multimodal processing
        """
        return ["dna_tokenized", "batch_idx_map"]

    def get_non_generate_params(self) -> List[str]:
        """
        Get parameter names that should be excluded from generation.

        Returns:
            List of parameter names to exclude from generation calls
        """
        return ['answer', 'prompt']

    def get_custom_processing_keywords(self) -> List[tuple]:
        """
        Get custom processing keywords for the processor.

        Returns:
            List of (component, parameter) tuples for custom processing
        """
        return [("dna_tokenizer", "max_length")]

    def prepare_prompt(
        self, processing_class: Any, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> List[str]:
        """
        Prepare prompts from input examples.

        Args:
            processing_class: The processor to use
            inputs: List of input examples

        Returns:
            List of prepared prompts
        """
        prompts_text = [
            maybe_apply_chat_template(example, processing_class)["prompt"]
            for example in inputs
        ]
        return prompts_text

    def prepare_model_inputs(
        self,
        processing_class: Any,
        model: Any,
        prompts_text: List[str],
        batch_dna_sequences: List[List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        padding_side: str = "left",
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the model.

        Args:
            processing_class: The processor to use
            model: The model to prepare inputs for
            prompts_text: List of text prompts
            batch_dna_sequences: List of lists of DNA sequences
            return_tensors: Return format for tensors
            padding: Whether to pad inputs
            padding_side: Side to pad on
            add_special_tokens: Whether to add special tokens

        Returns:
            Processed inputs for the model
        """
        # Handle DataParallel wrapped models by accessing the module attribute if needed
        max_length_text = model.max_length_text if not hasattr(model, 'module') else model.module.max_length_text
        max_length_dna = model.max_length_dna if not hasattr(model, 'module') else model.module.max_length_dna
        
        prompt_inputs = processing_class(
            text=prompts_text,
            batch_dna_sequences=batch_dna_sequences,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            max_length_text=max_length_text,
            max_length_dna=max_length_dna,
        )

        return prompt_inputs

    def is_embeds_input(self) -> bool:
        """
        Whether the model uses embeddings as input (instead of token IDs).

        Returns:
            Boolean indicating if the model takes embedding inputs
        """
        return True

    @staticmethod
    def get_question_template() -> str:
        """
        Get the template for formatting questions.

        Returns:
            String template for questions
        """
        return "{Question}"

    @staticmethod
    def _extract_xml_answer(text: str) -> str:
        # Primary: scan each chunk after a </think> for explicit "Answer:" format.
        # Model must write "Answer: X" after </think>; anything else gets 0 reward,
        # which trains it to follow the format.
        for part in text.split("</think>")[1:]:
            if "Answer:" in part:
                answer = re.split(r'[Aa]nswer:\s*', part, maxsplit=1)[-1]
                answer = answer.split('\n')[0]
                answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                if answer:
                    return answer
        # Fallback: Answer: written inside <think> without closing it
        m = re.search(r'[Aa]nswer:\s*(.+?)(?:<\|im_end\|>|<\|endoftext\|>|\n|$)', text)
        if m:
            return m.group(1).strip()
        return ""

    # Reward functions
    @staticmethod
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [NucleotideDNAModule._extract_xml_answer(r) for r in responses]
        # extracted_responses = [r.lower().replace("answer:", "").strip() for r in extracted_responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [3.0 if a.lower() in r.lower() else 0.0 for r, a in zip(extracted_responses, answer)]

    @staticmethod
    def completion_quality_reward_func(completions, **kwargs) -> List[float]:
        """Merges concise + natural_eos.
        +0.5  natural <|im_end|> emitted (not truncated at max_length)
        +0.5  answer ≤ 10 words   |  0.0  11-50 words  |  -0.5  > 50 words"""
        rewards = []
        for comp in completions:
            text = comp[0]['content']
            score = 0.5 if "<|im_end|>" in text else 0.0
            answer = NucleotideDNAModule._extract_xml_answer(text).replace("<|im_end|>", "").strip()
            n = len(answer.split())
            if n == 0 or n > 50:
                score -= 0.5
            elif n <= 10:
                score += 0.5
            rewards.append(score)
        return rewards

    @staticmethod
    def length_penalty_reward_func(completions, **kwargs) -> List[float]:
        """Penalise long TOTAL completions (reasoning + latents), not just the answer.

        completion_quality/concise only score the extracted answer length; nothing
        bounds the reasoning/latent bloat that drives inference time. This adds a
        smooth penalty on the full completion's word count:
            0.0        for n <= FREE words   (a normal chain-of-thought is free)
            0 .. -0.5  linearly between FREE and FLOOR
            -0.5       for n >= FLOOR words
        FREE/FLOOR are in WORDS. Run 055211 sat at ~500 completion tokens
        (~375 words) with time regressing, so FREE=250 actively rewards trimming
        while still leaving room for genuine reasoning. Tune if the observed
        completions/mean_length shifts materially.
        """
        FREE  = 250
        FLOOR = 500
        rewards = []
        for comp in completions:
            text = comp[0]["content"]
            n = len(text.split())
            if n <= FREE:
                rewards.append(0.0)
            elif n >= FLOOR:
                rewards.append(-0.5)
            else:
                rewards.append(-0.5 * (n - FREE) / (FLOOR - FREE))
        return rewards

    @staticmethod
    def reasoning_quality_reward_func(completions, **kwargs) -> List[float]:
        """Merges diversity + non_degenerate.
        Hard -0.5 for: repetition loops (same sentence ≥3×), or < 15 unique thinking words.
        Otherwise: -0.5 × mean pairwise Jaccard similarity of thinking-block word sets."""
        from collections import Counter

        def _think_words(text: str):
            end = text.find("</think>")
            block = text[:end] if end != -1 else text
            return set(block.lower().split())

        def _has_loop(text: str) -> bool:
            end = text.find("</think>")
            block = text[:end] if end != -1 else text
            lines = [l.strip() for l in re.split(r'\n|(?<=[.!?])\s+', block) if len(l.strip()) > 15]
            return any(v >= 3 for v in Counter(lines).values())

        responses = [comp[0]['content'] for comp in completions]
        word_sets = [_think_words(r) for r in responses]

        rewards = []
        for i, (r, s) in enumerate(zip(responses, word_sets)):
            if not s or len(s) < 15 or _has_loop(r):
                rewards.append(-0.5)
                continue
            sims = []
            for j, t in enumerate(word_sets):
                if i != j and t and len(t) >= 15:
                    union = len(s | t)
                    sims.append(len(s & t) / union if union else 0.0)
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            rewards.append(-0.5 * avg_sim)
        return rewards

    @staticmethod
    def diversity_reward_func(completions, **kwargs) -> List[float]:
        """Penalise mode collapse in reasoning using pairwise Jaccard similarity
        of thinking-block word sets. Measures reasoning diversity, not answer
        diversity, so correct groups that all name the same disease are not penalised.
        All-identical reasoning → -0.5 for all.  All-unique reasoning → 0.0 for all."""
        responses = [completion[0]['content'] for completion in completions]

        def _think_words(text: str):
            end = text.find("</think>")
            block = text[:end] if end != -1 else text
            return set(block.lower().split())

        word_sets = [_think_words(r) for r in responses]
        n = len(word_sets)
        if n <= 1:
            return [0.0] * n

        rewards = []
        for i, s in enumerate(word_sets):
            if not s:
                rewards.append(-0.5)
                continue
            # Degenerate outputs (repetitive loops) have tiny unique-word vocabularies.
            # A real reasoning block on a gene/disease question should have ≥15 unique words.
            # Penalise hard instead of giving a near-0 "unique" diversity bonus.
            if len(s) < 15:
                rewards.append(-0.5)
                continue
            sims = []
            for j, t in enumerate(word_sets):
                if i != j and t:
                    union = len(s | t)
                    sims.append(len(s & t) / union if union else 0.0)
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            rewards.append(-0.5 * avg_sim)
        return rewards

    @staticmethod
    def concise_reward_func(completions, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [NucleotideDNAModule._extract_xml_answer(r) for r in responses]
        def _score(r: str) -> float:
            tokens = r.split()
            if len(tokens) > 50:
                return -0.5
            return 0.5 if len(tokens) <= 10 else 0.0
        return [_score(r) for r in extracted_responses]

    @staticmethod
    def strict_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # Stricter than soft_format: Answer: must be on its own line immediately after
        # </think> and the answer must be 5–50 characters (concise but non-trivial).
        pattern = r"</think>\n+Answer:\s+[^\n]{5,50}(?:\n|<\|im_end\|>|$)"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.1 if match else -0.1 for match in matches]

    @staticmethod
    def soft_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<think>.*?</think>\s*Answer:\s*\S"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.1 if match else -0.1 for match in matches]

    @staticmethod
    def latent_format_reward_func(completions, **kwargs) -> List[float]:
        """Penalise completions where <start-latent> is not matched by <end-latent>.
        Each unmatched <start-latent> (i.e. a block cut short by </think>) scores -0.5.
        Completions with no latent markers score 0.0 (neutral).
        """
        rewards = []
        for comp in completions:
            text = comp[0]["content"]
            n_start   = text.count("<start-latent>")
            n_end     = text.count("<end-latent>")
            unmatched = max(0, n_start - n_end)
            rewards.append(-0.5 * unmatched)
        return rewards

    @staticmethod
    def xmlcount_reward_func(completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [NucleotideDNAModule._count_xml(c) for c in contents]

    @staticmethod
    def non_degenerate_reward_func(completions, **kwargs) -> List[float]:
        """Penalise completions that repeat the same sentence 3+ times in the thinking block.
        Targets repetition loops ('Step 1: ...' × 12, 'SOD1 mutation likely disrupts...' × 20).
        Does not penalise disease names appearing multiple times in different sentences."""
        from collections import Counter
        rewards = []
        for comp in completions:
            text = comp[0]['content']
            end = text.find("</think>")
            block = text[:end] if end != -1 else text
            lines = [l.strip() for l in re.split(r'\n|(?<=[.!?])\s+', block) if len(l.strip()) > 15]
            counts = Counter(lines)
            if any(v >= 3 for v in counts.values()):
                rewards.append(-0.5)
            else:
                rewards.append(0.0)
        return rewards

    @staticmethod
    def single_think_close_reward_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = completion[0]["content"]
            n_open  = text.count("<think>")
            n_close = text.count("</think>")
            rewards.append(0.1 if (n_open == 1 and n_close == 1) else -0.1)
        return rewards

    @staticmethod
    def _count_xml(text) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        after_think = text.split("</think>", 1)[1] if "</think>" in text else ""
        if after_think.strip().lower().startswith("answer:"):
            count += 0.125
        return count

    @staticmethod
    def format_reward_func(completions, **kwargs) -> List[float]:
        """Unified structural-format reward — replaces xmlcount + soft_format +
        single_think_close, which each rewarded the same <think>…</think>…Answer:
        structure. Every structural property is scored exactly once here.

        Additive (max +0.5; floor -0.2 when completely malformed):
          +0.15  exactly one <think>
          +0.15  exactly one newline-wrapped </think>  (\\n</think>\\n)
          +0.20  non-empty 'Answer:' immediately after </think>
        """
        rewards = []
        for comp in completions:
            text = comp[0]["content"]
            s = 0.0
            if text.count("<think>") == 1:
                s += 0.15
            if text.count("\n</think>\n") == 1:
                s += 0.15
            after = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
            if after.lower().startswith("answer:") and len(after) > len("answer:"):
                s += 0.20
            rewards.append(s if s > 0.0 else -0.20)
        return rewards

    @staticmethod
    def format_reward_rec(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
        """
        Check if the Qwen model output matches a specific format.

        Args:
            completions: List of model completions
            **kwargs: Additional arguments

        Returns:
            List of reward scores (1.0 for match, 0.0 for no match)
        """
        import re
        import os
        from datetime import datetime

        # Pattern to match the expected output format
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.search(pattern, content, re.DOTALL) is not None
            for content in completion_contents
        ]

        # Log format results if in debug mode
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(
                log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")

        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def select_reward_func(func: str, task_type: str) -> Callable:
        """
        Select the appropriate reward function based on function name and task type.

        Args:
            func: The type of reward function ('accuracy', 'format', etc.)
            task_type: The type of task ('rec', etc.)

        Returns:
            The reward function to use

        Raises:
            ValueError: If the function or task type is not supported
        """
        if func == "accuracy":
            match task_type:
                case "rec":
                    return NucleotideDNAModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return NucleotideDNAModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")