"""Qwen2.5-VL generation with recurrent continuous latent embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.generation.utils import GenerateDecoderOnlyOutput


# Transformers 5 stores the decoder under model.language_model. These are the
# official Qwen2.5-VL conversion rules for checkpoints saved with the flat layout.
LEGACY_QWEN_KEY_MAPPING = {
    r"^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}


def validate_checkpoint_load(loading_info: dict, checkpoint: str) -> None:
    """Reject partially initialized models; latent recurrence amplifies bad weights."""
    problems = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if problems:
        details = "; ".join(f"{key}={value}" for key, value in problems.items())
        raise RuntimeError(f"Checkpoint {checkpoint} did not load exactly: {details}")


@dataclass(frozen=True)
class LatentDecodingSpec:
    protocol: str
    start_id: int
    placeholder_id: int
    end_id: int
    fixed_steps: Optional[int] = None
    max_steps: int = 64


@dataclass
class LatentDecodeState:
    spec: LatentDecodingSpec
    active: bool = False
    steps: int = 0
    pending_embedding: Optional[torch.Tensor] = None

    def advance(self, sampled_id: int, current_hidden: torch.Tensor) -> int:
        """Apply one protocol transition and return the visible bookkeeping ID."""
        if self.active:
            self.steps += 1
            should_end = (
                self.steps >= self.spec.fixed_steps
                if self.spec.fixed_steps is not None
                else sampled_id == self.spec.end_id or self.steps >= self.spec.max_steps
            )
            if should_end:
                self.active = False
                self.pending_embedding = None
                return self.spec.end_id
            self.pending_embedding = current_hidden.detach()
            return self.spec.placeholder_id

        if sampled_id == self.spec.start_id:
            self.active = True
            self.steps = 0
            self.pending_embedding = current_hidden.detach()
        return sampled_id


def latent_decoding_spec(model, protocol: str) -> LatentDecodingSpec:
    """Build and validate a decoding protocol from checkpoint configuration."""
    config = model.config
    if protocol == "monet":
        steps = int(os.getenv("MONET_LATENT_SIZE", os.getenv("LATENT_SIZE", "10")))
        if steps < 1:
            raise ValueError("MONET_LATENT_SIZE must be at least 1")
        return LatentDecodingSpec(
            protocol="monet",
            start_id=int(config.latent_start_id),
            placeholder_id=int(config.latent_token_id),
            end_id=int(config.latent_end_id),
            fixed_steps=steps,
            max_steps=steps,
        )
    if protocol == "lvr":
        max_steps = int(os.getenv("LVR_MAX_LATENT_STEPS", "64"))
        if max_steps < 1:
            raise ValueError("LVR_MAX_LATENT_STEPS must be at least 1")
        return LatentDecodingSpec(
            protocol="lvr",
            start_id=int(config.lvr_start_id),
            placeholder_id=int(config.lvr_id),
            end_id=int(config.lvr_end_id),
            max_steps=max_steps,
        )
    raise ValueError(f"Unsupported latent decoding protocol: {protocol}")


class LatentAwareQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL whose decode loop feeds final-layer states back as inputs."""

    latent_decoding: Optional[LatentDecodingSpec] = None

    def configure_latent_decoding(self, protocol: str) -> None:
        self.latent_decoding = latent_decoding_spec(self, protocol)
        # Continuous one-position feedback requires an incremental KV cache.
        self.config.use_cache = True
        self.generation_config.use_cache = True

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs,
    ):
        spec = self.latent_decoding
        if spec is None:
            return super()._sample(
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if input_ids.shape[0] != 1:
            raise ValueError("Latent decoding currently supports a batch size of 1")
        if not model_kwargs.get("use_cache", generation_config.use_cache):
            raise ValueError("Latent decoding requires use_cache=True")

        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample

        scores = () if return_dict and output_scores else None
        raw_logits = () if return_dict and output_logits else None
        attentions = () if return_dict and output_attentions else None
        hidden_states = () if return_dict and output_hidden_states else None

        unfinished = torch.ones(1, dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        prefill_consumed = False
        state = LatentDecodeState(spec)

        # Force hidden-state production because it is the next latent input.
        generation_config.output_hidden_states = True
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            if prefill_consumed:
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids,
                    next_sequence_length=1,
                    **model_kwargs,
                )
                if state.active:
                    model_inputs["input_ids"] = None
                    model_inputs["inputs_embeds"] = state.pending_embedding[:, None, :]
                model_inputs["output_hidden_states"] = True
                with self._optimize_model_for_decode():
                    outputs = self(**model_inputs, return_dict=True)
            prefill_consumed = True

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            if synced_gpus and this_peer_finished:
                continue

            next_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )
            next_scores = logits_processor(input_ids, next_logits)
            current_hidden = outputs.hidden_states[-1][:, -1, :]

            if return_dict:
                if output_scores:
                    scores += (next_scores,)
                if output_logits:
                    raw_logits += (next_logits,)
                if output_attentions:
                    attentions += (outputs.attentions,)
                if output_hidden_states:
                    hidden_states += (outputs.hidden_states,)

            if do_sample:
                probabilities = nn.functional.softmax(next_scores, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_scores, dim=-1)

            emitted_id = state.advance(int(next_tokens.item()), current_hidden)
            next_tokens = torch.tensor([emitted_id], device=input_ids.device)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            stopped = stopping_criteria(input_ids, scores)
            if not state.active:
                unfinished = unfinished & ~stopped
            this_peer_finished = unfinished.max() == 0
            del outputs

        if streamer is not None:
            streamer.end()
        if not return_dict:
            return input_ids
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=attentions,
            hidden_states=hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
