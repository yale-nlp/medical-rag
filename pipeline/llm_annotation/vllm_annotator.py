import os
import json
import torch
import re
import gc
import time
import asyncio
from pathlib import Path
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_VLLM_Engine, _VLLM_SamplingParams = None, None
try:
    from vllm import LLM as VLLM_Engine_Import, SamplingParams as VLLM_SamplingParams_Import
    _VLLM_Engine = VLLM_Engine_Import
    _VLLM_SamplingParams = VLLM_SamplingParams_Import
    print("[INFO] annotator.py: vLLM imported successfully.")
except ImportError:
    print("[WARN] annotator.py: vLLM not installed.")

class BaseAnnotator(ABC):
    def __init__(self, config_path="annotation_configs.json"):
        with open(config_path, encoding="utf-8") as cf:
            self.config = json.load(cf)
        self.prompts_config = self.config["annotation_prompts"]
        self.system_role = self.prompts_config["system_role"]

    def _construct_prompt(self, template_key: str, context_vars: dict) -> str:
        template = self.prompts_config[template_key]["prompt_template"]
        return template.format(**context_vars)

class OpenAIAnnotator(BaseAnnotator):
    def __init__(self, config_path="annotation_configs.json"):
        super().__init__(config_path)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key: raise ValueError("OPENAI_API_KEY not found.")
        # [MODIFIED] Initialize both sync and async clients
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.model_name = self.config["openai_annotator_settings"]["model"]

    def evaluate(self, prompt_text: str) -> dict:
        # This synchronous version is kept for simple tests or compatibility
        try:
            response = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_role}, {"role": "user", "content": prompt_text}],
                temperature=0.0, max_tokens=5
            )
            raw_output = response.choices[0].message.content
            score_match = re.search(r'\d', raw_output)
            return {"score": int(score_match.group(0))} if score_match else {"error": "No digit in response"}
        except Exception as e:
            return {"error": str(e)}

    # [ADDED] Asynchronous evaluation method for high performance
    async def evaluate_async(self, prompt_text: str, max_retries: int = 5, initial_delay: int = 1) -> dict:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": self.system_role}, {"role": "user", "content": prompt_text}],
                    temperature=0.0, max_tokens=5
                )
                raw_output = response.choices[0].message.content
                score_match = re.search(r'\d', raw_output)
                if score_match:
                    return {"score": int(score_match.group(0))}
                return {"error": "No digit in OpenAI response", "raw_output": raw_output}
            except Exception as e:
                if "429" in str(e) or "Rate limit" in str(e):
                    print(f"[WARN] OpenAI Rate limit hit. Attempt {attempt + 1}/{max_retries}. Retrying in {delay}s...")
                    if attempt + 1 == max_retries:
                        return {"error": f"Rate limit exceeded after {max_retries} retries.", "details": str(e)}
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return {"error": "Non-retryable API error", "details": str(e)}
        return {"error": f"All {max_retries} retries failed."}

class GeminiAnnotator(BaseAnnotator):
    def __init__(self, config_path="annotation_configs.json"):
        super().__init__(config_path)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.config["gemini_annotator_settings"]["model_name"], system_instruction=self.system_role)

    async def evaluate_async(self, prompt_text: str, max_retries: int = 5, initial_delay: int = 1) -> dict:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(prompt_text, generation_config={"temperature": 0.0})
                raw_output = response.text.strip()
                score_match = re.search(r'\d', raw_output)
                if score_match:
                    return {"score": int(score_match.group(0))}
                return {"error": "No digit in response", "raw_output": raw_output}
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    print(f"[WARN] Rate limit hit. Attempt {attempt + 1}/{max_retries}. Retrying in {delay}s...")
                    if attempt + 1 == max_retries:
                        return {"error": f"Rate limit exceeded after {max_retries} retries.", "details": str(e)}
                    await asyncio.sleep(delay)
                    delay *= 5
                else:
                    return {"error": "Non-retryable API error", "details": str(e)}
        return {"error": f"All {max_retries} retries failed."}

class HuggingFaceAnnotator(BaseAnnotator):
    def __init__(self, config_path="annotation_configs.json"):
        super().__init__(config_path)
        if _VLLM_Engine is None: raise ImportError("vLLM is required.")
        hf_settings = self.config["huggingface_annotator_settings"]
        self.engine = _VLLM_Engine(model=hf_settings["model_name"], **hf_settings.get("vllm_params", {}))
        self.tokenizer = self.engine.get_tokenizer()
        self.score_sampling_params = _VLLM_SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"]))

    def evaluate(self, prompt_text: str) -> dict:
        full_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_role}, {"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
        try:
            outputs = self.engine.generate([full_prompt], self.score_sampling_params, use_tqdm=False)
            return {"score": int(outputs[0].outputs[0].text.strip())} if outputs and outputs[0].outputs else {"error": "No vLLM output"}
        except Exception as e:
            return {"error": str(e)}

def get_annotator(config_path="annotation_configs.json") -> BaseAnnotator:
    with open(config_path, encoding="utf-8") as cf:
        config = json.load(cf)
    choice = config.get("annotator_choice", "openai").lower()
    if choice == "openai": return OpenAIAnnotator(config_path)
    if choice == "gemini": return GeminiAnnotator(config_path)
    if choice == "huggingface": return HuggingFaceAnnotator(config_path)
    raise ValueError(f"Unsupported annotator: {choice}")