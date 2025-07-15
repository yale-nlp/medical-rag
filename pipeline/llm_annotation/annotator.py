import os
import json
import time
import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import random
from enum import Enum

from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int
    requests_per_second: Optional[float] = None
    burst_size: Optional[int] = None
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    
    def __post_init__(self):
        if self.requests_per_second is None:
            self.requests_per_second = self.requests_per_minute / 60.0
        if self.burst_size is None:
            self.burst_size = max(1, min(10, self.requests_per_minute // 10))


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.capacity = config.burst_size
        self.tokens = self.capacity
        self.refill_rate = config.requests_per_second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket. Returns wait time if needed."""
        async with self.lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return wait_time


class CircuitBreaker:
    """Circuit breaker pattern for handling failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = asyncio.Lock()
    
    async def call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self.lock:
            if self.state == "open":
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "half-open"
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            async with self.lock:
                if self.state == "half-open":
                    self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")
            raise


@dataclass
class APIRequest:
    """Represents a single API request"""
    id: str
    prompt: str
    retry_count: int = 0
    max_retries: int = 5
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Represents an API response"""
    request_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    duration: float = 0.0


class BaseAnnotator(ABC):
    """Base class for all annotators"""
    
    def __init__(self, config_path: str = "annotation_configs.json"):
        with open(config_path, encoding="utf-8") as cf:
            self.config = json.load(cf)
        self.prompts_config = self.config["annotation_prompts"]
        self.system_role = self.prompts_config["system_role"]
    
    def _construct_prompt(self, template_key: str, context_vars: dict) -> str:
        template = self.prompts_config[template_key]["prompt_template"]
        return template.format(**context_vars)
    
    @abstractmethod
    async def evaluate_async(self, prompt_text: str) -> dict:
        pass


class OpenAIAnnotator(BaseAnnotator):
    """OpenAI API annotator with professional-grade error handling"""
    
    def __init__(self, config_path: str = "annotation_configs.json"):
        super().__init__(config_path)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        
        # Initialize clients
        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.model_name = self.config["openai_annotator_settings"]["model"]
        
        # Rate limiting configuration
        rpm = self.config.get("openai_annotator_settings", {}).get("requests_per_minute", 60)
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=rpm,
            burst_size=min(10, rpm // 6)
        )
        self.rate_limiter = TokenBucket(self.rate_limit_config)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=60)
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0.0
        }
    
    def evaluate(self, prompt_text: str) -> dict:
        """Synchronous evaluation for backward compatibility"""
        try:
            response = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_role},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                max_tokens=5
            )
            raw_output = response.choices[0].message.content
            score_match = re.search(r'\d', raw_output)
            return {"score": int(score_match.group(0))} if score_match else {"error": "No digit in response"}
        except Exception as e:
            return {"error": str(e)}
    
    async def evaluate_async(self, prompt_text: str) -> dict:
        """Asynchronous evaluation with professional error handling"""
        request_id = f"{time.time()}_{random.randint(1000, 9999)}"
        request = APIRequest(id=request_id, prompt=prompt_text)
        
        return await self._process_request_with_retry(request)
    
    async def _process_request_with_retry(self, request: APIRequest) -> dict:
        """Process request with exponential backoff and jitter"""
        start_time = time.time()
        
        while request.retry_count <= request.max_retries:
            try:
                # Wait for rate limit
                wait_time = await self.rate_limiter.acquire()
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                
                # Make API call with circuit breaker
                result = await self.circuit_breaker.call_with_circuit_breaker(
                    self._make_api_call,
                    request.prompt
                )
                
                # Update metrics
                self.metrics["total_requests"] += 1
                self.metrics["successful_requests"] += 1
                self.metrics["total_duration"] += time.time() - start_time
                
                return result
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Request {request.id} failed (attempt {request.retry_count + 1}): {error_str}")
                
                self.metrics["failed_requests"] += 1
                request.retry_count += 1
                
                if request.retry_count > request.max_retries:
                    return {"error": f"Max retries exceeded: {error_str}"}
                
                # Calculate backoff with jitter
                if "429" in error_str or "rate_limit" in error_str.lower():
                    base_delay = min(60, 2 ** request.retry_count)
                    jitter = random.uniform(0, base_delay * 0.1)
                    delay = base_delay + jitter
                else:
                    delay = min(10, 1 * (2 ** (request.retry_count - 1)))
                
                logger.info(f"Retrying request {request.id} in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        return {"error": "Max retries exceeded"}
    
    async def _make_api_call(self, prompt_text: str) -> dict:
        """Make the actual API call"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_role},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0,
            max_tokens=5,
            timeout=30
        )
        
        raw_output = response.choices[0].message.content
        score_match = re.search(r'\d', raw_output)
        
        if score_match:
            return {"score": int(score_match.group(0))}
        return {"error": "No digit in response", "raw_output": raw_output}
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        metrics = self.metrics.copy()
        if metrics["total_requests"] > 0:
            metrics["average_duration"] = metrics["total_duration"] / metrics["total_requests"]
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
        return metrics


class GeminiAnnotator(BaseAnnotator):
    """Gemini API annotator"""
    
    def __init__(self, config_path: str = "annotation_configs.json"):
        super().__init__(config_path)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            self.config["gemini_annotator_settings"]["model_name"],
            system_instruction=self.system_role
        )
        
        # Rate limiting
        rpm = self.config.get("gemini_annotator_settings", {}).get("requests_per_minute", 60)
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=rpm,
            burst_size=min(10, rpm // 6)
        )
        self.rate_limiter = TokenBucket(self.rate_limit_config)
        self.circuit_breaker = CircuitBreaker()
    
    async def evaluate_async(self, prompt_text: str) -> dict:
        """Asynchronous evaluation"""
        request = APIRequest(id=f"gemini_{time.time()}", prompt=prompt_text)
        
        while request.retry_count <= request.max_retries:
            try:
                wait_time = await self.rate_limiter.acquire()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                result = await self.circuit_breaker.call_with_circuit_breaker(
                    self._make_api_call,
                    prompt_text
                )
                return result
                
            except Exception as e:
                error_str = str(e)
                request.retry_count += 1
                
                if request.retry_count > request.max_retries:
                    return {"error": f"Max retries exceeded: {error_str}"}
                
                if "429" in error_str or "resource has been exhausted" in error_str.lower():
                    delay = min(60, 3 ** request.retry_count)
                else:
                    delay = min(10, 2 ** request.retry_count)
                
                await asyncio.sleep(delay)
        
        return {"error": "Max retries exceeded"}
    
    async def _make_api_call(self, prompt_text: str) -> dict:
        """Make the actual API call"""
        response = await self.model.generate_content_async(
            prompt_text,
            generation_config={"temperature": 0.0}
        )
        raw_output = response.text.strip()
        score_match = re.search(r'\d', raw_output)
        
        if score_match:
            return {"score": int(score_match.group(0))}
        return {"error": "No digit in response", "raw_output": raw_output}


# HuggingFace annotator remains the same as it's synchronous
VLLMEngine, VLLMSamplingParams = None, None
try:
    from vllm import LLM as VLLM_Engine_Import, SamplingParams as VLLM_SamplingParams_Import
    VLLMEngine = VLLM_Engine_Import
    VLLMSamplingParams = VLLM_SamplingParams_Import
    logger.info("vLLM imported successfully.")
except ImportError:
    logger.warning("vLLM not installed.")


class HuggingFaceAnnotator(BaseAnnotator):
    """HuggingFace annotator using vLLM"""
    
    def __init__(self, config_path: str = "annotation_configs.json"):
        super().__init__(config_path)
        if VLLMEngine is None:
            raise ImportError("vLLM is required for HuggingFace annotator.")
        
        hf_settings = self.config["huggingface_annotator_settings"]
        self.engine = VLLMEngine(
            model=hf_settings["model_name"],
            **hf_settings.get("vllm_params", {})
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.score_sampling_params = VLLMSamplingParams(
            temperature=0,
            max_tokens=1,
            allowed_token_ids=self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])
        )
    
    def evaluate(self, prompt_text: str) -> dict:
        """Synchronous evaluation"""
        full_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_role},
                {"role": "user", "content": prompt_text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        
        try:
            outputs = self.engine.generate(
                [full_prompt],
                self.score_sampling_params,
                use_tqdm=False
            )
            if outputs and outputs[0].outputs:
                return {"score": int(outputs[0].outputs[0].text.strip())}
            return {"error": "No vLLM output"}
        except Exception as e:
            return {"error": str(e)}


def get_annotator(config_path: str = "annotation_configs.json") -> BaseAnnotator:
    """Factory function to get the appropriate annotator"""
    with open(config_path, encoding="utf-8") as cf:
        config = json.load(cf)
    
    choice = config.get("annotator_choice", "openai").lower()
    
    annotator_map = {
        "openai": OpenAIAnnotator,
        "gemini": GeminiAnnotator,
        "huggingface": HuggingFaceAnnotator
    }
    
    if choice not in annotator_map:
        raise ValueError(f"Unsupported annotator: {choice}")
    
    return annotator_map[choice](config_path)


# Re-export for backward compatibility
import re
__all__ = ['get_annotator', 'BaseAnnotator', 'OpenAIAnnotator', 'GeminiAnnotator', 'HuggingFaceAnnotator']