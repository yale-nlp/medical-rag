import math
import logging
from typing import Dict, List

class VLLMHandler:
    """
    Encapsulates the actual inference logic for a vLLM model.
    This class is used within a ModelWorker process.
    """
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.is_reranker = "reranker" in self.config.get('model', '').lower()

        if self.is_reranker:
            self._setup_reranker_params()

    def _setup_reranker_params(self):
        # This method remains unchanged.
        from vllm import SamplingParams
        
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_length = self.config.get("max_model_len", 4096)
        
        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[yes_token_id, no_token_id]
        )
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def _format_rerank_instruction(self, instruction: str, query: str, doc: str) -> List[Dict]:
        # This method remains unchanged.
        return [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]

    def generate_text(self, payload: Dict) -> str:
        """Handles standard text generation for a single prompt."""
        from vllm import SamplingParams
        
        settings = payload['settings']
        messages = [{"role": "system", "content": payload['system_persona']}, {"role": "user", "content": payload['user_prompt']}]
        
        enable_thinking_flag = settings.get("enable_thinking", True)
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking_flag
        )
        
        params = SamplingParams(
            max_tokens=settings.get("max_tokens", 1024),
            temperature=settings.get("temperature", 0.1),
            top_p=settings.get("top_p", 0.9),
            stop=settings.get("stop_tokens")
        )
        
        output = self.model.generate([prompt], params, use_tqdm=False)
        return output[0].outputs[0].text.strip()
        
    # MODIFIED: Added a new method to handle batch generation requests.
    def generate_text_batch(self, payload: Dict) -> List[str]:
        """
        Handles batch text generation tasks by processing a list of prompts.
        This fully leverages vLLM's continuous batching for high throughput.
        """
        from vllm import SamplingParams
        
        settings = payload['settings']
        prompts_data = payload['prompts']  # Expects a list of (system_persona, user_prompt)

        # Apply chat template to all prompts in the batch
        full_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_persona}, {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=settings.get("enable_thinking", True)
            ) for system_persona, user_prompt in prompts_data
        ]
        
        # Define sampling parameters for the batch
        params = SamplingParams(
            max_tokens=settings.get("max_tokens", 1024),
            temperature=settings.get("temperature", 0.1),
            top_p=settings.get("top_p", 0.9),
            stop=settings.get("stop_tokens")
        )
        
        # Call vLLM's generate method with the entire list of prompts
        logging.info(f"vLLM Handler received a batch of {len(full_prompts)} prompts. Starting generation...")
        outputs = self.model.generate(full_prompts, params, use_tqdm=True)
        logging.info("vLLM Handler finished batch generation.")
        
        # Extract the text from each output and return as a list
        return [output.outputs[0].text.strip() for output in outputs]

    def rerank(self, payload: Dict) -> List[float]:
        # This method is already efficient for reranking and remains unchanged.
        from vllm.inputs import TokensPrompt

        query = payload['query']
        documents = payload['documents']
        instruction = payload['instruction']
        
        pairs = [(query, doc) for doc in documents]
        messages = [self._format_rerank_instruction(instruction, q, doc) for q, doc in pairs]
        
        token_ids_list = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        
        max_len_for_prompt = self.max_length - len(self.suffix_tokens)
        processed_token_ids = [ele[:max_len_for_prompt] + self.suffix_tokens for ele in token_ids_list]
        
        inputs = [TokensPrompt(prompt_token_ids=ele) for ele in processed_token_ids]
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
        
        scores = []
        for output in outputs:
            logprobs = output.outputs[0].logprobs[0]
            logprob_yes_obj = logprobs.get(self.yes_token_id)
            logprob_no_obj = logprobs.get(self.no_token_id)

            logprob_yes = logprob_yes_obj.logprob if logprob_yes_obj else -math.inf
            logprob_no = logprob_no_obj.logprob if logprob_no_obj else -math.inf
            
            exp_yes = math.exp(logprob_yes)
            exp_no = math.exp(logprob_no)
            score = exp_yes / (exp_yes + exp_no) if (exp_yes + exp_no) > 0 else 0.0
            scores.append(score)
            
        return scores