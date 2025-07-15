import asyncio
import uuid
from typing import Dict, Any, List
from multiprocessing import Queue

class LocalLLMService:
    """
    A service that acts as a client to a ModelWorker process running in the background.
    It sends requests to the worker via a queue and waits for the response.
    """
    def __init__(
        self,
        model_name: str,
        request_queue: Queue,
        pending_futures_dict: dict,
    ):
        """
        Initializes the service with communication channels to the worker process.

        Args:
            model_name (str): The name of the model this service communicates with.
            request_queue (Queue): The queue to send requests to the worker.
            pending_futures_dict (dict): A local dictionary to store pending asyncio.Future objects.
        """
        self.model_name = model_name
        self.request_queue = request_queue
        self.pending_futures = pending_futures_dict
        self.is_reranker = "reranker" in self.model_name.lower()

    async def _send_request_and_wait(self, payload: Dict) -> Any:
        """
        Sends a request to the worker and waits for the result using an asyncio.Future.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        self.pending_futures[request_id] = future
        
        self.request_queue.put({"request_id": request_id, "payload": payload})
        
        try:
            result = await future
            return result
        finally:
            # Clean up the dictionary to prevent memory leaks
            if request_id in self.pending_futures:
                del self.pending_futures[request_id]

    async def generate_text(
        self,
        system_persona: str,
        user_prompt: str,
        settings: Dict[str, Any]
    ) -> str:
        """Sends a single text generation request to the appropriate model worker."""
        payload = {
            "task": "generate",
            "system_persona": system_persona,
            "user_prompt": user_prompt,
            "settings": settings,
        }
        return await self._send_request_and_wait(payload)

    # MODIFIED: Added a new method to handle batch generation requests.
    async def generate_text_batch(
        self,
        prompts_data: List[tuple[str, str]],  # A list of (system_persona, user_prompt) tuples
        settings: Dict[str, Any]
    ) -> List[str]:
        """Sends a batch text generation request to the model worker."""
        payload = {
            "task": "generate_batch",
            "prompts": prompts_data,
            "settings": settings,
        }
        return await self._send_request_and_wait(payload)

    async def rerank(self, query: str, documents: List[str], instruction: str) -> List[float]:
        """Sends a reranking request to the reranker worker."""
        if not self.is_reranker:
            raise TypeError("The 'rerank' method can only be called on a reranker model.")
        
        payload = {
            "task": "rerank",
            "query": query,
            "documents": documents,
            "instruction": instruction,
        }
        return await self._send_request_and_wait(payload)