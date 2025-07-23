import os
# Specify which GPUs to use (at the very top of the script)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import subprocess
import time
import sys
import json
import asyncio
import aiohttp
from tqdm import tqdm
import threading
import queue
import argparse 

# --------------------- Configuration ---------------------
# vLLM Server and Model Settings
VLLM_MODEL = "google/medgemma-27b-text-it"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.75
MAX_NUM_SEQS = 2048  # Number of sequences to process in parallel 
MAX_MODEL_LEN = 8192 # available input tokens

# API Endpoint Settings
VLLM_HOST = "localhost"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_API_KEY = "not-needed"

# Client Settings
BATCH_SIZE = 2048  # Number of lines to read into memory at once
CONCURRENT_REQUESTS = 1024 # Number of concurrent requests to send to the server (aligned with MAX_NUM_SEQS)


# --------------------- vLLM Server Management (Robust Version) ---------------------

def enqueue_output(stream, q):
    """Thread target function to read a subprocess's output and put it on a queue."""
    for line in iter(stream.readline, ''):
        q.put(line)
    stream.close()

def start_vllm_server():
    """Starts the vLLM server as a background process."""
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLLM_MODEL,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--enforce-eager",
    ]
    print("🚀 Starting vLLM server...")
    print(f"   Command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    return process

def wait_for_server_ready(process, timeout=600):
    """Waits for the server to be ready in a non-blocking way using a separate thread."""
    print("⏳ Waiting for vLLM server to be ready...")
    
    q = queue.Queue()
    thread = threading.Thread(target=enqueue_output, args=(process.stdout, q))
    thread.daemon = True
    thread.start()

    ready_messages = [
        f"Uvicorn running on http://{VLLM_HOST}:{VLLM_PORT}",
        "Application startup complete"
    ]
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            print("❌ Server process terminated unexpectedly during startup.")
            while not q.empty():
                print(f"[Server Log] {q.get_nowait().strip()}")
            return False

        try:
            line = q.get(timeout=0.1)
            print(f"[Server Log] {line.strip()}")
            if any(msg in line for msg in ready_messages):
                print("✅ vLLM Server is ready!")
                print("   Waiting 3 seconds for server to stabilize...")
                time.sleep(3)
                return True
        except queue.Empty:
            continue
            
    print("❌ Server readiness check timed out.")
    return False

# --------------------- Client & Processing Logic ---------------------

async def fetch_completion(session, report_dict, system_prompt):
    """Asynchronously fetches a completion from the vLLM server for a single report."""
    user_content = report_dict.get("report", "")
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 256,
        "repetition_penalty": 1.2,
        "stop": ["}\n"]
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    try:
        async with session.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers) as response:
            if response.status == 200:
                resp_json = await response.json()
                content = resp_json.get("choices", [{}])[0].get("message", {}).get("content")
                report_dict["_raw_response"] = content if content else ""
            else:
                error_text = await response.text()
                report_dict["_raw_response"] = f"API_ERROR: Status {response.status}, Body: {error_text}"
    except Exception as e:
        report_dict["_raw_response"] = f"REQUEST_ERROR: {e}"
    return report_dict

async def generate_keywords_batch(reports, system_prompt, pbar):
    """Processes a batch of reports asynchronously with a semaphore for concurrency control."""
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for rpt in reports:
            await semaphore.acquire()
            task = asyncio.create_task(fetch_completion(session, rpt, system_prompt))
            task.add_done_callback(lambda t: (semaphore.release(), pbar.update(1)))
            tasks.append(task)
        processed_reports = await asyncio.gather(*tasks)
    return processed_reports

def extract_keywords_from_response(resp: str) -> list[str]:
    """Extracts a list of keywords from the raw JSON string response from the model."""
    if not resp or not resp.strip(): raise ValueError("Empty response from model")
    try:
        obj = json.loads(resp)
        kws = obj.get("keywords", [])
        if not isinstance(kws, list): raise ValueError("'keywords' key does not contain a list")
        return kws
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}. Raw response: '{resp[:200]}...'")

# -------------------------- Main Execution (JSONL Version) --------------------------

def main(args):
    """Main function to orchestrate the server startup, processing, and shutdown."""
    server_process = None
    try:
        server_process = start_vllm_server()
        if not wait_for_server_ready(server_process):
            raise RuntimeError("Failed to start vLLM server.")

        try:
            with open(args.prompts, "r", encoding="utf-8") as f:
                system_prompt = json.load(f)["system"]["keyword_generator"]
        except (FileNotFoundError, KeyError) as e:
            print(f"❌ Error loading prompt: {e}")
            print(f"   Please ensure '{args.prompts}' contains 'system.keyword_generator'.")
            return

        try:
            print(f"⏳ Counting total lines in '{args.input}'...")
            with open(args.input, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f)
            print(f"   - Total cases to process: {total_rows:,}")
        except FileNotFoundError:
            print(f"❌ Input file not found: {args.input}")
            return

        print(f"\n✨ Starting keyword extraction from '{args.input}'...")
        
        with open(args.output, "w", encoding="utf-8") as f_out:
            with open(args.input, "r", encoding="utf-8") as f_in:
                with tqdm(total=total_rows, desc="Processing Reports") as pbar:
                    
                    batch_buffer = []
                    for line in f_in:
                        try:
                            report_dict = json.loads(line)
                            batch_buffer.append(report_dict)
                            
                            if len(batch_buffer) >= BATCH_SIZE:
                                processed_reports = asyncio.run(generate_keywords_batch(batch_buffer, system_prompt, pbar))
                                
                                for rpt in processed_reports:
                                    try:
                                        kws = extract_keywords_from_response(rpt.get("_raw_response", ""))
                                        unique_kws = list(dict.fromkeys(kws))
                                        rpt["keywords"] = unique_kws
                                    except Exception as e:
                                        tqdm.write(f"[⚠️ WARNING] Case ID {rpt.get('case_id', 'N/A')} parse failed: {e}")
                                        rpt["keywords"] = []
                                    f_out.write(json.dumps(rpt) + '\n')
                                
                                batch_buffer = []

                        except json.JSONDecodeError:
                            tqdm.write(f"[⚠️ WARNING] Skipping malformed JSON line: {line.strip()}")
                            pbar.update(1)
                            continue

                    if batch_buffer:
                        processed_reports = asyncio.run(generate_keywords_batch(batch_buffer, system_prompt, pbar))
                        for rpt in processed_reports:
                            try:
                                kws = extract_keywords_from_response(rpt.get("_raw_response", ""))
                                unique_kws = list(dict.fromkeys(kws))
                                rpt["keywords"] = unique_kws
                            except Exception as e:
                                tqdm.write(f"[⚠️ WARNING] Case ID {rpt.get('case_id', 'N/A')} parse failed: {e}")
                                rpt["keywords"] = []
                            f_out.write(json.dumps(rpt) + '\n')

        print(f"\n🎉 Batch processing complete. Results saved to '{args.output}'.")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")
    finally:
        if server_process:
            print("\n🔌 Shutting down vLLM server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
                print("   Server shut down successfully.")
            except subprocess.TimeoutExpired:
                print("   Server did not terminate gracefully, forcing kill.")
                server_process.kill()
            
            remaining_output, _ = server_process.communicate()
            if remaining_output:
                print("\n--- Final Server Logs ---")
                print(remaining_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keywords from radiology reports using a vLLM server.")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file where results will be saved."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to the JSON file containing the system prompts."
    )
    
    parsed_args = parser.parse_args()
    
    main(parsed_args)