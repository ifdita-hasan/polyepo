import asyncio
import importlib.util
import json
import os
from typing import Callable, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from verl.trainer.ppo.cluster_prompts import math_cluster_fn as default_cluster_fn

load_dotenv()



#### DIVERSITY AND CLUSTERING FUNCTIONS

# Module-level cluster prompt function - set once during initialization
# This avoids reading config/file in hot paths (async_get_cluster_assignments)
_active_cluster_prompt_fn = None

def _load_cluster_prompt_fn_from_file(prompt_file_path: str, prompt_fn_name: str) -> Callable[[str, List[str]], tuple[str, str]]:
    """
    Load cluster prompt text from a file and return a function that uses it.
    
    Args:
        prompt_file_path: Path to the file containing the clustering function (default: cluster_prompts.py )
        prompt_fn_name: Name of the function in the file containing our clustering function.
    
    Returns:
        A function that takes (context, responses) and returns (system_prompt, user_prompt)
    """
    try:
        if not os.path.exists(prompt_file_path):
            print(f"Warning: Cluster prompt file not found: {prompt_file_path}. Using default.")
            return None
        
        # Load the module from the given file path
        spec = importlib.util.spec_from_file_location(
            "cluster_prompt_module",
            prompt_file_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {prompt_file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function by name
        cluster_prompt_fn = getattr(module, prompt_fn_name, None)
        if not callable(cluster_prompt_fn):
            raise AttributeError(
                f"'{prompt_fn_name}' not found or not callable in {prompt_file_path}"
            )
        
        print(f"Successfully loaded cluster prompt function {prompt_fn_name} from file: {prompt_file_path}")
        
        return cluster_prompt_fn
        
    except Exception as e:
        print(f"Warning: Failed to read cluster prompt from {prompt_file_path}: {e}. Using default.")
        return None
    

def set_cluster_prompt_fn(prompt_file_path: Optional[str] = None, prompt_fn_name: Optional[str] = None):
    """
    Set the global cluster prompt function once during initialization.
    This should be called from ray_trainer.py before any rollouts.
    
    Args:
        prompt_fn_path: Path to text file with prompt template (None = use default)
    """
    global _active_cluster_prompt_fn
    
    if prompt_file_path and prompt_fn_name:
        custom_fn = _load_cluster_prompt_fn_from_file(prompt_file_path, prompt_fn_name)
        if custom_fn is not None:
            _active_cluster_prompt_fn = custom_fn
            return
    
    # Use default (don't set _active_cluster_prompt_fn, cluster_prompt will use default implementation)
    _active_cluster_prompt_fn = None


def cluster_prompt(context: str, responses: List[str]) -> tuple[str, str]:
    """
    LLM clustering prompt for grouping responses based on semantic and conceptual diversity.
    Uses pre-configured function if set, otherwise uses default clustering implementation for math.
    """
    # If a custom function was set, use it
    if _active_cluster_prompt_fn is not None:
        return _active_cluster_prompt_fn(context, responses)
    else: 
        return default_cluster_fn(context, responses)
    
def calculate_set_diversity_from_clusters(
    set_cluster_ids: List[int]
) -> float:
    """
    Calculates the diversity score for a single set given its cluster IDs.

    New definition:
        diversity = (# unique clusters in the set) / n

    For n = 4:
      - all same cluster      -> 1/4
      - 2 clusters represented -> 2/4
      - 3 clusters            -> 3/4
      - 4 clusters            -> 4/4 = 1
    """
    n = len(set_cluster_ids)
    if n <= 0:
        return 0.0

    k = len(set(set_cluster_ids))
    diversity_score = k / n
    return float(diversity_score)

async def async_get_cluster_assignments(
    client: AsyncOpenAI, 
    context: str, 
    responses: list[str], 
    semaphore: asyncio.Semaphore
) -> List[int]:
    """
    Updated to accept the client from the parent orchestrator.
    """
    async with semaphore:
        system_prompt, user_prompt = cluster_prompt(context, responses)
        num_responses = len(responses)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            # Use the client passed from fetch_all_cluster_assignments_async
            # completion = await client.chat.completions.create(
            #     model="Qwen/Qwen3-4B-Instruct-2507", # or your preferred model
            #     messages=[{"role": "user", "content": full_prompt}],
            #     response_format={"type": "json_object"},
            #     temperature=0.0,
            # )

            completion = await client.chat.completions.create(
                model="gemini-2.0-flash", # or your preferred model
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            response_text = completion.choices[0].message.content
            response_data = json.loads(response_text)
            
            cluster_ids = [
                response_data[str(i)]["cluster_id"]
                for i in range(1, num_responses + 1)
            ]
            return cluster_ids

        except Exception as e:
            print(f"Warning: LLM-judge call failed. Error: {e}. Defaulting to distinct.")
            return list(range(1, num_responses + 1))
        
async def fetch_all_cluster_assignments_async(groups_data: list[dict]) -> list[List[int]]:
    # 1. Initialize client using the environment key as in your example
    
    # FOR QWEN JUDGE:
    # client = AsyncOpenAI(
    #     base_url="http://<host>:30126/v1", # Put host url 
    #     api_key="random",
    # )
    
    # FOR GEMINI JUDGE:
    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ.get('GEMINI_API_KEY'),
    )
    
    MAX_CONCURRENT_REQUESTS = 32
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    try:
        tasks = []
        for group in groups_data:
            # 2. Pass the client instance to the child function
            tasks.append(
                async_get_cluster_assignments(
                    client,
                    group["context"],
                    group["responses"],
                    semaphore,
                )
            )

        # 3. Gather all tasks. return_exceptions=True prevents one crash from stopping the whole batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print("Finished fetching all cluster assignments from Gemini.")

    except Exception as e:
        print(f"Unexpected error during async processing: {e}")
        # Fallback: return distinct cluster IDs for everyone
        return [list(range(1, len(g["responses"]) + 1)) for g in groups_data]
    
    finally:
        # 4. CRITICAL: Always close the client to prevent memory leaks/SSL warnings
        await client.close()

    # 5. Process results and handle exceptions per individual prompt
    final_assignments = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"Task {i} failed with error: {res}")
            num_responses = len(groups_data[i]["responses"])
            final_assignments.append(list(range(1, num_responses + 1)))
        else:
            final_assignments.append(res)

    return final_assignments