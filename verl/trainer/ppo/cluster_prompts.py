# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Cluster prompt templates for grouping responses based on semantic and conceptual diversity.
All functions must have input context, responses and output a system_prompt and user_prompt
"""

from typing import List

def math_cluster_fn(context: str, responses: List[str]) -> tuple[str, str]:
    """
    LLM clustering prompt for grouping responses based on semantic and conceptual diversity.
    """
    n_responses = len(responses)

    system_prompt = f"""
    Your ONLY task is to cluster the {n_responses} responses into buckets based on their reasoning algorithm, including both the overall strategy and the methods used at key intermediate steps.

    **INPUT FORMAT:** You will receive:
    1) A "Context" describing the task.
    2) A numbered list of Responses from 1 to {n_responses}. Each response contains a reasoning process and final answer. 
       Note: Responses may or may not explicitly state their strategy; you must infer the strategy by analyzing the mathematical steps taken.

    **CLUSTERING CRITERIA:**
    (1) Macro-strategy: The overall conceptual framework (e.g., recursion vs infinite series; prime factorization vs gcd-based formula).
    (2) Micro-strategy: The specific method used to resolve key intermediate steps. Examples include: how absolute values are removed (± case split vs squaring), how intervals are partitioned, or how a basis is chosen.

    **CLUSTERING RULES:** - Cluster strictly based on logic and approach. NOT on wording, tone, or formatting.  
    - Two responses share a cluster_id IF AND ONLY IF they use the same macro-strategy AND the same micro-strategy at every key step.
    - Arithmetic errors do NOT create new clusters if the underlying logic is identical.
    - **SPECIAL CLUSTER 100:** You MUST assign `cluster_id: 100` to any response that is:
        * Gibberish (random characters, nonsense strings).
        * Irrelevant to the math problem (off-topic text).
        * Non-mathematical reasoning (e.g., writing code to solve it instead of math, or making a random guess at the final answer without logical steps).
        * Excessive repetition (e.g., repeating the final answer multiple times at the end of the response).

    **OUTPUT RULES (STRICT):** 1. Respond ONLY with a JSON object. No text outside the JSON.
    2. The JSON must contain exactly {n_responses} keys: "1", "2", ..., "{n_responses}".  
    3. The value for each key must be:  
        "chain_of_thought": "Macro: [short description]. Micro: [short description]."
        "cluster_id": integer.
    4. chain_of_thought must be concise and avoid repeating the actual calculations.

    **Few-Shot Example 1:**

    **Context:**
    What is the smallest value of x such that |5x - 1| = |3x + 2|?

    **Responses:**
    1. We can split this into two cases: 5x - 1 = 3x + 2 or 5x - 1 = -(3x + 2). Solving the first gives 2x = 3 so x = 1.5. The second gives 8x = -1 so x = -0.125.
    2. The expression 5x-1 changes sign at 1/5, and 3x+2 changes at -2/3. For x < -2/3, we have -(5x-1) = -(3x+2). For -2/3 < x < 1/5, we have -(5x-1) = 3x+2. Solve -(5x - 1) = 3x + 2 for the range, yielding x = -0.125.
    3. Using the property that |a|=|b| implies a=b or a=-b, we get 5x-1 = 3x+2 (x=1.5) and 5x-1 = -3x-2 (x=-0.125). So the answer is x = -0.125
    4. To get rid of the absolute values, square both sides: (5x - 1)^2 = (3x + 2)^2. This expands to 25x^2 - 10x + 1 = 9x^2 + 12x + 4. Solve 16x^2 - 22x - 3 = 0. So, x = -1/8, 3/2. Final answer is x = -1/8
    5. Either 5x - 1 = 3x + 2 or 5x - 1 = -3x - 2. This leads to x = 3/2 and x = 0. So, final answer is x = 0.
    6. I think the answer is probably 0 or maybe 1.5. 
    7. 基金份额；份额；部分. asdf qwer zxcv 9999 ---- ??? Let's write Python to check each x from -10 to 10: `if abs(5*x-1) == abs(3*x+2): print(x)`. The answer is -0.125. 

    **Expected Output:**
    {{
    "1": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values.",
        "cluster_id": 1
    }},
    "2": {{
        "chain_of_thought": "Macro: Interval analysis. Micro: Testing expression sign changes across number line partitions.",
        "cluster_id": 2
    }},
    "3": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values.",
        "cluster_id": 1
    }},
    "4": {{
        "chain_of_thought": "Macro: Algebraic transformation. Micro: Squaring both sides to create and solve a quadratic equation.",
        "cluster_id": 3
    }},
    "5": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values (contains arithmetic error).",
        "cluster_id": 1
    }},
    "6": {{
        "chain_of_thought": "Macro: Non-mathematical. Micro: Random guessing without any logical derivation.",
        "cluster_id": 100
    }},
    "7": {{
        "chain_of_thought": "Macro: Gibberish/Non-mathematical. Micro: Contains random non-English text, nonsense strings, and code-based iteration.",
        "cluster_id": 100
    }}
    }}

    **Few-Shot Example 2:**

    **Context:**
    What is the least common multiple of 72 and 96?

    **Responses:**
    1. 72 = 2^3 * 3^2. 96 = 2^5 * 3^1. To find the LCM, we take the highest power of each prime factor present: 2^5 * 3^2 = 32 * 9 = 288.
    2. Prime factors of 72: 2, 2, 2, 3, 3. Prime factors of 96: 2, 2, 2, 2, 2, 3. The union of these sets is five 2s and two 3s. Total: 276.
    3. First find the GCD using the Euclidean algorithm: 96 = 72(1) + 24; 72 = 24(3) + 0. GCD is 24. LCM is (72 * 96) / 24.
    4. 72 = 8*9, 96 = 8*12. The answer is 288. The answer is 288. The answer is 288. The answer is 288.

    **Expected Output:**
    {{
    "1": {{
        "chain_of_thought": "Macro: Prime factorization analysis. Micro: LCM via maximum exponents of prime factors.",
        "cluster_id": 1
    }},
    "2": {{
        "chain_of_thought": "Macro: Prime factorization analysis. Micro: LCM via maximum exponents of prime factors (contains arithmetic error).",
        "cluster_id": 1
    }},
    "3": {{
        "chain_of_thought": "Macro: Product-GCD relationship. Micro: GCD calculation via Euclidean algorithm followed by the LCM formula.",
        "cluster_id": 2
    }},
    "4": {{
        "chain_of_thought": "Macro: Excessive repetition. Micro: Response loops the final answer multiple times at the end.",
        "cluster_id": 100
    }}
    }}
    """

    # format responses
    response_list_str = "\n".join(f"{i+1}. {resp}" for i, resp in enumerate(responses))

    user_prompt = f"""
    **Context:**
    {context}

    **Responses:**
    {response_list_str}
    """
    return system_prompt, user_prompt


def highlevel_math_cluster_fn(context: str, responses: List[str]) -> tuple[str, str]:
    """
    LLM clustering prompt for grouping responses based on semantic and conceptual diversity.
    """
    n_responses = len(responses)

    system_prompt = f"""
    Your ONLY task is to cluster the {n_responses} responses into buckets based on their reasoning algorithm, including both the overall strategy and the methods used at key intermediate steps.

    **INPUT FORMAT:** You will receive:
    1) A "Context" describing the task.
    2) A numbered list of Responses from 1 to {n_responses}.
        - Each response will typically contain:
            - <high_level>...</high_level>: a short summary of the intended strategy.
            - solution: the detailed reasoning and final result.

    Your job is to ONLY cluster responses by their reasoning algorithm, defined as:
    (1) Macro-strategy: The overall conceptual framework (e.g., recursion vs infinite series; prime factorization vs gcd-based formula).
    (2) Micro-strategy: The specific method used to resolve key intermediate steps. Examples of key steps include (non-exhaustive): eliminating absolute values (direct ± case split vs squaring-and-checking), choosing how to partition cases or intervals, etc.

    **CLUSTERING RULES:** 
    - Cluster strictly based on logic and approach. NOT on wording, tone, or structure.  
    - Two responses share a cluster_id IF AND ONLY IF they use the same macro-strategy AND the same micro-strategy at every key step.
    - Arithmetic errors do not create new clusters if the logic is identical.
    - ALL Gibberish or irrelevant responses go in the same cluster.

    **OUTPUT RULES (STRICT):** 
    1. Respond ONLY with a JSON object. No text outside the JSON is allowed.
    2. The JSON must contain exactly {n_responses} keys: "1", "2", ..., "{n_responses}", where each key corresponds to the index of the response.  
    3. The value for each key must be a JSON object with EXACTLY two fields:  
        "chain_of_thought": A string in the format: "Macro: [short description]. Micro: [short description]."
        "cluster_id": integer.
    4. chain_of_thought must be concise and avoid calculations or results.

    **Few-Shot Example 1:**

    **Context:**
    What is the smallest value of x such that |5x - 1| = |3x + 2|?

    **Responses:**
    1. <high_level>Solve by considering the cases A = B and A = -B.</high_level> 5x - 1 = 3x + 2 and 5x - 1 = -(3x + 2). x = 1.5, x = -0.125.
    2. <high_level>Determine where each expression changes sign and solve in the appropriate interval.</high_level> Signs change at x=0.2 and x=-0.66. Solve -(5x - 1) = 3x + 2 for the range, yielding x = -0.125.
    3. <high_level>Analyze both algebraic cases.</high_level> |A|=|B| means A=B or A=-B. x=3/2 and x=1/8.
    4. <high_level>Square both sides.</high_level> (5x - 1)^2 = (3x + 2)^2. 16x^2 - 22x - 3 = 0. x = -1/8, 3/2.
    5. <high_level>Solve by considering A = ±B.</high_level> 5x - 1 = 3x + 2 and 5x - 1 = -(3x + 2). x = 1.5, x = 0.
    6. asdf qwer zxcv 9999 ---- ??? text 123.

    **Expected Output:**
    {{
    "1": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values.",
        "cluster_id": 1
    }},
    "2": {{
        "chain_of_thought": "Macro: Interval analysis. Micro: Solving sub-equations based on expression sign changes.",
        "cluster_id": 2
    }},
    "3": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values (contains arithmetic error).",
        "cluster_id": 1
    }},
    "4": {{
        "chain_of_thought": "Macro: Algebraic transformation. Micro: Squaring both sides to eliminate absolute values and solving the quadratic.",
        "cluster_id": 3
    }},
    "5": {{
        "chain_of_thought": "Macro: Algebraic casework. Micro: Direct ± case split to remove absolute values (contains arithmetic error).",
        "cluster_id": 1
    }},
    "6": {{
        "chain_of_thought": "Macro: Response is nonsensical or irrelevant. Micro: Response is nonsensical or irrelevant.",
        "cluster_id": 4
    }}
    }}

    **Few-Shot Example 2:**

    **Context:**
    What is the least common multiple of 72 and 96?

    **Responses:**
    1. <high_level>Prime factorize both numbers and take maximum exponents.</high_level> 72 = 2^3 * 3^2, 96 = 2^5 * 3^1. LCM = 2^5 * 3^2 = 288.
    2. <high_level>Use prime factorization.</high_level> 72 = 2*2*2*3*3, 96 = 2*2*2*2*2*3. Union is 2^5 * 3^2 = 276.
    3. <high_level>Compute gcd using the Euclidean algorithm and apply lcm(a,b)=ab/gcd(a,b).</high_level> gcd(96,72)=24. LCM = (72*96)/24 = 288.

    **Expected Output:**
    {{
    "1": {{
        "chain_of_thought": "Macro: Prime factorization analysis. Micro: Finding LCM by taking the maximum exponent of all prime factors.",
        "cluster_id": 1
    }},
    "2": {{
        "chain_of_thought": "Macro: Prime factorization analysis. Micro: Finding LCM by taking the maximum exponent of all prime factors (contains arithmetic error).",
        "cluster_id": 1
    }},
    "3": {{
        "chain_of_thought": "Macro: Division-based algorithm. Micro: Calculating GCD via Euclidean algorithm and using the LCM-GCD product formula.",
        "cluster_id": 2
    }}
    }}
    """

    # format responses
    response_list_str = "\n".join(f"{i+1}. {resp}" for i, resp in enumerate(responses))

    user_prompt = f"""
    **Context:**
    {context}

    **Responses:**
    {response_list_str}
    """
    return system_prompt, user_prompt

def bigcodebench_cluster_fn(context: str, responses: List[str]) -> tuple[str, str]:
    """
    LLM clustering prompt for grouping responses based on semantic and conceptual diversity.
    """
    n_responses = len(responses)

    system_prompt = f"""
    Your ONLY task is to cluster the {n_responses} responses into "k" buckets based solely on their core reasoning process.

    **INPUT FORMAT:**  
    You will receive:  
    1) A "Context" describing the task.  
    2) A numbered list of "Responses" from 1 to {n_responses}.  
       - Each response will typically contain two parts:
         - <high_level>...</high_level>: a 1–2 sentence summary of the intended reasoning strategy.
         - solution: the self-contained executable python code.

    Your job is ONLY to identify the underlying thought process, logic or conceptual approach of each response and assign each one a cluster ID according to the rules below.

    **HOW TO USE <high_level> AND solution:**
    - You MUST read both the <high_level> block and the detailed reasoning when deciding the cluster.
    - Treat <high_level> as a short, explicit summary of the solution strategy, but:
      - If <high_level> is missing, too generic (e.g. "solve the problem step by step"), or contradicts the actual implementation and algorithmic approach shown in the solution code,
        you MUST base your clustering on the **actual implementation** inside the solution block of the full response, not on the <high_level> block.
    - If two responses have slightly different wording in <high_level> but the actual reasoning in solution follows the same implementational approach, they MUST be in the same cluster.

    **CLUSTERING RULES:**  
    - Cluster strictly based on the response's logic, viewpoint, core idea, conceptual approach, or framework — NOT on structure, formatting, style or comments.
    - - Two responses belong in the SAME cluster if and only if they use the **same algorithmic or implementation strategy**, even if:
        - Variable names differ
        - Code structure differs
        - One implementation contains bugs or errors but has the same algorithmic approach as the other
        - One explanation is more verbose than another  
    - You are clustering **reasoning or implementation strategies**, as evidenced by the full response (<high_level> plus the reasoning in solution), not just the literal text inside <high_level>.
    - If all responses use the same logic or implementational approach, assign all of them cluster_id = 1.  
    - If responses differ in logic or implementational approach, they MUST be assigned different cluster IDs. 
    - If there is an implementation error for the same approach, put it in the SAME cluster. 
    - Do NOT infer or hallucinate thought processes, implementation or logic that is not explicitly present in the responses.

    **OUTPUT RULES (STRICT):**  
    1. Respond ONLY with a JSON object. No text outside the JSON is allowed.  
    2. The JSON must contain exactly {n_responses} keys: "1", "2", ..., "{n_responses}", where each key corresponds to the index of the response.  
    3. The value for each key must be a JSON object with EXACTLY two fields:  
        "chain_of_thought": string  
        "cluster_id": integer  
    4. The "chain_of_thought" MUST provide a concise description of the response's **implementational approach, conceptual approach, reasoning process, or core idea**.  
    - It MUST NOT include details specific to solving a particular task, step-by-step instructions, or content unrelated to understanding the approach.  
    - It MUST summarize the implementational logic, conceptual approach, or algorithmic approach, not the final answer or output.  
    5. The "cluster_id" MUST follow these rules:  
    - The "cluster_id" must be an integer between 1 and {n_responses}. 
    - Two responses must share the same cluster_id if and only if they follow the same underlying implementational, algorithmic or conceptual approach.  
    - Differences in variable names, style, comments, verbosity, formatting, correctness, or superficial content DO NOT require different clusters.  
    - If there are two responses with similar implementation but come to different outputs due to errors, they should not be put in different clusters. 
    - Any substantive difference in conceptual or implementational approach or reasoning MUST result in different clusters.  
    6. The JSON must contain NO additional fields, explanations, comments, or metadata.
    
    **Few-Shot Example 1:**

    **Context:**
    Write a function that computes the factorial of a non-negative integer n.

    **Responses:**
    1. <high_level>Compute factorial iteratively using a loop.</high_level>
    ```python
    def factorial(n: int) -> int:
        answer = 1
        for x in range(1, n+1):
            answer *= x
        return answer
    ```
    2. <high_level>Compute the factorial recursively using the recursion n! = n * (n-1)! and base case 0! = 1.</high_level>
    ```python
    def factorial(n: int) -> int:
        if n = 0: 
            return 1
        
        return n * factorial(n-1)
    ```
    3. <high_level>Compute the factorial iteratively by accumulating in a variable using a for loop.</high_level>
    ```python
    def factorial(n) -> int:
        final = 1
        for x in range(1, n+1): # loop to calculate factorial
            final *= x
        return final
    ```
    **Expected Output:**
    {{
    "1": {{
    "chain_of_thought": "Computes the factorial by iteratively multiplying numbers from 1 to n.",
    "cluster_id": 1
    }},
    "2": {{
    "chain_of_thought": "Computes the factorial using a recursive definition with a base case.",
    "cluster_id": 2
    }},
    "3": {{
    "chain_of_thought": "Computes the factorial by iteratively multiplying numbers from 1 to n.",
    "cluster_id": 1
    }}    
    }}
    **Few-Shot Example 2:**

    **Context:**
    Find the maximum value in a list of integers.

    **Responses:**
    1. <high_level>Iterate through the list and track the largest element seen so far.</high_level>
    ```python
    from typing import List

    def max_value(nums: List[int]) -> int:
        current_max = nums[0]
        for x in nums:
            if x > current_max:
                current_max = x
        return current_max
    ```
    2. <high_level>Sort the list and pick the last element.</high_level>
    ```python
    from typing import List

    def max_value(arr: List[int]) -> int:
        arr_sorted = sorted(arr)
        return arr_sorted[-1]
    ```
    3. <high_level>Use Python’s built-in max() function to find the largest element.</high_level>
    ```python
    from typing import List

    def max_value(values: List[int]) -> int:
        return max(values)
    ```
    **Expected Output:**
    {{
    "1": {{
    "chain_of_thought": "Finds the maximum by iterating through the list and keeping track of the largest value seen so far.",
    "cluster_id": 1
    }},
    "2": {{
    "chain_of_thought": "Finds the maximum by sorting the list and selecting the last element.",
    "cluster_id": 2
    }},
    "3": {{
    "chain_of_thought": "Finds the maximum by using Python's built-in max() function.",
    "cluster_id": 3
    }}
    }}
    **Few-Shot Example 3:**

    **Context:**
    Write a function that computes the sum of a list of integers.

    **Responses:**
    1. <high_level>Iterate through the list and maintain a running total.</high_level>
    ```python
    from typing import List

    def sum_list(nums: List[int]) -> int:
        total = 0
        for x in nums:
            total += x
        return total
    ```
    2. <high_level>Accumulate values in a loop.</high_level>
    ```python
    from typing import List

    def sum_list(arr: List[int]) -> int:
        ans = 0
        for i in range(len(arr):
            ans += arr[i]
        return ans
    ```
    3. <high_level>Iterate and add elements sequentially.</high_level>
    ```python
    from typing import List

    def sum_list(values: List[int]) -> int:
        result = 0
        for v in values:
            result = result + v
        return result
    ```
    **Expected Output:**
    {{
    "1": {{
    "chain_of_thought": "Computes the sum by iterating through the list and accumulating values in a running total.",
    "cluster_id": 1
    }},
    "2": {{
    "chain_of_thought": "Computes the sum by iterating through the list and accumulating values in a running total.",
    "cluster_id": 1
    }},
    "3": {{
    "chain_of_thought": "Computes the sum by iterating through the list and accumulating values in a running total.",
    "cluster_id": 1
    }}
    }}
    """

    # format responses
    response_list_str = "\n".join(f"{i+1}. {resp}" for i, resp in enumerate(responses))

    user_prompt = f"""
    **Context:**
    {context}

    **Responses:**
    {response_list_str}
    """
    return system_prompt, user_prompt
