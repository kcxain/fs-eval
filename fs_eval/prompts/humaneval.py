# system_prompt = """You are a Software Engineer."""
system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. \u00a0Now the user asks you to write python code. After thinking, when you finally reach a conclusion, enclose the final python code in ```python ``` within <answer> </answer> tags. i.e., <answer> ```python\n ... ``` </answer>.\n"""


user_prompt = """Read the following function signature and docstring, and fully implement the function described.
```python
{function_signature}
```
"""

# user_prompt = """You need to write the corresponding Python code based on the given problem, and your answer should include the corresponding Python code enclosed in ```python ```. Note that:
# - Think carefully before providing your final answer.
# - Make sure to use the precise function signature in your answer if it's provided in the query.

# Now you are given the below instruction:
# {problem}
# The function signature is:
# ```python
# {function_signature}
# ```
# Please respond with thinking process and your final answer.
# """

user_prompt_oneshot = """You need to write the corresponding Python code based on the given problem, and your answer should include the corresponding Python code enclosed in ```python ```. Note that:
- Think carefully before providing your final answer.
- Make sure to use the precise function signature in your answer if it's provided in the query.

Here is an basic example:
===== Example Start =====
Given instruction:

Given a list of numbers, propose code that returns the number of distinct elements in the list. Additionally, ensure that the solution has a time complexity of O(nlogn), where n is the length of the list.
The function definition of the code should be:
```python
def count_distinct_elements(nums):
```
Think carefully before providing your final answer and please do not include any additional headers in your answer.

You should respond with:
<think>
[Your thinking process]
</think>
Then the final answer:

```python
def count_distinct_elements(lst):
    if not lst:
        return 0
    lst.sort()
    count = 1
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            count += 1
    return count
```
===== Example End =====

Now you are given the below instruction:
{problem}
The function signature is:
```python
{function_signature}
```
Please respond with thinking process and your final answer.
"""
