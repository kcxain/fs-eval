# system_prompt = """You are a Hardware Engineer."""
system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. \u00a0Now the user asks you to write verilog code. After thinking, when you finally reach a conclusion, enclose the final verilog code in ```verilog ``` within <answer> </answer> tags. i.e., <answer> ```verilog\n module top_module(in, out, ...) ... ``` </answer>.\n"""

user_prompt = (
    "{question}\nThe module head of the code should be:\n```verilog\n{module_head}\n```"
)

# user_prompt = """You need to write the corresponding Verilog code based on the given problem, and your output should only include the corresponding Verilog code enclosed in ```verilog ```. Note that:
# - Think carefully before providing your final answer.
# - Make sure to use the precise module signature in your response if it's provided in the query.

# Now you are given the below instruction:

# {question}
# The module head of the code should be:
# ```verilog
# {module_head}
# ```

# Please respond with thinking process and your final answer.
# """

user_prompt_oneshot = """You need to write the corresponding Verilog code based on the given problem, and your output should only include the corresponding Verilog code enclosed in ```verilog ```. Note that:
- Think carefully before providing your final answer.
- Make sure to use the precise module signature in your response if it's provided in the query.

Here is an basic example:

===== Example Start =====
Given instruction:

Design a 4-bit synchronous down counter that decrements by 1 on each positive edge of the clock signal. When the reset signal is active high, the counter should be reset to its maximum value.
The module head of the code should be:
```verilog
module fbitsyncdw(clk,rst,count);
```
Think carefully before providing your final answer.

You should respond with:
<think>
[Your thinking process]
</think>
Then the final answer:
```verilog
module fbitsyncdw(clk,rst,count);
input clk,rst;
output reg[3:0]count;
always @(posedge clk or posedge rst ) begin
    if(rst)
    count<=4'hF;
    else
    count<=count-1'b1;
end
endmodule
```
===== Example End =====

Now you are given the below instruction:

{question}
The module head of the code should be:
```verilog
{module_head}
```

Please respond with thinking process and your final answer.
"""
