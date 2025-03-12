from vllm import LLM, SamplingParams
import re
import vllm
import ray
from abc import ABC, abstractmethod
from dataclasses import dataclass
from beartype import beartype
from typing import *


import pymongo
import uuid
from datetime import datetime
import dotenv
import os, json
import random
dotenv.load_dotenv()



Message = Dict[str, str]
Reward = float
AgentState = Any  # State needed to track conversation progress

@dataclass
class AgentConversation:
    messages: List[Message]
    tokens_by_turn: List[Dict[str, Any]]
    first_prompt_tokens: List[int]
    all_output_tokens: List[int]

class AgentInterface(ABC):
    def __init__(
        self, 
        full_data: List[dict],
        sampling_params: SamplingParams, 
        vllm_engine: vllm.LLM, 
        **kwargs
    ):
        self.num_envs = len(full_data)
        self.full_data = full_data
        self.sampling_params = sampling_params
        self.vllm_engine = vllm_engine

        # As an example of full_data, for a given swe_bench task, it is a list of dicts, each with the following keys:
        # "repo", "instance_id", "base_commit", "patch", "test_patch", "problem_statement", "hints_text", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit"

    def generate_many(self) -> List[Tuple[AgentConversation, Reward]]:
        # Initialize states for all conversations
        states = [self.init_state(data) for data in self.full_data]
        all_messages = [list() for _ in range(self.num_envs)]
        active_indices = list(range(self.num_envs))

        tokens_by_turn = [list() for _ in range(self.num_envs)]
        total_tokens = [0 for _ in range(self.num_envs)]
        first_prompt_tokens = [None for _ in range(self.num_envs)]
        all_output_tokens = [[] for _ in range(self.num_envs)]
        # Continue until all conversations are complete
        while active_indices:
             # Get next prompts for all active conversations
            try:
                # Temporarily remove vllm engine before sending through Ray
                vllm_engine = self.vllm_engine
                self.vllm_engine = None
                all_prompts_and_states = ray.get([
                    get_next_prompt_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
                    for idx in active_indices
                ])
                self.vllm_engine = vllm_engine
            except Exception as e:
                self.vllm_engine = vllm_engine  # Restore in case of error
                logger.error(f"Error getting prompts: {str(e)}")
                raise
                
            active_conversations = []
            indices_to_remove = []
            for i, idx in enumerate(active_indices):
                result = all_prompts_and_states[i]
                if result is None:
                    active_indices.remove(idx)
                    continue
                    
                prompt, states[idx] = result
                if prompt is None or states[idx] is None:
                    # The environment is done, so we don't need to generate any more prompts
                    # active_indices.remove(idx)
                    indices_to_remove.append(idx)
                    continue
                if isinstance(prompt, list):
                    all_messages[idx].extend(prompt)
                elif isinstance(prompt, dict):  
                    all_messages[idx].append(prompt)
                else:
                    raise ValueError(f"Invalid prompt type: {type(prompt)}")
                active_conversations.append(all_messages[idx])

            for idx in indices_to_remove:
                active_indices.remove(idx)

            # Leave the loop if all conversations are done
            if len(active_conversations) == 0:
                break

            # Batch generate responses
            # TODO: Maybe use their tool API instead of handrolling?
            outputs = self.vllm_engine.chat(
                messages=active_conversations,
                sampling_params=self.sampling_params
            )

            # Process outputs and update states
            vllm_engine = self.vllm_engine
            self.vllm_engine = None
            all_is_done = ray.get([
                is_done_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
                for idx in active_indices
            ])
            self.vllm_engine = vllm_engine
            new_active_indices = []
            for i, output in enumerate(outputs):
                real_idx = active_indices[i]
                if total_tokens[real_idx] == 0:
                    first_prompt_tokens[real_idx] = output.prompt_token_ids

                input_tokens = output.prompt_token_ids[total_tokens[real_idx]:]
                output_tokens = output.outputs[0].token_ids
                
                generation_starter_text = output.prompt[-10:]
                if "think" in generation_starter_text.lower():
                    output_message = {"role": "assistant", "content": "<think>" + output.outputs[0].text}
                else:
                    output_message = {"role": "assistant", "content": output.outputs[0].text}

                all_messages[real_idx].append(output_message)
                tokens_by_turn[real_idx].append({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
                total_tokens[real_idx] += len(input_tokens) + len(output_tokens)
                
                all_tokens[real_idx] = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                all_tokens_text[real_idx] = output.prompt + output.outputs[0].text
                if not all_is_done[i]:
                    new_active_indices.append(real_idx)
            
            active_indices = new_active_indices
            
        # Calculate rewards for completed conversations
        results = []
        vllm_engine = self.vllm_engine
        self.vllm_engine = None
        all_rewards = ray.get([
            get_reward_remote.remote(self, messages=all_messages[idx], state=states[idx]) 
            for idx in range(self.num_envs)
        ])
        self.vllm_engine = vllm_engine
        
        # Create results list
        results_data = []
        for i, (messages, tokens_by_turn_one_env, fpt, aot) in enumerate(zip(all_messages, tokens_by_turn, first_prompt_tokens, all_tokens)):
            reward = all_rewards[i]
            conversation = AgentConversation(messages=messages, tokens_by_turn=tokens_by_turn_one_env, first_prompt_tokens=fpt, all_tokens=aot)
            results.append((conversation, reward))
        
        return results

    @abstractmethod
    def init_state(self, data: dict) -> AgentState:
        """Initialize the state for a new RL env, given a dict of the info in one element of the dataset"""
        pass

    @abstractmethod
    def get_next_prompt(self, messages: List[Message], state: AgentState) -> Optional[Tuple[Message, AgentState]]:
        """Input:
        - messages: the messages in the conversation
        - state: the state of the environment
        
        Output:
        - next_prompt: the next prompt to send to the model
        - next_state: the updated state of the environment
        
        Note: an output of None means that the environment is done and the agent should stop generating.
        
        Get the next prompt to send to the model and updated state.
        In this function, you should (1) use the model's last message to update the state. 
        Then (2) create the prompt to send to the model, which should probably incorporate observations about the environment.
        Finally, (3) return the next prompt for the model to send, along with the updated state."""
        pass

    @abstractmethod
    def is_done(self, messages: List[Message], state: AgentState) -> bool:
        """Determine if the conversation is complete"""
        pass

    @abstractmethod
    def get_reward(self, messages: List[Message], state: AgentState) -> Reward:
        pass


# Message = dict[str, str]

# Function to save a complete trajectory
def save_trajectory(messages, metadata=None):
    """
    Save a complete conversation trajectory to local file
    
    Parameters:
    - messages: list, the complete list of messages in the trajectory
    - metadata: dict, experiment details, agent config, etc.
    
    Returns:
    - trajectory_id: the ID of the saved trajectory
    """
    trajectory = {
        "trajectory_id": str(uuid.uuid4()),
        "messages": messages,
        "timestamp": datetime.now(),
        "metadata": metadata or {}
    }
    print(f"Saving trajectory to {trajectory['trajectory_id']}.json")
    with open(f"{trajectory['trajectory_id']}.json", "w") as f:
        json.dump(trajectory, f)


@beartype
class Tool(ABC):
    """
    Base class for a tool than an LLM agent can use.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the tool.
        To use the tool, the agent will have to write <name>arguments</name>, where name is the string this function returns.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        Returns the description of the tool.
        It will be included in the LLM agent's prompt.
        """
        pass

    @abstractmethod
    def argument_description(self) -> dict[str, str] | str | None:
        """
        Returns the descriptions of the arguments given to the tool in English.
        They will be included in the LLM agent's prompt.
        If the tool does not take arguments, returns `None`.
        If the tool takes one argument, returns a string, which is the description of the argument in English.
        If the tools takes multiple arguments, returns a dictionary mapping the names of the arguments to their descriptions in English.
        """
        pass

    @abstractmethod
    def _call(self, *args, **kwargs) -> str:
        """
        Override this method to execute the tool call.
        It should take no arguments, one positional argument, or keyword arguments if `self.argument_description()` is `None`, a `str`, or a `dict[str, str]`, respectively. In the latter case, the namse of the keyword arguments it takes should be the same as `self.argument_description().keys()`.
        It should return the output of the tool as a string.
        If the tool does not have an output, return something like "Tool call successful." or "".
        """
        pass

    def __post_init__(self) -> None:
        assert self.name() != "think", (
            "A tool cannot be named 'think' because this would conflict with DeepSeek-R1's <think>...</think> chain of thought tags."
        )

    def call_arguments_valid(self, arguments: dict[str, str] | str | None) -> bool:
        argument_description = self.argument_description()

        if argument_description is None:
            return arguments is None

        if isinstance(argument_description, str):
            return isinstance(arguments, str)

        if isinstance(argument_description, dict):
            return isinstance(arguments, dict) and set(arguments.keys()) == set(
                argument_description.keys()
            )

        assert False, (
            "Tool.argument_description should return an object of type dict[str, str] | str | None."
        )

    def __call__(self, arguments: dict[str, str] | str | None) -> str:
        if not self.call_arguments_valid(arguments):
            return f"An invalid combination of arguments was provided to the tool {self.name()}."

        try:
            if arguments is None:
                return self._call()
            if isinstance(arguments, str):
                return self._call(arguments)
            if isinstance(arguments, dict):
                return self._call(*arguments)
            return "Tool call failed."
        except Exception:
            return "Tool call failed."

    def usage_message(self) -> str:
        message = f"# Tool: {self.name()}\n"
        message += "\n"
        message += self.description() + "\n"
        message += "\n"
        message += "To use the tool, simply write the following:\n"
        message += "\n"

        argument_description = self.argument_description()

        if argument_description is None:
            message += f"<{self.name()}/>"
            return message

        if isinstance(argument_description, str):
            message += f"<{self.name()}>\n"
            message += argument_description + "\n"
            message += f"</{self.name()}>"
            return message

        if isinstance(argument_description, dict):
            message += f"<{self.name()}>\n"
            for argument_name, description in argument_description.items():
                message += f"<{argument_name}>{description}</{argument_name}>\n"
            message += f"</{self.name()}>"

        assert False, (
            "Tool.argument_description should return an object of type dict[str, str] | str | None."
        )


@beartype
def agent_instruction_message(prompt: str, tools: list[Tool], can_finish: bool) -> str:
    # message = "You are an agent which has access to tools. You have a goal which you should accomplish. In order to call tools, write tool calls, formatted exactly as in the examples. You will see the outputs of those tool calls in the next user message. If you want to do multiple things in sequence, you must do them one by one, returning control to the user after each thing, as you will only see the output of tool calls to do one thing in the next user message. Be thorough and do not give up - if you did something and it failed, you should fix it or do something eles that will achieve your goal that doesn't fail. When you think you are done with your goal, you should test whether you are actually done and everything you did actually works. If it doesn't, you should either fix it or start over with a different approach.\n"
    message = "You are a software engineering assistant. You are asked to write code to accomplish tasks. In order to write this code, and to debug, you are given access to various tools. Please make at least one tool call in each message. After you finish the message entirely, you will be shown the results of every tool call you made. Remember to be persistent, fast, and not to give up!"
    if can_finish:
        message += " If you are done, you should call the finish_tool. However, you should make sure that you have indeed accomplished the goal before calling finish_tool. If you realize that you have not accomplished it, you should try again and not call finish_tool until you are sure you accomplished your goal.\n"
    message += "\n"
    message += "You can use the following tools. To use a tool, you must format the tool call exactly as it is formatted below, or the tool call will not work. Remember that if you wish to write to a file, you must use one of the provided tools, e.g. using the <bash> tool to echo to a file. I REPEAT: JUST THINKING ABOUT BASH CODE WILL NOT WRITE IT TO A FILE. The best thing to do would be to draft what you want to write to a file, and then use the <bash> tool to echo or cat the text to a file. Also remember that you should only think about a tool in <> brackets if you plan to use it.\n"
    message += "\n"
    message += "\n\n".join(tool.usage_message() for tool in tools)
    message += "\n"
    message += "\n"
    message += "# Your Goal\n"
    message += "\n"
    message += prompt

    return message


@beartype
@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, str] | str | None


@beartype
@dataclass
class ToolCallResult:
    finish_tool_called: bool
    tool_name: str
    tool_response: str


@beartype
def call_tool(tool_call: ToolCall, all_tools: list[Tool]) -> ToolCallResult:
    tool = find_tool_by_name(tool_call.tool_name, all_tools)

    if tool is None:
        return ToolCallResult(
            finish_tool_called=False,
            tool_name=tool_call.tool_name,
            tool_response=f"Error: There is no tool named '{tool_call.tool_name}'.",
        )

    if isinstance(tool, FinishTool):
        return ToolCallResult(
            finish_tool_called=True,
            tool_name=tool_call.tool_name,
            tool_response="Finished.",
        )

    tool_response = tool(tool_call.arguments)

    return ToolCallResult(
        finish_tool_called=False,
        tool_name=tool_call.tool_name,
        tool_response=tool_response,
    )


@beartype
def find_tool_by_name(tool_name: str, tools: list[Tool]) -> Tool | None:
    matching_tools = [tool for tool in tools if tool.name() == tool_name]

    if len(matching_tools) == 0:
        return None

    if len(matching_tools) > 1:
        print(
            f"WARNING: Found more than one tool with name '{tool_name}'. Tool names should be unique."
        )

    return matching_tools[0]


@beartype
@dataclass
class FinishTool(Tool):
    """
    Special tool to be used by the agent if it finished its task.
    It is treated specially by the agent loop.
    """

    def name(self) -> str:
        return "finish"

    def description(self) -> str:
        return "Call this tool when you are finished. Only call it when you are very sure that you indeed finished your task correctly, and made the best effort possible to verify that what you did is indeed correct."

    def argument_description(self) -> dict[str, str] | str | None:
        return None

    def _call(self) -> str:
        assert False, "Do not call FinishTool.__call__."


@beartype
@beartype
def extract_tool_calls(llm_message: str) -> list[ToolCall]:
    """
    Extracts tool calls of the following forms:
      1) <tool_name/>               (no arguments)
      2) <tool_name>single_argument</tool_name>   (string argument)
      3) <tool_name><arg>val</arg>...</tool_name> (dictionary arguments)
      4) ```tool_name
         single_argument
         ```                (backtick format)
         (Similarly, can also hold <arg>val</arg> blocks interpreted as a dict.)

    Returns a list of ToolCall objects.
    """

    # We define a single combined pattern with alternation to capture:
    #   * The triple-backtick form: ```tool_name ... ```
    #   * The self-closing XML form: <tool_name/>
    #   * The open/close XML form: <tool_name>...</tool_name>
    #
    # Explanation of capture groups:
    #   (1) triple_name, (2) triple_content
    #   (3) self_closing_name
    #   (4) open_close_name, (5) inner_content
    pattern = re.compile(
        r"""
        # 1) The triple-backtick form: ```tool_name ... ```
        ```(\w+)\s+(.*?)```

        |

        # 2) A self-closing tag: <tool_name/>
        <(\w+)\s*/>

        |

        # 3) An open/close pair: <tool_name>...</tool_name>
        <(\w+)>(.*?)</\4>
        """,
        re.DOTALL | re.VERBOSE,
    )

    # Pattern used to detect dictionaries of the form <key>value</key>
    # repeated multiple times.
    dict_pattern = re.compile(r"^(?:\s*<(\w+)>(.*?)</\1>\s*)+$", re.DOTALL)

    matches = pattern.findall(llm_message)
    tool_calls: list[ToolCall] = []

    for (
        triple_name,
        triple_content,
        self_closing_name,
        open_close_name,
        inner_content,
    ) in matches:
        if triple_name:
            # Matched the triple-backtick pattern: ```tool_name ... ```
            tool_name = triple_name
            content_stripped = triple_content.strip()

            # Check if content is a series of <arg>value</arg> blocks
            if dict_pattern.fullmatch(content_stripped):
                pairs = re.findall(r"<(\w+)>(.*?)</\1>", content_stripped, re.DOTALL)
                arguments_dict = {k: v.strip() for k, v in pairs}
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=arguments_dict)
                )
            else:
                # Otherwise treat as a single string argument
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=content_stripped)
                )

        elif self_closing_name:
            # Matched the self-closing pattern <tool_name/>
            tool_calls.append(ToolCall(tool_name=self_closing_name, arguments=None))

        else:
            # Matched the open/close pattern <tool_name>...</tool_name>
            tool_name = open_close_name
            content_stripped = inner_content.strip()

            # Check if content is purely made up of <arg>value</arg> blocks
            if dict_pattern.fullmatch(content_stripped):
                # Parse into a dictionary
                pairs = re.findall(r"<(\w+)>(.*?)</\1>", content_stripped, re.DOTALL)
                arguments_dict = {k: v.strip() for k, v in pairs}
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=arguments_dict)
                )
            else:
                # Otherwise treat as a single string argument
                tool_calls.append(
                    ToolCall(tool_name=tool_name, arguments=content_stripped)
                )

    # Filter out tool calls whose name is "think"
    # (Some systems reserve <think>...</think> as scratchpad, etc.)
    tool_calls = [tc for tc in tool_calls if tc.tool_name != "think"]

    return tool_calls


@beartype
@dataclass
class BasicAgentState:
    data: dict
    tools: list[Tool]
    finished: bool = False
    step: int = 0


@beartype
class BasicAgentEnv(AgentInterface):
    def __init__(
        self,
        *args,
        max_steps: int = 4,
        can_finish: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.can_finish = can_finish

    @abstractmethod
    def get_tools(self, data: dict) -> list[Tool]:
        pass

    @abstractmethod
    def get_prompt(self, data: dict) -> str:
        pass

    def init_state(self, data: dict) -> BasicAgentState:
        tools = self.get_tools(data)
        if self.can_finish:
            tools.append(FinishTool())

        return BasicAgentState(data=data, tools=tools)

    def get_next_prompt(
        self, messages: list[Message], state: BasicAgentState
    ) -> tuple[Message | None, BasicAgentState]:
        assert not state.finished

        if len(messages) == 0:
            prompt = agent_instruction_message(
                prompt=self.get_prompt(state.data),
                tools=state.tools,
                can_finish=self.can_finish,
            ) 
            message = {"role": "user", "content": prompt}
            return (message, state)

        if state.step >= self.max_steps:
            state.finished = True
            return (None, state)
        state.step += 1

        assert messages[-1]["role"] == "assistant"
        tool_calls: list[ToolCall] = extract_tool_calls(messages[-1]["content"])
        # tool_calls.append(ToolCall(tool_name="bash", arguments="echo hello"))

        for tool_call in tool_calls:
            tool_call_result: ToolCallResult = call_tool(
                tool_call, all_tools=state.tools
            )

            if tool_call_result.finish_tool_called:
                state.finished = True
                return (None, state)

            messages.append(
                {
                    "role": "tool",
                    # "tool_call_id": tool_call_result.tool_name,
                    "content": tool_call_result.tool_response,
                }
            )
            
        if len(tool_calls) == 0:
            messages.append(
                {
                    "role": "user",
                    "content": "Please call at least one tool in each message. You may have meant to call a tool, but forgot to put the tool call in the correct tags. Remember also that in order to write you should echo the code you write to a file.",
                }
            )
        # else:
        #     messages.append({"role": "assistant", "content": ""}) # Hack to get the tokenizer to add the <think>...</think> tags

        last_message = messages.pop()

        return (last_message, state)

    def is_done(self, messages: list[Message], state: BasicAgentState) -> bool:
        finished = state.finished
        # if finished:
        #     # Save 1 in 10 trajectories
        #     if random.random() < 0.1:
        #         save_trajectory(messages, metadata={"task": "bash_bench"})
        return finished


@beartype
def print_agent_loops(
    conversations: list[list[Message]], rewards: list[float | None] | None = None
) -> None:
    if rewards is None:
        rewards = [None for _ in range(len(conversations))]

    print("+" * 100)
    print("+" * 100)
    print("+" * 100)
    print("PRINTING AGENT LOOPS")
    for i, (conversation, reward) in enumerate(
        zip(conversations, rewards, strict=True)
    ):
        print("+" * 100)
        print("+" * 100)
        print("+" * 100)
        print(f"AGENT LOOP {i + 1} ------- REWARD: {reward}")
        for message in conversation:
            print("=" * 100)
            for field, value in message.items():
                if field == "content":
                    continue
                print(field.upper(), value)
            print("CONTENT:", message["content"])
