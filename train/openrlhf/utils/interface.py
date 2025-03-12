from abc import ABC, abstractmethod
from typing import *
import vllm
from vllm import SamplingParams
from dataclasses import dataclass

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
            active_conversations = []
            for idx in active_indices:
                #TODO:
                prompt, states[idx] = self.get_next_prompt(all_messages[idx], states[idx])
                if prompt is None or states[idx] is None:
                    # The environment is done, so we don't need to generate any more prompts
                    active_indices.remove(idx)
                    continue
                all_messages[idx].append(prompt)
                active_conversations.append(all_messages[idx])

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
            new_active_indices = []
            for i, output in enumerate(outputs):
                real_idx = active_indices[i]
                if total_tokens[real_idx] == 0:
                    first_prompt_tokens[real_idx] = output.prompt_token_ids

                input_tokens = output.prompt_token_ids[total_tokens[real_idx]:]
                output_tokens = output.outputs[0].token_ids
                output_message = {"role": "assistant", "content": output.outputs[0].text}
                
                all_messages[real_idx].append(output_message)
                tokens_by_turn[real_idx].append({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
                total_tokens[real_idx] += len(input_tokens) + len(output_tokens)
                
                all_output_tokens[real_idx] = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    
                if not self.is_done(all_messages[real_idx], states[real_idx]):
                    new_active_indices.append(real_idx)
            active_indices = new_active_indices
        # Calculate rewards for completed conversations
        results = []
        for messages, tokens_by_turn, state, fpt, aot in zip(all_messages, tokens_by_turn, states, first_prompt_tokens, all_output_tokens):
            reward = self.get_reward(messages, state)
            conversation = AgentConversation(messages=messages, tokens_by_turn=tokens_by_turn, first_prompt_tokens=fpt, all_output_tokens=aot)
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