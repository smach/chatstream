from __future__ import annotations

from typing import Literal, TypedDict

OpenAiModel = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]

# Add updated token limits
openai_model_context_limits: dict[OpenAiModel, int] = {
    "gpt-3.5-turbo": 16384,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 163846,
    "gpt-3.5-turbo-0301": 16384,
    "gpt-3.5-turbo-16k-0613": 16384,
    "gpt-4": 16384,
    "gpt-4-0613": 16384,
    "gpt-4-0314": 16384,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-1106": 16385,
}

openai_models: list[OpenAiModel] = list(openai_model_context_limits)


class Usage(TypedDict):
    completion_tokens: int  # Note: this doesn't seem to be present in all cases.
    prompt_tokens: int
    total_tokens: int


class ChatMessage(TypedDict):
    content: str
    role: Literal["system", "user", "assistant"]


class ChoiceDelta(TypedDict):
    content: str


class ChoiceBase(TypedDict):
    finish_reason: Literal["stop", "length"] | None
    index: int


class ChoiceNonStreaming(ChoiceBase):
    message: ChatMessage


class ChoiceStreaming(ChoiceBase):
    delta: ChoiceDelta


class ChatCompletionBase(TypedDict):
    id: str
    created: int
    model: str


class ChatCompletionNonStreaming(TypedDict):
    object: Literal["chat.completion"]
    choices: list[ChoiceNonStreaming]
    usage: Usage


class ChatCompletionStreaming(ChatCompletionBase):
    object: Literal["chat.completion.chunk"]
    choices: list[ChoiceStreaming]
