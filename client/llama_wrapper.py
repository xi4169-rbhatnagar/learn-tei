import abc
import dataclasses
import enum
from typing import Generator, Any, Sequence
from typing import Optional, List, Dict, Iterable

import openai
from llama_index.core.base.llms.types import LLMMetadata, ChatMessage, ChatResponse, CompletionResponseAsyncGen, \
    ChatResponseAsyncGen, CompletionResponse, CompletionResponseGen, ChatResponseGen
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai.utils import to_openai_message_dicts
from llama_index_client import MessageRole
from openai.types.chat import ChatCompletionChunk


def user(message) -> dict:
    return dict(role='user', content=message)


Purpose = Dict[str, str]
PURPOSE_CHAT = {
    "level1": "CHAT",
    "level2": "GENERATIONS",
    "level3": "NA",
}


class OpenAIInterface(abc.ABC):
    @abc.abstractmethod
    def query_model(
            self, model: str, p: Purpose, messages: List[Dict], stream: bool
    ) -> Optional[str | Iterable[Dict[str, str]]]:
        """
        :param stream: whether or not to stream the response
        :param model: name of the model to query
        :param messages: list of prompts (user, system prompts)
        :param p: Purpose based on which a querying context is established
        :return: response from the requested llm model
        """
        raise NotImplementedError


class OpenAIModelInterface(abc.ABC):
    """
    Wrapper over OpenAIInterface that helps abstract away the underlying model being used for querying
    """

    def __init__(self, model: str, context_size: int, openai_interface: OpenAIInterface):
        self.model = model
        self.context_size = context_size
        self.openai_interface = openai_interface

    def query(self, p: Purpose, messages: List[Dict], stream: bool = False) -> Optional[str | Iterable[Dict[str, str]]]:
        return self.openai_interface.query_model(self.model, p, messages, stream)


class Inferix(OpenAIInterface):
    def __init__(self, base_url: str, api_key):
        self._base_url = base_url
        self._api_key = api_key

    def query_model(
            self, model: str, purpose: Purpose, messages: List[Dict], stream: bool
    ) -> Optional[str | Iterable[Dict[str, str]]]:
        """Method to ask a user-given model a question"""
        # Create the OpenAI client
        client = self._create_openai_client()

        # Get the chat completion
        chat_completion: openai.ChatCompletion = client.chat.completions.create(
            messages=messages,
            temperature=0.1,
            top_p=0.95,
            model=model,
            stream=stream,
        )

        if stream:
            return self._get_response_stream(chat_completion)
        else:
            # Return the first completion content
            return chat_completion.choices[0].message.content

    @staticmethod
    def _get_response_stream(response: Iterable[ChatCompletionChunk]) -> Iterable[Dict[str, str]]:
        for chunk in response:
            yield dict(data=chunk.choices[0].delta.content)

    def _create_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url=self._base_url, api_key=self._api_key)


class AIObjType(enum.Enum):
    GenAIOS = enum.auto()
    Inferix = enum.auto()


@dataclasses.dataclass
class Response:
    response_gen: Generator[str, None, None]


class LlamaWrapper(LLM):
    query_msg: List[Dict] = [],
    ai_obj: OpenAIModelInterface = None

    def __init__(self, ai_obj: OpenAIModelInterface, **kwargs: Any):
        super().__init__(**kwargs)
        self.ai_obj = ai_obj

    def set_query_msg(self, msgs: List[Dict]):
        self.query_msg = msgs

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.ai_obj.context_size,
            is_chat_model=False,
            num_output=-1,
            is_function_calling_model=False,
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        msg_dicts = self._form_message_dicts(messages, self.query_msg)
        output: Iterable[Dict[str, str]] = self.ai_obj.query(PURPOSE_CHAT, msg_dicts, stream=True)
        content = ""
        for chunk in output:
            content += chunk
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                delta=chunk.get('data'),
            )

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        output: Iterable[Dict[str, str]] = self.ai_obj.query(PURPOSE_CHAT, messages=[user(prompt)], stream=True)
        text = ""
        for chunk in output:
            data = chunk.get('data')
            text += data
            yield CompletionResponse(text=text, delta=data)

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        output: Iterable[Dict[str, str]] = self.ai_obj.query(PURPOSE_CHAT, messages=[user(prompt)], stream=True)
        text = ""
        for chunk in output:
            data = chunk.get('data')
            text += data
        return CompletionResponse(text=text)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise NotImplementedError

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    @staticmethod
    def _form_message_dicts(messages: Sequence[ChatMessage], query_msg) -> List[Dict]:
        message_dicts = to_openai_message_dicts(messages)

        extend_sys_msg = LlamaWrapper._make_out_system_prompt()
        message_dicts[0]["content"] = message_dicts[0]["content"] + "\n" + extend_sys_msg
        return [message_dicts[0]] + query_msg + [message_dicts[1]]

    @staticmethod
    def _make_out_system_prompt():
        system_msg = """Provide answers in Markdown format and include references to any files used to obtain the information. Structure the response as follows:

    - Begin with an `##` section titled "Details" for a summary of the code's purpose.
    - Follow with a section for a detailed explanation that addresses the user's specific query.
    - Use a bullet list key points highlighting important aspects of the code.
    - Include code snippets to illustrate examples where applicable.
    - Add a section for further clarification on complex concepts or topics.
    - Conclude with an `##` section titled "References" listing the files referenced in your response.
        """
        return system_msg
