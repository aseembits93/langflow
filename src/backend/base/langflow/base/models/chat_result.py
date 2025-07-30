import warnings

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langflow.field_typing.constants import LanguageModel
from langflow.schema.message import Message


def build_messages_and_runnable(
    input_value: str | Message, system_message: str | None, original_runnable: LanguageModel
) -> tuple[list[BaseMessage], LanguageModel]:
    # Handle string input fast path first (most common and fastest)
    if isinstance(input_value, str):
        messages = [HumanMessage(content=input_value)]
        if system_message:
            # Prepend system message
            messages.insert(0, SystemMessage(content=system_message))
        return messages, original_runnable

    # input_value is Message, handle carefully
    messages: list[BaseMessage] = []
    system_message_added = False
    runnable = original_runnable

    # At this point input_value is assumed Message and not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(input_value, "prompt") and "prompt" in input_value:
            prompt = input_value.load_lc_prompt()
            if system_message:
                prompt.messages = [
                    SystemMessage(content=system_message),
                    *prompt.messages,  # type: ignore[has-type]
                ]
                system_message_added = True
            runnable = prompt | runnable
        else:
            messages.append(input_value.to_lc_message())

    if system_message and not system_message_added:
        messages.insert(0, SystemMessage(content=system_message))

    return messages, runnable


def get_chat_result(
    runnable: LanguageModel,
    input_value: str | Message,
    system_message: str | None = None,
    config: dict | None = None,
    *,
    stream: bool = False,
):
    # Fast null/empty input check before anything else
    if not input_value and not system_message:
        msg = "The message you want to send to the model is empty."
        raise ValueError(msg)

    messages, runnable = build_messages_and_runnable(
        input_value=input_value, system_message=system_message, original_runnable=runnable
    )

    inputs = messages if messages else {}

    try:
        if config:
            output_parser = config.get("output_parser")
            if output_parser is not None:
                runnable = runnable | output_parser

            run_name = config.get("display_name", "")
            project_name_func = config.get("get_project_name", None)
            if project_name_func is not None:
                project_name = project_name_func()
            else:
                project_name = ""
            callbacks_func = config.get("get_langchain_callbacks", None)
            if callbacks_func is not None:
                callbacks = callbacks_func()
            else:
                callbacks = []

            runnable = runnable.with_config(
                {
                    "run_name": run_name,
                    "project_name": project_name,
                    "callbacks": callbacks,
                }
            )

        if stream:
            return runnable.stream(inputs)
        message = runnable.invoke(inputs)
        return message.content if hasattr(message, "content") else message
    except Exception as e:
        if config:
            get_exc_msg = config.get("_get_exception_message")
            if get_exc_msg:
                message = get_exc_msg(e)
                if message:
                    raise ValueError(message) from e
        raise
