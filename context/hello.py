import nest_asyncio
from pydantic import BaseModel
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig
)
from agents import set_default_openai_client, set_tracing_disabled
import asyncio
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, RunContextWrapper, function_tool
from dotenv import load_dotenv
from dataclasses import dataclass
import os

load_dotenv()


nest_asyncio.apply()

gemini_api_key = os.getenv("GEMINI_API_KEY")


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_default_openai_client(external_client)
set_tracing_disabled(True)


@dataclass
class UserInfo:
    name: str
    uid: int

@function_tool
async def greet_user(context: RunContextWrapper[UserInfo], greeting: str) -> str:
  """Greets the User with their name.
  Args:
    greeting: A specialed greeting message for user
  """
  name = context.context.name
  return f"Hello {name}, {greeting}"

async def main():
    user_info = UserInfo(name="sadiq", uid=123)

    agent = Agent[UserInfo](
        name="Assistant",
        tools=[greet_user],
        model=model,
        instructions="Always greet the user using <function_call>greet_user</function_call> and welcome them to Panaversity"
    )

    result = await Runner.run(
        starting_agent=agent,
        input="Hello",
        context=user_info,
    )

    print(result.final_output)

asyncio.run(main())