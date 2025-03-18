import os
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider
provider =  AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
) 

# Step 2: 
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=provider
)

# Step 3: Defined at run level
run_config = RunConfig(
    model=model,
    model_provider= provider,
    tracing_disabled=True 
)


# Step 4 Create an Agent 
agent1 = Agent(
    name="General support agent",
    instructions="You are a helpfull assistence ",
    model=model
)



@cl.on_chat_start
async def handle_chart_start():
    cl.user_session.set("history", [])
    await cl.Message("Hello! Im helpfull Agent. How can i help You").send()



@cl.on_message
async def handle_message(message: cl.message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    # Standard Interface [{"role": "user", "content": "Hello"}], {"role": "assistence", "content": "Hello How Can I Help You today"}
    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.get("history", history)
    #wait cl.Message(content=result.final_output).send()  
