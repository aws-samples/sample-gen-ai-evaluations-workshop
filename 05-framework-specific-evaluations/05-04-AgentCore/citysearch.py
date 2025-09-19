from botocore.config import Config
import requests
from ddgs import DDGS
from strands import Agent, tool
from strands.models import BedrockModel
import pandas as pd
import re

#A custom config for Bedrock to only allow short connections - for our demo we expect all calls to be fast.
#here we turn off retries, and we time out after 20 seconds.
quick_config = Config(
    connect_timeout=5,
    read_timeout=20,
    retries={"max_attempts": 0}
)

from bedrock_agentcore.runtime import (
    BedrockAgentCoreApp,
)  #### AGENTCORE RUNTIME - LINE 1 ####
from strands import Agent
from strands.models import BedrockModel

from boto3.session import Session
boto_session = Session()
region = boto_session.region_name

@tool
def web_search(topic: str) -> str:
    """Search Duck Duck go Service for a given topic.
    Return a string listing the top 5 results including the url, title, and description of each result.
    """
    try:
        results = DDGS().text(topic, region=region, max_results=5)
        return results if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

#Create the chatbot.  We'll use Nova Micro to optimize for latency, cost, and capacity
chatbot_model_name = "us.amazon.nova-micro-v1:0"
#add custom timeout for the model, to keep the tool from hanging or retrying too much.
chatbot_model = BedrockModel(
    model_id=chatbot_model_name,
    boto_client_config=quick_config    
)
chatbot = Agent(tools=[web_search], model=chatbot_model)
#Call the chat bot with a simple request.
prompt = """How many people live in New York, and what's the area of the city in square miles?
After you respond, also include your answer in 'pop' and 'area' XML tags, for programatic processing.
The values in the XML tags should only be numbers, no words or commas."""
chatbot_response = chatbot(prompt)

# Initialize the AgentCore Runtime App
app = BedrockAgentCoreApp()  #### AGENTCORE RUNTIME - LINE 2 ####

@app.entrypoint  #### AGENTCORE RUNTIME - LINE 3 ####
def invoke(payload):
    """AgentCore Runtime entrypoint function"""
    user_input = payload.get("prompt", "")

    # Invoke the agent
    response = chatbot(user_input)
    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()  #### AGENTCORE RUNTIME - LINE 4 ####
