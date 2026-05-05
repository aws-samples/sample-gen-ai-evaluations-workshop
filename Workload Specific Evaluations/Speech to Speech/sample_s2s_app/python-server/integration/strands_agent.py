from strands import Agent, tool
from strands.models import BedrockModel
import datetime
import boto3 
import os
import json
import requests
import re

@tool
def date_time() -> str:
    """Get date and time information"""
    return f"The date and time is {str(datetime.datetime.now())}"

@tool
async def slow_tool(query: str = "") -> dict:
    """run the slow tool"""
    import asyncio
    # Simulate a slow API call with 20-second delay
    await asyncio.sleep(20)
    # Return hardcoded weather data
    return {
        "slow_tool_result": "done"

    }    

class StrandsAgent:

    def __init__(self):

        session = boto3.Session(
            region_name='us-east-1',
        )
        # Specify Bedrock LLM for the Agent
        bedrock_model = BedrockModel(
            model_id="amazon.nova-lite-v1:0",
            boto_session=session
        )
        # Create a Strands Agent
        self.agent = Agent(
            tools=[date_time, slow_tool],
            model=bedrock_model,
            system_prompt="You are a chat agent tasked with answering time and running the slow tool. Please include your response within the <response></response> tag."
        )


    '''
    Send the input to the agent, allowing it to handle tool selection and invocation. 
    The response will be generated after the selected LLM performs reasoning. 
    This approach is suitable when you want to delegate tool selection logic to the agent, and have a generic toolUse definition in Sonic ToolUse.
    Note that the reasoning process may introduce latency, so it's recommended to use a lightweight model such as Nova Lite.
    Sample parameters: input="What time is it?"
    '''
    def query(self, input):
        output = str(self.agent(input))
        if "<response>" in output and "</response>" in output:
            match = re.search(r"<response>(.*?)</response>", output, re.DOTALL)
            if match:
                output = match.group(1)
        elif "<answer>" in output and "</answer>" in output:
            match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
            if match:
                output = match.group(1)
        return output

    '''
    Invoke the tool directly and return the raw response without any reasoning.
    This approach is suitable when tool selection is managed within Sonic and the exact toolName is already known. 
    It offers lower query latency, as no additional reasoning is performed by the agent.
    Sample parameters: tool_name="search_places", input="largest zoo in Seattle"
    '''
    def call_tool(self, tool_name, input):
        if isinstance(input, str):
            input = json.loads(input)
        if "query" in input:
            input = input.get("query")

        tool_func = getattr(self.agent.tool, tool_name)
        return tool_func(query=input)

    def close(self):
        print("Closing Strands Agent")