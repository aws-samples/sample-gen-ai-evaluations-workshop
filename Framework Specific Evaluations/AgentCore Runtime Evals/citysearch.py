from botocore.config import Config
from ddgs import DDGS
from strands import Agent, tool
from strands.models import BedrockModel

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from boto3.session import Session

boto_session = Session()
region = boto_session.region_name

# Custom config for Bedrock — short timeouts, no retries
quick_config = Config(
    connect_timeout=5,
    read_timeout=20,
    retries={"max_attempts": 0}
)


@tool
def web_search(topic: str) -> str:
    """Search the web for a given topic using DDGS metasearch.
    Return a string listing the top results including the url, title, and description of each result.
    """
    import subprocess, json, sys
    try:
        result = subprocess.run(
            [sys.executable, '-c',
             'import json; from ddgs import DDGS; '
             f'r=DDGS(timeout=4).text({json.dumps(topic)}, max_results=3, backend="html"); '
             'print(json.dumps(r))'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return f"Search error: {result.stderr.strip()}"
        results = json.loads(result.stdout)
        if not results:
            return "No search results found"
        result_string = ""
        for i, r in enumerate(results):
            result_string += f"Result {i+1}: {r.get('title', 'No title')}\nURL: {r.get('href', 'No URL')}\nSnippet: {r.get('body', 'No description')}\n\n"
        return result_string
    except subprocess.TimeoutExpired:
        return "Search timed out after 10 seconds"
    except Exception as e:
        return f"Search error: {str(e)}"


SYSTEM_PROMPT = """You are a helpful city information assistant.

You help users find population and land area data for US cities.
Always use the web_search tool to look up current data — do not guess.
Be concise, professional, and friendly.

After your response, also include your answer in 'pop' and 'area' XML tags
for programmatic processing.
The values in the XML tags should only be numbers, no words or commas."""

chatbot_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
    boto_client_config=quick_config
)
chatbot = Agent(tools=[web_search], model=chatbot_model, system_prompt=SYSTEM_PROMPT)

# Initialize the AgentCore Runtime App
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """AgentCore Runtime entrypoint function"""
    user_input = payload.get("prompt", "")
    response = chatbot(user_input)
    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
