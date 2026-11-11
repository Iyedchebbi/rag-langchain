
from deepagents import create_deep_agent
from langchain_openrouter import ChatOpenRouter
from langchain.messages import HumanMessage

from prompts import (
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
    CHUNK_ANALYST_INSTRUCTIONS,
)

from tools import search_documentation, backend


# 1. Configure maximum concurrent analysis
max_concurrent_analysts = 3


# 2. Combine all agent instructions
INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts
    )
)


# 3. Define the chunk analyst subagent
chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyse one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}


# 4. Initialize the language model
model = ChatOpenRouter(
    model="nvidia/nemotron-3.5-super-vl:free",
    temperature=1.0,
)


# 5. Create the main Deep Agent
agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)


# 6. Ask a question
example_query = "What skills existed"


# 7. Execute the agent
result = agent.invoke(
    {
        "messages": [
            HumanMessage(content=example_query)
        ]
    }
)


# 8. Display the answer
print(result["messages"][-1].content)
