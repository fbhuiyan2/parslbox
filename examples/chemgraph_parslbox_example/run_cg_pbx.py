import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

prompt = "Add the following jobs to ParslBox: test1/, test2/, test3, test4/ and test5/. These are LAMMPS calculations on Polaris. Use 'mcp_test' as tag for test1 and test5, and mcp_prod for test2, test3 and test4."

# prompt = "Remove job 2 and 4 from Parslbox"

# prompt = "Update Job 5 to use 7 GPUs. It should be a VASP app and using 2 nodes"

# prompt = "Filter all jobs with mcp_prod tag"

""" qsub doesn't work yet.
#prompt = "Submit job 1 to the queue. This is on Polaris. Job name is test. Queue is debug-scaling. Use 2 nodes, 5 mins queue. The project is IQC and the file system is //lus/grand. App is lammps with production task"
"""
client = MultiServerMCPClient(
    {
        "ParslBox Tool MCP": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:9005/mcp/",
        },
    }
)


async def bootstrap():
    tools = await client.get_tools()
    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        structured_output=False,
        return_option="state",
        tools=tools,
    )
    result = await cg.run(prompt)

    print(result)


asyncio.run(bootstrap())
