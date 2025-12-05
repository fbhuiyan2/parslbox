# Using ChemGraph + ParslBox MCP via Port-Forwarding on Polaris (ALCF)

This directory provides an example of how to use ParslBox **MCP (Model Control Protocol)** with ChemGraph using port-forwarding on **Polaris at ALCF**. The instructions below guide you through launching the MCP server and connecting ChemGraph to it.

## Prerequisites

- ChemGraph and ParslBox installed. In this example, we will have ChemGraph installed in **chemgraph_env** and ParslBox installed in **parslbox_env**.
- `OPENAI_API_KEY` set (or enter interactively when running ChemGraph)

## Step-by-Step Instructions

### 1. Secure a Compute Node

Request an interactive job on a Polaris compute node:

```bash
qsub -I -q debug -l select=1,walltime=60:00 -A your_account_name -l filesystems=eagle
```
### 2. SSH to the Compute Node
```bash
ssh YOUR_COMPUTE_NODE_ID
```
### 3. Launch the ParslBox MCP Server
Navigate to this directory, activate the environment and start the MCP server.
```bash
# Activate ParslBox environment
conda activate parslbox_env # ParslBox environment
pip install mcp # Run this to install MCP (not yet included in ParslBox pyproject.toml)

# Start MCP server
python -m parslbox.mcp.mcp_server
```
The server will run on port 9005 by default.

### 4. Launch ChemGraph
In another terminal session, ssh to the same compute node that the ParslBox MCP is running
```bash
ssh YOUR_COMPUTE_NODE
```
Run ChemGraph with the example prompts available in run_mcp_parslbox.py
```bash
# Activate ChemGraph environment
conda activate chemgraph_env

# Set proxy
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

# Run ChemGraph. You can try different prompts in the run_cg_pbx.py
python run_cg_pbx.py
```