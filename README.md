# Anthropic_Router

enable running claude code with any LLM provider by routing all claude code LM api traffic via litellm.

suitable for cli vibe coding / batch run for LM benchmarking

compatible on windows host (claude code seems have problems in windows container hmmm...)

## Quick start

```bash
pip install -r requirements.txt
```

```python
from server import Server
server = Server()
base_url, token = server.start_from_api_key(
    model = "openrouter/openai/gpt-5.5", # model should be in litellm model name format
    base_url = "https://openrouter.ai/api/v1",
    api_key = "your api key",
    # put other arguments you need here...
)
# for azure openai user please use the method server.start_from_azure_openai()
print(base_url, token, sep = " ; ")
# example result: http://10.190.175.111:54657 ; dummy
```

```bash
curl -fsSL https://claude.ai/install.sh | bash -s -- 2.0.65
alias claude="$HOME/.local/bin/claude"

# on windows cmd: curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd 2.1.89 && del install.cmd

export ANTHROPIC_BASE_URL=http://10.190.175.111:54657 # the base url you got from the above python script
export ANTHROPIC_AUTH_TOKEN=dummy # the token you got from the above python script
claude # or `claude -p "user prompt"` to do a task at backend
```

## One important caveat: rule of CLI agent

The submit/stop/exit rule of CLI agent like codex / claude code / gemini-cli is that LM outputs one pure-text response without tool call. Many LMs especially open-source ones have not been trained on such submission rule of CLI agent, so they may either
 
1. do not know how to stop the agent, after the LM finishes the task, it started to send meaningless tool calls...
2. stop the agent early unintentionally because it forget to generate a tool call in one response / or its generated tool call has JSON decode error and is thus parsed as pure text.

For cli vibe coding users it should be tried which LM can use claude code normally and which cannot.

For benchmaking LM agents with claude code, researchers can typically install a standalone claude code in the container of a task instance. However, it should be noted that LMs that cannot handle claude code normally (for the two reasons above) may result in very low success rate.

