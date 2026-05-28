import time
from argparse import ArgumentParser

import yaml
from server import Server


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", help="config yaml file path", required=True)
    parser.add_argument("--port", type=int, help="local server port")
    parser.add_argument("--token", help="token for ANTHROPIC_AUTH_TOKEN")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert config.get("claude_model", False)
    assert config.get("provider_model", False)
    assert config["provider_model"].get("model")

    server_config = config.get("server", {})
    if args.port is not None:
        server_config["port"] = args.port
    if args.token is not None:
        server_config["auth_token"] = args.token
    elif "token" in server_config:
        server_config["auth_token"] = server_config.pop("token")
    provider_config = {**config["provider_model"], **server_config}

    server = Server(config["claude_model"]["sonnet"], config["claude_model"]["haiku"])
    model = provider_config["model"]
    if "azure" in model:
        url, token = server.start_from_azure_openai(**provider_config)
    else:
        assert provider_config.get("api_key", False)
        url, token = server.start_from_api_key(**provider_config)
    print("Claude Code server started...")
    print(f"ANTHROPIC_BASE_URL={url}")
    print(f"ANTHROPIC_AUTH_TOKEN={token}")
    print("If want to stop just type Ctrl^C")
    while True:
        time.sleep(120)

if __name__ == "__main__":
    main()
