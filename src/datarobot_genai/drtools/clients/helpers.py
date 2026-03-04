from fastmcp.server.dependencies import get_http_headers


def get_api_key_from_headers(header_name: str) -> str | None:
    if header_name.startswith("x-datarobot-"):
        header_name = header_name[len("x-datarobot-") :]
    if header_name.startswith("x-"):
        header_name = header_name[len("x-") :]

    headers = get_http_headers()

    # Try to get from x-{header}
    if api_key := headers.get(f"x-{header_name}"):
        return api_key

    # Try go get from x-datarobot-{header}
    if api_key := headers.get(f"x-datarobot-{header_name}"):
        return api_key

    return None
