# Elicitation Support

The DataRobot MCP server includes support for **elicitation** - a feature that allows tools to request user input when needed. This is particularly useful for authentication tokens, API keys, or other sensitive information that cannot be hardcoded.

## Overview

Elicitation allows MCP tools to request additional information from users through the client. When a tool needs information from the user, it can use FastMCP's built-in `ctx.elicit()` method to request it. The client will handle prompting the user and returning the response.

## Usage

### Using FastMCP's Built-in Elicitation

The DataRobot MCP server uses **FastMCP's built-in elicitation** via `ctx.elicit()`. This is the recommended approach:

```python
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from mcp.types import ClientCapabilities, ElicitationCapability
from datarobot_genai.drmcp.core.mcp_instance import mcp

@mcp.tool(name="my_tool")
async def my_tool(ctx: Context, token: str | None = None) -> dict:
    """Tool that requires authentication token."""
    if not token:
        # Check if client supports elicitation before using it
        try:
            has_elicitation = ctx.session.check_client_capability(
                ClientCapabilities(elicitation=ElicitationCapability())
            )
        except (AttributeError, TypeError):
            has_elicitation = False
        
        if not has_elicitation:
            # Return graceful no-op response when elicitation not supported
            return {
                "status": "skipped",
                "message": "Elicitation not supported by client. Token parameter is required.",
                "elicitation_supported": False,
            }
        
        # Request token from user via elicitation
        result = await ctx.elicit(
            message="Authentication token required for GitHub",
            response_type=str,
        )
        
        if isinstance(result, AcceptedElicitation):
            token = result.data
        elif isinstance(result, DeclinedElicitation):
            return {
                "status": "error",
                "error": "Token declined by user",
                "message": "Cannot proceed without authentication token",
            }
        else:  # CancelledElicitation
            return {
                "status": "error",
                "error": "Operation cancelled",
                "message": "Request was cancelled",
            }
    
    # Proceed with token
    return {"status": "success", "token_provided": True}
```

### Elicitation Response Types

FastMCP's `ctx.elicit()` returns one of three response types:

- **`AcceptedElicitation[T]`**: User provided the requested information. Access via `result.data`
- **`DeclinedElicitation`**: User declined to provide the information
- **`CancelledElicitation`**: User cancelled the operation

### Supported Response Types

You can request different types of data:

```python
# String response
result = await ctx.elicit(message="Enter username:", response_type=str)

# Integer response
result = await ctx.elicit(message="Enter port number:", response_type=int)

# Boolean response
result = await ctx.elicit(message="Enable feature?", response_type=bool)

# Choice from list
result = await ctx.elicit(
    message="Select environment:",
    response_type=["development", "staging", "production"]
)

# Custom dataclass/model
from dataclasses import dataclass

@dataclass
class Credentials:
    username: str
    password: str

result = await ctx.elicit(
    message="Enter credentials:",
    response_type=Credentials
)
```

## Examples

### Authentication Token Elicitation

```python
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from datarobot_genai.drmcp.core.mcp_instance import mcp

@mcp.tool(name="connect_service")
async def connect_service(ctx: Context, service: str, token: str | None = None) -> dict:
    """Connect to an external service."""
    if not token:
        # Check if client supports elicitation before using it
        from mcp.types import ClientCapabilities, ElicitationCapability
        
        try:
            has_elicitation = ctx.session.check_client_capability(
                ClientCapabilities(elicitation=ElicitationCapability())
            )
        except (AttributeError, TypeError):
            has_elicitation = False
        
        if not has_elicitation:
            # Return graceful no-op response when elicitation not supported
            return {
                "status": "skipped",
                "message": f"Elicitation not supported by client. Token parameter is required to connect to {service}.",
                "elicitation_supported": False,
            }
        
        result = await ctx.elicit(
            message=(
                f"Authentication token required for {service}. "
                f"Please provide your {service} API token. "
                f"It will be stored securely in your workspace."
            ),
            response_type=str,
        )
        
        if isinstance(result, AcceptedElicitation):
            token = result.data
            # Store token securely
            # ... storage logic ...
        elif isinstance(result, DeclinedElicitation):
            return {
                "status": "error",
                "error": "Token declined",
                "message": f"Cannot connect to {service} without token",
            }
        else:  # CancelledElicitation
            return {
                "status": "error",
                "error": "Operation cancelled",
                "message": "Connection request was cancelled",
            }
    
    # Proceed with connection using token
    return {"status": "connected", "service": service}
```

### Configuration Value Elicitation

```python
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    deployment_id: str
    api_key: str

@mcp.tool(name="configure_deployment")
async def configure_deployment(
    ctx: Context,
    deployment_id: str,
    api_key: str | None = None
) -> dict:
    """Configure a deployment with an API key."""
    if not api_key:
        # Check if client supports elicitation before using it
        from mcp.types import ClientCapabilities, ElicitationCapability
        
        try:
            has_elicitation = ctx.session.check_client_capability(
                ClientCapabilities(elicitation=ElicitationCapability())
            )
        except (AttributeError, TypeError):
            has_elicitation = False
        
        if not has_elicitation:
            # Return graceful no-op response when elicitation not supported
            return {
                "status": "skipped",
                "message": "Elicitation not supported by client. API key parameter is required for deployment configuration.",
                "elicitation_supported": False,
            }
        
        result = await ctx.elicit(
            message="API key required for deployment configuration",
            response_type=str,
        )
        
        if isinstance(result, AcceptedElicitation):
            api_key = result.data
        else:
            return {
                "status": "error",
                "error": "API key not provided",
                "message": "Cannot configure deployment without API key",
            }
    
    # Proceed with configuration
    return {"status": "configured", "deployment_id": deployment_id}
```

## Best Practices

1. **Always provide clear messages**: The `message` parameter should clearly explain what input is needed and why
2. **Handle all response types**: Always check for `AcceptedElicitation`, `DeclinedElicitation`, and `CancelledElicitation`
3. **Store securely**: When users provide sensitive information (tokens, passwords), store it securely
4. **Validate input**: After receiving user input, validate it before using it
5. **Provide fallbacks**: If elicitation is declined or cancelled, provide meaningful error messages

## MCP Protocol Capabilities

According to the MCP specification, **elicitation is a CLIENT capability**, not a server capability. This means:

- **Clients MUST announce** elicitation capability if they want to receive elicitation requests
- **Servers do NOT announce** elicitation capability (they just use it if the client supports it)
- **Servers MUST NOT send** elicitation requests if the client doesn't announce support

### Client Capabilities (Required)

Clients **MUST** announce elicitation capability when initializing an MCP session if they want to support elicitation:

```python
from mcp import ClientSession
from mcp.types import ClientCapabilities

# According to MCP spec, elicitation is a top-level CLIENT capability
client_capabilities = ClientCapabilities(
    elicitation={"form": {}, "url": {}}
)

async with ClientSession(read_stream, write_stream) as session:
    await session.initialize(
        protocol_version="2024-11-05",
        capabilities=client_capabilities,
        client_info={"name": "my_client", "version": "1.0.0"},
    )
```

### Server Capabilities

The DataRobot MCP server **does NOT announce** elicitation capability because:
- Elicitation is a CLIENT capability according to the MCP specification
- Servers don't need to announce capabilities they use - only capabilities they provide
- The server will use elicitation if the client supports it, but doesn't need to announce it

### Behavior When Client Supports Elicitation

When the client announces elicitation support:
- **Server** can use `ctx.elicit()` in tools to request user input
- **Client** will receive and handle elicitation requests appropriately
- **Both** benefit from a standardized elicitation flow

### Behavior When Client Does NOT Support Elicitation

If the client does NOT announce elicitation capability:
- **Server MUST NOT send** elicitation requests
- Tools should check client capability before using elicitation
- Tools should return a graceful no-op response (not an error) when elicitation is not supported
- This follows MCP standards for graceful degradation

## Implementation Details

The DataRobot MCP server uses **FastMCP's built-in elicitation** via `ctx.elicit()`. This provides:

- **Type-safe responses**: Responses are validated against the requested type
- **Protocol-level handling**: Elicitation is handled at the MCP protocol level
- **Automatic client integration**: Clients that support elicitation automatically handle requests
- **Standardized flow**: Follows the MCP specification for elicitation

### Checking Client Capability

Tools should check if the client supports elicitation before using it. According to MCP standards, when elicitation is not supported, tools should return a graceful no-op response rather than throwing an error:

```python
from fastmcp import Context
from fastmcp.server.context import AcceptedElicitation
from mcp.types import ClientCapabilities, ElicitationCapability

@mcp.tool(name="my_tool")
async def my_tool(ctx: Context) -> dict:
    # Check if client supports elicitation
    try:
        has_elicitation = ctx.session.check_client_capability(
            ClientCapabilities(elicitation=ElicitationCapability())
        )
    except (AttributeError, TypeError):
        # If check_client_capability doesn't exist or fails, assume no support
        has_elicitation = False
    
    if not has_elicitation:
        # Client doesn't support elicitation - return graceful no-op response
        # According to MCP spec, don't throw an error, return a response indicating
        # the operation couldn't complete without elicitation
        return {
            "status": "skipped",
            "message": "Elicitation not supported by client. This operation requires elicitation support.",
            "elicitation_supported": False,
        }
    
    # Safe to use elicitation
    result = await ctx.elicit(message="Enter value:", response_type=str)
    if isinstance(result, AcceptedElicitation):
        value = result.data
        # Proceed with the operation using the elicited value
        return {"status": "success", "value": value}
    else:
        # Handle declined or cancelled elicitation
        return {
            "status": "error",
            "error": "Elicitation declined or cancelled",
            "message": "Operation requires user input",
        }
```

## Client Integration

Clients that support elicitation will automatically:
1. Receive elicitation requests from the server
2. Prompt the user for the requested information
3. Return the user's response (accept, decline, or cancel)
4. Handle the response appropriately

The elicitation flow is handled entirely at the protocol level by FastMCP and the MCP SDK, so no custom client code is needed beyond announcing the capability during initialization.
