#!/usr/bin/env python3
"""
Post PR documentation review to Slack commit review thread.
Usage: python post_slack_message.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional
from urllib import request, error


def get_slack_webhook_url() -> Optional[str]:
    """Get Slack webhook URL from environment."""
    return os.environ.get("SLACK_WEBHOOK_URL")


def post_slack_message(webhook_url: str, message_blocks: dict) -> bool:
    """Post message to Slack using webhook."""
    try:
        data = json.dumps(message_blocks).encode('utf-8')
        req = request.Request(webhook_url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        
        with request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                print("✓ Slack message posted successfully")
                return True
            else:
                print(f"✗ Slack API error: {response.status}")
                return False
    except error.HTTPError as e:
        if e.code == 200:  # Success
            print("✓ Slack message posted successfully")
            return True
        print(f"✗ Slack API error: {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"✗ Error posting to Slack: {e}")
        return False


def create_message_payload() -> dict:
    """Create the Slack message payload."""
    pr_593_url = "https://github.com/datarobot-oss/datarobot-genai/pull/593"
    
    return {
        "text": "PR Documentation Review - Last 24 Hours",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Commit Review Thread - Documentation Updates Required",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "PRs merged in the last 24 hours requiring documentation review:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔗 <{pr_593_url}|PR #593 - Revert PR 580 with ragas removal>\n_Significant API changes affecting how all frameworks (CrewAI, LangGraph, LlamaIndex) serialize pipeline interactions; LangGraph MCP API change (`load_mcp_tools(session=None, connection=...)`); README.md updated to reflect ragas dependency removal_"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📋 Review completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | 1 PR requires docs update"
                    }
                ]
            }
        ]
    }


def main():
    """Main entry point."""
    webhook_url = get_slack_webhook_url()
    
    if not webhook_url:
        print("WARNING: SLACK_WEBHOOK_URL not set in environment")
        print("Skipping Slack notification")
        print("\nMessage that would have been posted:")
        print(json.dumps(create_message_payload(), indent=2))
        return 0
    
    message_payload = create_message_payload()
    
    print("Posting PR documentation review to Slack...")
    print(f"Webhook URL: {webhook_url[:50]}...")
    
    success = post_slack_message(webhook_url, message_payload)
    
    if success:
        print("\nPR Documentation Review:")
        print("  • PR #593: Requires documentation updates")
        print("    - Significant API changes")
        print("    - README.md already updated in PR")
        print("    - LangGraph MCP API changes")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
