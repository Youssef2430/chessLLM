#!/usr/bin/env python3
"""
Debug wrapper to intercept and log all LLM client creation attempts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and patch the LLMClient class before anything else uses it
from chess_llm_bench.llm.client import LLMClient

original_init = LLMClient.__init__

def debug_init(self, *args, **kwargs):
    print(f"🔍 LLMClient.__init__ called!")
    print(f"  Args: {args}")
    print(f"  Kwargs: {kwargs}")
    
    if args:
        spec = args[0]
        print(f"  Bot spec: {spec.provider}:{spec.model}:{spec.name}")
        
    try:
        result = original_init(self, *args, **kwargs)
        print(f"  ✅ Client created successfully")
        print(f"  Provider type: {type(self.provider)}")
        print(f"  Use agent: {getattr(self, 'use_agent', 'unknown')}")
        return result
    except Exception as e:
        print(f"  ❌ Client creation failed: {e}")
        raise

# Monkey patch the constructor
LLMClient.__init__ = debug_init

# Now run a quick test
if __name__ == "__main__":
    import asyncio
    from chess_llm_bench.core.models import BotSpec
    
    print("Testing with debug wrapper...")
    
    bot = BotSpec(provider='openai', model='gpt-4o-mini', name='TestBot')
    try:
        client = LLMClient(bot, use_agent=True, agent_strategy='balanced')
        print("Final result: Client created successfully")
    except Exception as e:
        print(f"Final result: Client creation failed: {e}")
