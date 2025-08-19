#!/usr/bin/env python3
"""
Debug script to trace what happens with agents when API keys are missing.
"""

import asyncio
import logging
import chess
from chess_llm_bench.llm.client import LLMClient
from chess_llm_bench.core.models import BotSpec
from chess_llm_bench.core.budget import start_budget_tracking, get_budget_tracker

# Enable logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

async def debug_agent_behavior():
    """Debug what happens when we use agents without API keys."""
    print("🔍 Debugging agent behavior without API keys...")
    
    # Start budget tracking
    tracker = start_budget_tracking(1.0)
    
    # Create a bot spec for OpenAI
    bot = BotSpec(provider='openai', model='gpt-4o-mini', name='TestBot')
    
    try:
        print("\n1. Creating LLM client with agent mode...")
        client = LLMClient(bot, use_agent=True, agent_strategy='balanced', verbose_agent=True)
        print(f"   ✅ Client created successfully")
        print(f"   Agent mode: {client.use_agent}")
        print(f"   Provider type: {type(client.provider)}")
        
        # Try to make a move
        board = chess.Board()
        print(f"\n2. Making a move from starting position...")
        print(f"   Position: {board.fen()}")
        
        try:
            move = await client.pick_move(board)
            print(f"   ✅ Move generated: {move.uci()}")
            
            # Check if any costs were recorded
            current_cost = tracker.get_current_cost()
            print(f"   💰 Cost recorded: ${current_cost:.4f}")
            
            if current_cost == 0:
                print("   ⚠️  No costs recorded - likely using random fallback")
            else:
                print("   ✅ Costs recorded - LLM was actually used")
            
        except Exception as e:
            print(f"   ❌ Move generation failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Client creation failed: {e}")
        return
    
    # Show budget summary
    print(f"\n3. Final budget status:")
    print(f"   Total cost: ${tracker.get_current_cost():.4f}")
    print(f"   Total requests: {tracker.summary.total_requests}")
    
    # Show detailed usage records
    if tracker.usage_records:
        print(f"\n4. Usage records ({len(tracker.usage_records)} total):")
        for i, record in enumerate(tracker.usage_records, 1):
            print(f"   {i}. {record.provider}:{record.model} - ${record.cost:.4f} ({'✅' if record.success else '❌'})")
            if record.error_message:
                print(f"      Error: {record.error_message}")
    else:
        print(f"\n4. No usage records found")

if __name__ == "__main__":
    asyncio.run(debug_agent_behavior())
