#!/usr/bin/env python3
"""
Quick test script for the Thought Tracer logit lens application.
This script tests the model loading and basic functionality.
"""

import sys
import time
from pathlib import Path

# Add the src directory to Python path so we can import the module
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logitlens_tui.modeling import load_ministral_model
from logitlens_tui.lens import prepare_prompt_state


def test_model_loading():
    """Test that the model loads correctly on CPU."""
    print("🚀 Testing Thought Tracer - Model Loading...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Load the model
        print("Loading Ministral-3-3B model...")
        loaded = load_ministral_model('./Ministral-3-3B-Instruct-2512')
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
        print(f"📍 Device: {loaded.input_device}")
        print(f"🧠 Model type: {type(loaded.model).__name__}")
        
        return loaded
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def test_tokenization(loaded):
    """Test the tokenization pipeline."""
    print("\n🧩 Testing Tokenization Pipeline...")
    print("=" * 50)
    
    try:
        # Test with a sample prompt
        test_prompts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is transforming the world"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: '{prompt}'")
            
            start_time = time.time()
            prompt_state = prepare_prompt_state(
                loaded, 
                system_prompt="You are a helpful assistant.",
                user_prompt=prompt
            )
            
            token_time = time.time() - start_time
            print(f"   ✅ Tokenized in {token_time:.3f} seconds")
            print(f"   🔢 Tokens: {len(prompt_state.input_ids)}")
            print(f"   🏷️  First 5 tokens: {prompt_state.token_texts[:5]}")
            
        print(f"\n✅ All tokenization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Thought Tracer - Quick Test Suite")
    print("=" * 60)
    print("Testing model compatibility and basic functionality...\n")
    
    # Test model loading
    loaded = test_model_loading()
    
    if loaded:
        # Test tokenization
        tokenization_ok = test_tokenization(loaded)
        
        if tokenization_ok:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("✅ Model loads correctly on CPU")
            print("✅ Tokenization pipeline working")
            print("✅ Ready for interactive use!")
            print("\n💡 Run the full app with:")
            print("   .venv/bin/python -m logitlens_tui --model-path ./Ministral-3-3B-Instruct-2512")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    else:
        print("\n❌ Model loading failed")
        sys.exit(1)


if __name__ == "__main__":
    main()