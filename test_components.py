"""
Test script for individual components
"""
import sys
from utils.data_loader import HotpotQALoader
from utils.rag_components import RAGRetriever, format_retrieved_context


def test_data_loader():
    """Test HotpotQA data loader"""
    print("=" * 50)
    print("Testing HotpotQA Data Loader")
    print("=" * 50)
    
    try:
        loader = HotpotQALoader(split="validation", sample_size=5)
        print(f"✓ Successfully loaded {len(loader)} examples")
        
        # Test getting an example
        example = loader.get_example(0)
        print(f"\n✓ Example 0:")
        print(f"  Question: {example['question']}")
        print(f"  Answer: {example['answer']}")
        print(f"  Level: {example['level']}")
        print(f"  Type: {example['type']}")
        
        # Test getting context documents
        documents = loader.get_context_documents(0)
        print(f"\n✓ Retrieved {len(documents)} context documents")
        print(f"  First document title: {documents[0]['title']}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_retriever():
    """Test RAG retriever"""
    print("\n" + "=" * 50)
    print("Testing RAG Retriever")
    print("=" * 50)
    
    try:
        # Load sample data
        loader = HotpotQALoader(split="validation", sample_size=1)
        documents = loader.get_context_documents(0)
        example = loader.get_example(0)
        
        # Initialize retriever
        retriever = RAGRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=3
        )
        print("✓ Retriever initialized")
        
        # Build index
        retriever.build_index(documents)
        print(f"✓ Index built with {len(documents)} documents")
        
        # Test retrieval
        query = example['question']
        retrieved = retriever.retrieve(query, k=2)
        print(f"\n✓ Retrieved {len(retrieved)} documents for query:")
        print(f"  Query: {query}")
        print(f"  First retrieved doc: {retrieved[0].page_content[:100]}...")
        
        # Test formatting
        context = format_retrieved_context(retrieved)
        print(f"\n✓ Formatted context length: {len(context)} characters")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vllm_connection():
    """Test vLLM server connection (requires running server)"""
    print("\n" + "=" * 50)
    print("Testing vLLM Connection")
    print("=" * 50)
    
    try:
        from utils.vllm_client import vLLMClient
        
        client = vLLMClient(
            base_url="http://localhost:8000/v1",
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )
        print("✓ vLLM client initialized")
        
        # Try to generate (will fail if server is not running)
        print("\nAttempting to connect to vLLM server...")
        print("(This will fail if vLLM server is not running)")
        
        response = client.generate("Hello, world!", max_tokens=10)
        print(f"✓ Successfully generated: {response}")
        
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nNote: This is expected if vLLM server is not running.")
        print("To start vLLM server, run:")
        print("  python -m vllm.entrypoints.openai.api_server --model <model_name> --port 8000")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("ADAPTIVE RAG COMPONENT TESTS")
    print("=" * 50 + "\n")
    
    results = {
        "Data Loader": test_data_loader(),
        "Retriever": test_retriever(),
        "vLLM Connection": test_vllm_connection()
    }
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
