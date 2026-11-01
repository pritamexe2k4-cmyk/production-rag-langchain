"""
Lesson 6.5: GraphRAG Introduction

Traditional RAG struggles with multi-hop reasoning:
Q: "Which employees work in the same department as the CEO's assistant?"
- Need to find: CEO → CEO's assistant → their department → other employees
- Vector similarity can't traverse these relationships

GraphRAG builds a knowledge graph from documents, enabling:
- Multi-hop reasoning across entities
- Relationship-based retrieval
- Global summarization of topics

This lesson introduces the concepts. For full implementation,
see Microsoft's GraphRAG library or LangGraph's graph-based retrieval.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import networkx as nx

load_dotenv()

# ============================================================
# THE PROBLEM: MULTI-HOP REASONING
# ============================================================


def demonstrate_multihop_problem():
    """
    Show why traditional RAG fails at multi-hop reasoning.
    """

    print("=" * 60)
    print("THE PROBLEM: Multi-Hop Reasoning")
    print("=" * 60)

    # Sample documents (imagine these are chunked)
    documents = [
        "John Smith is the CEO of TechCorp. He joined the company in 2015.",
        "Sarah Johnson is John Smith's executive assistant. She manages his calendar.",
        "Sarah Johnson works in the Executive department on the 5th floor.",
        "The Executive department also includes Mike Brown and Lisa Chen.",
        "Mike Brown is the Chief Financial Officer. Lisa Chen is the Chief Legal Officer.",
    ]

    query = "Who works in the same department as the CEO's assistant?"

    print(f'\nQuery: "{query}"')
    print("\nDocuments in our corpus:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    print("\n" + "-" * 60)
    print("WHY TRADITIONAL RAG FAILS:")
    print("-" * 60)

    reasoning = """
    To answer this query, we need to:

    1. Find the CEO → "John Smith is the CEO"
    2. Find the CEO's assistant → "Sarah Johnson is John Smith's executive assistant"
    3. Find Sarah's department → "Sarah Johnson works in the Executive department"
    4. Find others in that department → "Mike Brown and Lisa Chen"

    Traditional vector search might retrieve:
    - Documents mentioning "department" (step 3, 4)
    - But miss the connection: CEO → assistant → department

    The query "CEO's assistant" won't semantically match
    "Sarah Johnson works in the Executive department" because
    Sarah's name isn't in the query!

    This is a MULTI-HOP problem: we need to traverse relationships.
    """
    print(reasoning)


# ============================================================
# THE SOLUTION: KNOWLEDGE GRAPHS
# ============================================================


def build_knowledge_graph():
    """
    Build a simple knowledge graph from our documents.
    This shows the CONCEPT - production implementations use
    LLMs to extract entities and relationships automatically.
    """

    print("\n" + "=" * 60)
    print("THE SOLUTION: Knowledge Graph")
    print("=" * 60)

    # Create a directed graph
    G = nx.DiGraph()

    # Add entities (nodes)
    entities = [
        ("John Smith", {"type": "Person", "role": "CEO"}),
        ("Sarah Johnson", {"type": "Person", "role": "Executive Assistant"}),
        ("Mike Brown", {"type": "Person", "role": "CFO"}),
        ("Lisa Chen", {"type": "Person", "role": "CLO"}),
        ("TechCorp", {"type": "Organization"}),
        ("Executive Department", {"type": "Department", "floor": "5th"}),
    ]

    for entity, attrs in entities:
        G.add_node(entity, **attrs)

    # Add relationships (edges)
    relationships = [
        ("John Smith", "TechCorp", {"relation": "CEO_OF"}),
        ("Sarah Johnson", "John Smith", {"relation": "ASSISTANT_TO"}),
        ("Sarah Johnson", "Executive Department", {"relation": "WORKS_IN"}),
        ("Mike Brown", "Executive Department", {"relation": "WORKS_IN"}),
        ("Lisa Chen", "Executive Department", {"relation": "WORKS_IN"}),
        ("John Smith", "Executive Department", {"relation": "WORKS_IN"}),
    ]

    for source, target, attrs in relationships:
        G.add_edge(source, target, **attrs)

    print("\nKnowledge Graph Structure:")
    print("-" * 60)

    # Print nodes
    print("\nENTITIES (Nodes):")
    for node, attrs in G.nodes(data=True):
        print(f"  [{attrs.get('type', 'Unknown')}] {node}")
        for key, value in attrs.items():
            if key != "type":
                print(f"       - {key}: {value}")

    # Print edges
    print("\nRELATIONSHIPS (Edges):")
    for source, target, attrs in G.edges(data=True):
        print(f"  {source} --[{attrs['relation']}]--> {target}")

    return G


def traverse_graph_for_answer(G: nx.DiGraph):
    """
    Traverse the knowledge graph to answer the multi-hop query.
    """

    print("\n" + "=" * 60)
    print("GRAPH TRAVERSAL FOR MULTI-HOP REASONING")
    print("=" * 60)

    query = "Who works in the same department as the CEO's assistant?"
    print(f'\nQuery: "{query}"')
    print("\nStep-by-step traversal:")
    print("-" * 60)

    # Step 1: Find the CEO
    print("\nStep 1: Find the CEO")
    ceo = None
    for node, attrs in G.nodes(data=True):
        if attrs.get("role") == "CEO":
            ceo = node
            print(f"  Found: {ceo}")
            break

    # Step 2: Find the CEO's assistant
    print("\nStep 2: Find the CEO's assistant")
    assistant = None
    for source, target, attrs in G.edges(data=True):
        if target == ceo and attrs["relation"] == "ASSISTANT_TO":
            assistant = source
            print(f"  Found: {assistant} --[ASSISTANT_TO]--> {ceo}")
            break

    # Step 3: Find the assistant's department
    print("\nStep 3: Find the assistant's department")
    department = None
    for source, target, attrs in G.edges(data=True):
        if source == assistant and attrs["relation"] == "WORKS_IN":
            department = target
            print(f"  Found: {assistant} --[WORKS_IN]--> {department}")
            break

    # Step 4: Find others in that department
    print("\nStep 4: Find others in the same department")
    coworkers = []
    for source, target, attrs in G.edges(data=True):
        if target == department and attrs["relation"] == "WORKS_IN":
            if source != assistant:  # Exclude the assistant themselves
                coworkers.append(source)
                print(f"  Found: {source} --[WORKS_IN]--> {department}")

    print("\n" + "-" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(f"The CEO's assistant is {assistant}.")
    print(f"{assistant} works in the {department}.")
    print(f"Others in the same department: {', '.join(coworkers)}")


# ============================================================
# LLM-BASED ENTITY EXTRACTION
# ============================================================


def extract_entities_with_llm(text: str) -> dict:
    """
    Use LLM to extract entities and relationships from text.
    This is how GraphRAG automates knowledge graph construction.
    """

    print("\n" + "=" * 60)
    print("LLM-BASED ENTITY EXTRACTION")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at extracting knowledge graph elements from text.

Given a text, extract:
1. ENTITIES: People, organizations, places, concepts
2. RELATIONSHIPS: How entities are connected

Output in this exact format:
ENTITIES:
- [TYPE] Name: description

RELATIONSHIPS:
- Source --[RELATION]--> Target

Be specific about relationship types (WORKS_FOR, MANAGES, LOCATED_IN, etc.)""",
            ),
            (
                "human",
                """Extract entities and relationships from this text:

{text}""",
            ),
        ]
    )

    sample_text = """
    Acme Corporation announced today that Jennifer Lee has been appointed as the new
    Chief Technology Officer. She will report directly to CEO Marcus Chen. Jennifer
    previously led the AI research team at DataTech Inc. in Boston. The company's
    headquarters will remain in San Francisco, where the engineering team of 50
    developers is based.
    """

    print(f"\nInput text:")
    print(f'  "{sample_text.strip()}"')

    chain = extraction_prompt | llm
    result = chain.invoke({"text": sample_text})

    print(f"\nExtracted knowledge graph elements:")
    print("-" * 60)
    print(result.content)

    return result.content


# ============================================================
# GRAPHRAG ARCHITECTURE
# ============================================================


def show_graphrag_architecture():
    """
    Show the full GraphRAG architecture.
    """

    print("\n" + "=" * 60)
    print("GRAPHRAG ARCHITECTURE")
    print("=" * 60)

    architecture = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    INDEXING PHASE                           │
    └─────────────────────────────────────────────────────────────┘

    Documents           LLM Extraction          Knowledge Graph
    ┌─────────┐        ┌─────────────┐         ┌─────────────┐
    │  Doc 1  │───────▶│  Extract    │────────▶│   Entities  │
    │  Doc 2  │        │  Entities & │         │      +      │
    │  Doc 3  │        │  Relations  │         │ Relationships│
    └─────────┘        └─────────────┘         └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │  Community  │  (Group related entities)
                       │  Detection  │
                       └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │  Generate   │  (LLM summarizes each community)
                       │  Summaries  │
                       └─────────────┘


    ┌─────────────────────────────────────────────────────────────┐
    │                    QUERY PHASE                              │
    └─────────────────────────────────────────────────────────────┘

    User Query             Processing              Response
    ┌─────────┐        ┌─────────────┐         ┌─────────────┐
    │ "Who    │───────▶│ 1. Identify │────────▶│  "Mike and  │
    │ works   │        │    entities │         │   Lisa work │
    │ with    │        │ 2. Traverse │         │   in the    │
    │ CEO's   │        │    graph    │         │   Executive │
    │ asst?"  │        │ 3. Generate │         │   Dept..."  │
    └─────────┘        └─────────────┘         └─────────────┘


    TWO QUERY MODES:

    LOCAL SEARCH (Multi-hop reasoning):
    - Identify entities in query
    - Traverse relationships to find answers
    - Good for: specific questions about connections

    GLOBAL SEARCH (Holistic understanding):
    - Use community summaries
    - Aggregate knowledge across documents
    - Good for: "What are the main themes?" type questions
    """

    print(architecture)


# ============================================================
# IMPLEMENTATION OPTIONS
# ============================================================


def show_implementation_options():
    """
    Show practical implementation options for GraphRAG.
    """

    print("\n" + "=" * 60)
    print("IMPLEMENTATION OPTIONS")
    print("=" * 60)

    options = """
    OPTION 1: Microsoft GraphRAG (Recommended for Full Implementation)
    ─────────────────────────────────────────────────────────────────
    ```bash
    pip install graphrag

    # Index documents
    graphrag index --root ./my_docs

    # Query
    graphrag query --method local "Who works with the CEO?"
    graphrag query --method global "What are the main themes?"
    ```
    Pros: Full-featured, well-documented, production-ready
    Cons: Heavy, requires significant compute for indexing


    OPTION 2: LangGraph + Neo4j (Custom Implementation)
    ─────────────────────────────────────────────────────────────────
    ```python
    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain

    graph = Neo4jGraph(url="bolt://localhost:7687")

    chain = GraphCypherQAChain.from_llm(
        llm=ChatOpenAI(),
        graph=graph,
        verbose=True
    )
    result = chain.invoke("Who reports to the CEO?")
    ```
    Pros: Full control, integrates with existing Neo4j
    Cons: Need to manage Neo4j, manual entity extraction


    OPTION 3: LlamaIndex Knowledge Graph Index
    ─────────────────────────────────────────────────────────────────
    ```python
    from llama_index.core import KnowledgeGraphIndex

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=10,
        include_embeddings=True
    )
    query_engine = kg_index.as_query_engine()
    response = query_engine.query("Who works in Engineering?")
    ```
    Pros: Easy to use, good LlamaIndex integration
    Cons: Less powerful than full GraphRAG


    OPTION 4: Hybrid - Vector + Graph
    ─────────────────────────────────────────────────────────────────
    Combine traditional RAG with graph traversal:

    1. Vector search to find relevant documents
    2. Extract entities from retrieved docs
    3. Graph traversal for multi-hop reasoning
    4. LLM synthesizes final answer

    Pros: Best of both worlds
    Cons: More complex to implement


    WHEN TO USE GRAPHRAG:
    ─────────────────────────────────────────────────────────────────
    ✅ Documents describe relationships (org charts, research papers)
    ✅ Queries need multi-hop reasoning
    ✅ "Who/what is connected to X?" type questions
    ✅ Need global summarization across documents

    ❌ Simple fact retrieval (use standard RAG)
    ❌ Small document sets (<100 docs)
    ❌ Real-time indexing requirements
    ❌ Cost-sensitive applications (indexing is expensive)
    """

    print(options)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.5: GRAPHRAG INTRODUCTION")
    print("=" * 60)
    print("\nKnowledge graphs for multi-hop reasoning in RAG")
    print("=" * 60)

    # 1. Show the problem
    demonstrate_multihop_problem()

    # 2. Build and show knowledge graph
    G = build_knowledge_graph()

    # 3. Traverse graph for answer
    traverse_graph_for_answer(G)

    # 4. LLM-based extraction
    extract_entities_with_llm("")

    # 5. Architecture overview
    show_graphrag_architecture()

    # 6. Implementation options
    show_implementation_options()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. Traditional RAG fails at multi-hop reasoning
    2. GraphRAG builds knowledge graphs from documents
    3. Entities (nodes) + Relationships (edges) = traversable structure
    4. Local search: traverse graph for specific answers
    5. Global search: use community summaries for themes
    6. Use when: relationships matter, multi-hop queries expected
    7. Microsoft GraphRAG is the go-to production implementation
    """
    )
