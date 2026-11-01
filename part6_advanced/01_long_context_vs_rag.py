"""
Lesson 6.1: Long Context vs RAG - When to Use Each
FreeCodeCamp Production RAG Course (April 2026)

This lesson addresses the "Is RAG dead?" question head-on.
With 1M+ token context windows, when should you still use RAG?

Key insight: It's not either/or - it's about cost, latency, and use case.
"""

import os
import time
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# ============================================================
# COST COMPARISON: Long Context vs RAG
# ============================================================

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)


def calculate_costs():
    """
    Compare costs: stuffing 100K tokens vs RAG retrieval.

    GPT-4o pricing (April 2026):
    - Input: $2.50 per 1M tokens
    - Output: $10.00 per 1M tokens

    RAG typically retrieves 4-8 chunks (~2K tokens total)
    """

    print("=" * 60)
    print("COST COMPARISON: Long Context vs RAG")
    print("=" * 60)

    # Scenario: Query against 100K tokens of documentation
    doc_tokens = 100_000
    query_tokens = 100
    output_tokens = 500

    # Long Context approach - stuff everything
    long_context_input = doc_tokens + query_tokens
    long_context_cost = (long_context_input * 2.50 / 1_000_000) + (
        output_tokens * 10.00 / 1_000_000
    )

    # RAG approach - retrieve 4 chunks of ~500 tokens each
    rag_chunks = 4
    tokens_per_chunk = 500
    rag_input = (rag_chunks * tokens_per_chunk) + query_tokens
    rag_cost = (rag_input * 2.50 / 1_000_000) + (output_tokens * 10.00 / 1_000_000)

    print(f"\nScenario: Query against {doc_tokens:,} tokens of docs")
    print(f"Query: {query_tokens} tokens, Expected output: {output_tokens} tokens")
    print()
    print(f"LONG CONTEXT:")
    print(f"  Input tokens: {long_context_input:,}")
    print(f"  Cost per query: ${long_context_cost:.5f}")
    print()
    print(f"RAG (4 chunks x 500 tokens):")
    print(f"  Input tokens: {rag_input:,}")
    print(f"  Cost per query: ${rag_cost:.5f}")
    print()
    print(f"RAG is {long_context_cost / rag_cost:.0f}x cheaper per query!")
    print()

    # At scale
    queries_per_day = 10_000
    print(f"At {queries_per_day:,} queries/day:")
    print(f"  Long Context: ${long_context_cost * queries_per_day:.2f}/day")
    print(f"  RAG: ${rag_cost * queries_per_day:.2f}/day")
    print(
        f"  Monthly savings with RAG: ${(long_context_cost - rag_cost) * queries_per_day * 30:.2f}"
    )

    return long_context_cost, rag_cost


# ============================================================
# LATENCY COMPARISON
# ============================================================


def compare_latency():
    """
    Demonstrate latency differences between approaches.

    Long context = more tokens to process = more time
    RAG = retrieval overhead + smaller context = usually faster
    """

    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)

    # Small context (simulating RAG)
    small_context = (
        "The company's return policy allows returns within 30 days with receipt."
    )

    # Large context (simulating long context - using repeated text for demo)
    large_context = (small_context + "\n\n") * 50  # ~2500 tokens

    query = "What is the return policy?"

    # Time small context
    start = time.time()
    response_small = llm.invoke(f"Context: {small_context}\n\nQuestion: {query}")
    small_time = time.time() - start

    # Time large context
    start = time.time()
    response_large = llm.invoke(f"Context: {large_context}\n\nQuestion: {query}")
    large_time = time.time() - start

    print(f"\nSmall context (~50 tokens): {small_time:.2f}s")
    print(f"Large context (~2500 tokens): {large_time:.2f}s")
    print(
        f"Difference: {large_time - small_time:.2f}s ({large_time/small_time:.1f}x slower)"
    )
    print("\nNote: At 100K+ tokens, latency difference is much more dramatic!")


# ============================================================
# DECISION FRAMEWORK
# ============================================================


def print_decision_framework():
    """
    When to use Long Context vs RAG - a practical decision tree.
    """

    print("\n" + "=" * 60)
    print("DECISION FRAMEWORK: Long Context vs RAG")
    print("=" * 60)

    framework = """
    USE LONG CONTEXT WHEN:
    ┌─────────────────────────────────────────────────────────────┐
    │ ✓ Document corpus is small (<50K tokens)                   │
    │ ✓ Query volume is low (<100 queries/day)                   │
    │ ✓ You need to analyze the ENTIRE document                  │
    │ ✓ Documents change frequently (no embedding overhead)      │
    │ ✓ Simplicity > cost optimization                           │
    └─────────────────────────────────────────────────────────────┘

    USE RAG WHEN:
    ┌─────────────────────────────────────────────────────────────┐
    │ ✓ Document corpus is large (>100K tokens)                  │
    │ ✓ Query volume is high (1000s of queries/day)              │
    │ ✓ Users ask about specific topics (not whole-doc analysis) │
    │ ✓ Cost and latency matter                                  │
    │ ✓ You need citations/source tracking                       │
    │ ✓ Documents are relatively stable                          │
    └─────────────────────────────────────────────────────────────┘

    HYBRID APPROACH (Best of Both):
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. RAG retrieves candidate documents/chunks                │
    │ 2. Load full document(s) into context for detailed answer  │
    │ 3. Best for: "Tell me about X, then analyze deeply"        │
    └─────────────────────────────────────────────────────────────┘

    THE REAL ANSWER: It's not "RAG vs Long Context"
    It's "RAG AND Long Context" - use both strategically!
    """

    print(framework)


# ============================================================
# PRACTICAL EXAMPLE: Hybrid Approach
# ============================================================


def demo_hybrid_approach():
    """
    Demonstrate a hybrid approach:
    1. Use RAG to find relevant documents
    2. Load full document(s) for detailed analysis
    """

    print("\n" + "=" * 60)
    print("HYBRID APPROACH DEMO")
    print("=" * 60)

    # Sample documents (in real use, these would be full documents)
    documents = [
        Document(
            page_content="""
            Remote Work Policy (Full Document)

            Section 1: Eligibility
            All full-time employees who have completed their 90-day probation period
            are eligible for remote work. Part-time employees may request remote work
            on a case-by-case basis.

            Section 2: Schedule
            Employees may work remotely up to 3 days per week. Core hours are
            10am-3pm in the employee's local timezone. All remote work must be
            logged in the time tracking system.

            Section 3: Equipment
            The company provides a laptop and monitor for remote work. Employees
            are responsible for their internet connection. A $500 home office
            stipend is available annually.

            Section 4: Communication
            Employees must be reachable via Slack during core hours. Video must
            be on during team meetings. Response time expectation is 30 minutes
            during core hours.
            """,
            metadata={"source": "remote_work_policy.pdf", "doc_type": "policy"},
        ),
        Document(
            page_content="""
            Expense Reimbursement Policy (Full Document)

            Section 1: Pre-Approval
            Expenses over $500 require manager pre-approval. Travel expenses
            require VP approval for amounts over $2000.

            Section 2: Documentation
            Receipts required for all expenses over $25. Digital receipts
            accepted. Credit card statements alone are not sufficient.

            Section 3: Submission Timeline
            Expense reports must be submitted within 30 days of the expense.
            Late submissions require director approval and may be denied.

            Section 4: Reimbursement
            Approved expenses are reimbursed within 10 business days.
            Reimbursement is via direct deposit to payroll account.
            """,
            metadata={"source": "expense_policy.pdf", "doc_type": "policy"},
        ),
        Document(
            page_content="""
            PTO and Leave Policy (Full Document)

            Section 1: Annual PTO
            New employees receive 15 days of PTO per year. PTO increases by
            1 day per year of service, up to maximum of 25 days.

            Section 2: Sick Leave
            Employees receive 10 days of sick leave per year. Sick leave does
            not roll over. Doctor's note required for absences over 3 days.

            Section 3: Holidays
            The company observes 10 paid holidays per year. Holiday schedule
            is published in January each year.

            Section 4: Leave of Absence
            Unpaid leave of up to 12 weeks may be requested for family or
            medical reasons. FMLA eligibility requirements apply.
            """,
            metadata={"source": "pto_policy.pdf", "doc_type": "policy"},
        ),
    ]

    # Step 1: Create vector store for RAG retrieval
    print("\nStep 1: Creating vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, collection_name="hybrid_demo"
    )

    # Step 2: User asks a question
    query = "What's the policy on working from home and what equipment do I get?"
    print(f"\nQuery: {query}")

    # Step 3: RAG retrieves relevant document
    print("\nStep 2: RAG retrieves relevant document...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    relevant_docs = retriever.invoke(query)

    print(f"Retrieved: {relevant_docs[0].metadata['source']}")

    # Step 4: Load FULL document into context (hybrid part)
    print("\nStep 3: Loading full document into context...")
    full_doc = relevant_docs[0].page_content

    # Step 5: Generate detailed answer with full context
    print("\nStep 4: Generating detailed answer with full document context...")
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful HR assistant. Use the full policy document
        below to give a comprehensive answer. Include all relevant details from
        the document.

        Policy Document:
        {document}""",
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"document": full_doc, "query": query})

    print("\nAnswer:")
    print(response.content)

    # Cleanup
    vectorstore.delete_collection()

    print("\n" + "=" * 60)
    print("HYBRID APPROACH BENEFITS:")
    print("=" * 60)
    print("1. RAG finds the RIGHT document quickly")
    print("2. Full document context ensures nothing is missed")
    print("3. Still cheaper than stuffing ALL documents")
    print("4. Best for: 'Find relevant doc, then analyze thoroughly'")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LESSON 6.1: LONG CONTEXT VS RAG")
    print("=" * 60)
    print("\nThe question isn't 'Is RAG dead?'")
    print("It's 'When should I use RAG vs Long Context?'")
    print("=" * 60)

    # 1. Cost comparison
    calculate_costs()

    # 2. Latency comparison
    compare_latency()

    # 3. Decision framework
    print_decision_framework()

    # 4. Hybrid approach demo
    demo_hybrid_approach()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
    1. RAG is NOT dead - it's about cost and latency at scale
    2. Long context wins for small docs, low volume, whole-doc analysis
    3. RAG wins for large corpora, high volume, specific queries
    4. Hybrid approach: RAG to find docs, long context to analyze
    5. The future is using BOTH strategically
    """
    )
