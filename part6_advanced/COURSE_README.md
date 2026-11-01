<!-- @format -->

# Production RAG Masterclass

**Build, Debug, Optimize, and Scale RAG Systems for Production**

> "You followed a RAG tutorial. It worked on 10 documents. Then you tried 10,000... and everything broke."

This course teaches what RAG tutorials skip: **why 90% of RAG projects fail in production and how to fix them.**

---

## What You'll Learn

| Part       | Topic                | What You'll Build                                       |
| ---------- | -------------------- | ------------------------------------------------------- |
| **Part 1** | Build the Foundation | Complete RAG pipeline from scratch                      |
| **Part 2** | Debug RAG Failures   | Fix the 5 failure modes that break most RAG systems     |
| **Part 3** | Optimize for Quality | Semantic chunking, reranking, multi-query retrieval     |
| **Part 4** | Scale for Production | Caching, monitoring, production vector databases        |
| **Part 5** | Production Project   | Full production-ready RAG application                   |
| **Part 6** | Advanced RAG 2026    | Agentic RAG, GraphRAG, Contextual Retrieval, Multimodal |

---

## The 5 RAG Failure Modes

Most RAG projects fail for the same 5 reasons:

| #   | Failure Mode           | What Happens                                       | The Fix                          |
| --- | ---------------------- | -------------------------------------------------- | -------------------------------- |
| 1   | **Bad Chunking**       | Wrong context retrieved, chunks split mid-sentence | Semantic chunking, smart overlap |
| 2   | **Embedding Mismatch** | User says "cancel", docs say "termination policy"  | Query rewriting, hybrid search   |
| 3   | **Retrieval Noise**    | 10 docs retrieved, only 2 are relevant             | Reranking, filtering             |
| 4   | **Context Overflow**   | Too much stuffed in prompt, LLM ignores half       | Smart truncation, map-reduce     |
| 5   | **Hallucination**      | Answer is in context but LLM makes things up       | Constrained prompts, citations   |

This course teaches you to **diagnose and fix each one**.

---

## Tech Stack

- **Python 3.10+**
- **LangChain 1.x** (2026 stable release)
- **LangGraph 1.x** for Agentic RAG
- **ChromaDB** for local development
- **OpenAI** for embeddings and LLMs
- **Streamlit** for production UI

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/pdichone/production-rag-course.git
cd production-rag-course
```

### 2. Set up your environment

```bash
cd code/part1-foundation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or with `uv` (faster):

```bash
uv venv && uv pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Run the first example

```bash
python 01_document_loading.py
```

---

## Course Structure

```
production-rag-course/
├──
│   │
│   └── part6-advanced/        # 2026 cutting-edge
│       ├── 01_long_context_vs_rag.py
│       ├── 02_contextual_retrieval.py
│       ├── 03_late_chunking.py
│       ├── 04_agentic_rag.py
│       ├── 05_graphrag_intro.py
│       └── 06_multimodal_rag.py
│

```

---

## Part 6: Advanced RAG 2026

The cutting-edge techniques that separate production RAG from tutorials:

| Technique                | What It Solves                                 | Improvement                  |
| ------------------------ | ---------------------------------------------- | ---------------------------- |
| **Long Context vs RAG**  | When to use 1M token windows vs retrieval      | Cost optimization            |
| **Contextual Retrieval** | Chunks losing document context                 | 67% fewer retrieval failures |
| **Late Chunking**        | Cross-chunk context lost in embeddings         | 10-12% accuracy boost        |
| **Agentic RAG**          | One-shot retrieval missing info                | Self-correcting loops        |
| **GraphRAG**             | Multi-hop reasoning failures                   | Relationship traversal       |
| **Multimodal RAG**       | Tables and charts destroyed by text extraction | Vision-based retrieval       |

---

## Free Resources

### Production AI Checklist

Get the free checklist for deploying AI applications to production - covers testing, monitoring, security, and scaling:

**[Download the Production AI Checklist](https://bit.ly/production-ai-pack)**

---

## Join the Community

### AI Guild

Join thousands of AI engineers building production AI systems:

- Live weekly sessions on AI development
- Code reviews and architecture feedback
- Private community of practitioners
- Early access to new courses and content

**[Join the AI Guild](https://bit.ly/ai-guild)**

---

## Prerequisites

| Requirement | Level                                    |
| ----------- | ---------------------------------------- |
| **Python**  | Comfortable with functions, classes, pip |
| **APIs**    | Basic understanding of REST APIs         |
| **LLMs**    | Helpful but not required                 |
| **ML/AI**   | Not required - we explain everything     |

---

## About the Instructor

**Paulo Dichone** - AI Engineer & Educator

- 350,000+ students taught across platforms
- 70+ courses on AI, Python, and mobile development
- Building AI systems in production since 2015
- Creator of the AI Developer Masterclass and Vector Databases Masterclass

---

## FAQ

**Q: Is this different from other RAG tutorials?**

Yes. Most tutorials show you how to build RAG. This course shows you why RAG breaks and how to fix it. The "5 Failure Modes" framework comes from teaching 300K+ students and seeing the same problems repeatedly.

**Q: Do I need a GPU?**

No. All code runs on CPU. Part 6's Multimodal RAG section mentions GPU for ColPali, but it's optional.

**Q: Which vector database should I use?**

We use ChromaDB for simplicity. The patterns apply to Pinecone, Weaviate, Supabase, or any vector store.

**Q: Is the code up to date?**

Yes. All code uses LangChain 1.x (2026 stable release) with current best practices.

---

## License

This course code is available under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Issues:** Open a GitHub issue for bugs or questions
- **Community:** Join the [AI Guild](https://bit.ly/ai-guild) for direct support
- **Updates:** Star this repo to get notified of updates

---

**Ready to build RAG that actually works in production? Let's go.**
