"""
MCP Article Server

Serves AI newsletter articles for the briefing system.
This is a mock server for development/testing. In production, this would
connect to real newsletter sources via RSS, APIs, or web scraping.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random
import hashlib

app = FastAPI(title="MCP Article Server", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATA MODELS
# ============================================================================

class Article(BaseModel):
    """Article schema matching Node 2 specification."""
    source: str
    url: str
    title: str
    published: str  # ISO-8601
    text: str
    retrieved_at: str  # ISO-8601


class ArticleResponse(BaseModel):
    """Response schema for article endpoint."""
    status: str
    count: int
    articles: List[Article]
    retrieved_at: str


# ============================================================================
# MOCK DATA GENERATOR
# ============================================================================

SOURCES = [
    "The Rundown AI",
    "TLDR AI",
    "The Neuron",
    "AI Breakfast",
    "Ben's Bites",
    "The AI Report",
    "Import AI",
    "Last Week in AI"
]

MOCK_ARTICLES = [
    {
        "title": "OpenAI Announces GPT-5 with Revolutionary Reasoning Capabilities",
        "text": "OpenAI has unveiled GPT-5, their latest large language model featuring unprecedented reasoning capabilities. The new model demonstrates significant improvements in mathematical problem-solving, code generation, and multi-step logical reasoning. Early benchmarks show GPT-5 outperforming previous models by 40% on complex reasoning tasks. The model also introduces a new 'chain of thought' mechanism that allows users to see the model's reasoning process in real-time. Industry experts predict this release will accelerate AI adoption across enterprise applications.",
        "topics": ["LLMs", "AI Product Launches"]
    },
    {
        "title": "Anthropic Releases Claude 4 with Enhanced Safety Features",
        "text": "Anthropic has launched Claude 4, featuring advanced constitutional AI techniques and improved safety guardrails. The new model includes better instruction following, reduced hallucinations, and enhanced ability to decline harmful requests while remaining helpful. Claude 4 also introduces 'artifact mode' for generating structured outputs like code, documents, and data analysis. Enterprise customers report 60% reduction in moderation issues compared to previous versions.",
        "topics": ["LLMs", "AI Safety & Ethics"]
    },
    {
        "title": "Google DeepMind's AlphaFold 3 Predicts Drug Interactions",
        "text": "Google DeepMind has extended AlphaFold to predict drug-protein interactions with 95% accuracy. AlphaFold 3 can now model how potential drug molecules will bind to protein targets, dramatically accelerating pharmaceutical research. Several major drug companies have already integrated the tool into their discovery pipelines. Researchers estimate this could reduce drug development timelines by 2-3 years and save billions in research costs.",
        "topics": ["AI in Healthcare", "AI Research & Papers"]
    },
    {
        "title": "Microsoft Copilot Now Handles 100-Step Workflows Autonomously",
        "text": "Microsoft has upgraded Copilot with autonomous agent capabilities, enabling it to execute complex multi-step workflows without human intervention. The new 'Copilot Agents' can handle tasks like expense report processing, meeting scheduling, and data analysis across multiple applications. Early enterprise adopters report 70% time savings on routine administrative tasks. The feature is rolling out to Microsoft 365 Enterprise customers this month.",
        "topics": ["Agents & Automation", "AI in Business"]
    },
    {
        "title": "EU AI Act Implementation Begins: What Companies Need to Know",
        "text": "The European Union's AI Act has entered its implementation phase, with the first compliance deadlines approaching. Companies must now classify their AI systems by risk level and implement appropriate safeguards. High-risk applications in healthcare, education, and employment face the strictest requirements. Legal experts recommend companies begin compliance audits immediately, as penalties can reach 7% of global revenue for violations.",
        "topics": ["AI Regulation & Policy", "AI Safety & Ethics"]
    },
    {
        "title": "Startup Raises $500M for Autonomous Coding Agents",
        "text": "AI coding startup Cognition has raised $500 million at a $2 billion valuation for its Devin autonomous coding agent. Devin can independently handle entire software development tasks, from requirements analysis to deployment. The company reports that Devin has completed over 10,000 real-world coding tasks with 94% success rate. Investors include major tech companies and leading AI-focused venture funds.",
        "topics": ["AI Startups & Funding", "Agents & Automation"]
    },
    {
        "title": "New Research Shows LLMs Can Learn from Single Examples",
        "text": "Researchers at Stanford and Berkeley have demonstrated that large language models can effectively learn new tasks from just one example, challenging assumptions about AI training requirements. The 'one-shot meta-learning' technique enables rapid adaptation without fine-tuning. This breakthrough could dramatically reduce the cost and time required to customize AI models for specific applications.",
        "topics": ["AI Research & Papers", "LLMs"]
    },
    {
        "title": "Apple Intelligence Expands with On-Device Image Generation",
        "text": "Apple has expanded Apple Intelligence with on-device image generation capabilities that run entirely on iPhone and Mac hardware. The feature enables private, fast image creation without cloud processing. Apple claims the on-device model produces results comparable to cloud-based alternatives while maintaining user privacy. The update will be available to all Apple Intelligence-supported devices next month.",
        "topics": ["AI Product Launches", "Edge AI & On-Device"]
    },
    {
        "title": "Healthcare AI Detects Cancer 5 Years Before Traditional Methods",
        "text": "A new AI system developed by MIT and Mass General Hospital can detect pancreatic cancer up to 5 years before conventional diagnosis. The model analyzes routine medical records and lab results to identify subtle patterns predictive of future cancer development. Clinical trials show 87% accuracy in early detection, potentially saving thousands of lives annually through earlier intervention.",
        "topics": ["AI in Healthcare", "AI Research & Papers"]
    },
    {
        "title": "Tesla FSD V13 Achieves Level 4 Autonomy in Limited Conditions",
        "text": "Tesla's Full Self-Driving version 13 has achieved Level 4 autonomy certification for highway driving in select states. The update enables true hands-free operation without driver monitoring under specific conditions. Tesla reports zero at-fault accidents in over 10 million miles of V13 testing. Regulatory approval for broader deployment is expected within six months.",
        "topics": ["Autonomous Vehicles", "AI Product Launches"]
    },
    {
        "title": "Open Source AI Alliance Releases 70B Parameter Model",
        "text": "The Open Source AI Alliance, backed by Meta, IBM, and Intel, has released a new 70 billion parameter model under Apache 2.0 license. The model matches GPT-4 performance on most benchmarks while being fully open for commercial use. The release includes training code, datasets, and fine-tuning guides. Researchers praise the move as a major step toward democratizing advanced AI.",
        "topics": ["Open Source AI", "LLMs"]
    },
    {
        "title": "AI-Powered Robots Enter Amazon Warehouses",
        "text": "Amazon has deployed 10,000 AI-powered humanoid robots across its fulfillment centers. The robots, developed by Figure AI, can pick, pack, and sort items with 99% accuracy. Workers report that robots handle the most physically demanding tasks, reducing injuries by 40%. Amazon plans to expand the robot workforce to 100,000 units by end of year.",
        "topics": ["Robotics & Physical AI", "AI in Business"]
    },
    {
        "title": "New Jailbreak Attack Affects All Major LLMs",
        "text": "Security researchers have discovered a universal jailbreak technique that bypasses safety measures in ChatGPT, Claude, Gemini, and other major LLMs. The 'prompt injection cascade' attack exploits how models process nested instructions. AI companies are racing to patch the vulnerability, highlighting ongoing challenges in AI safety. Researchers recommend additional input validation for production AI applications.",
        "topics": ["AI Safety & Ethics", "AI Research & Papers"]
    },
    {
        "title": "Nvidia Unveils B200 GPU with 3x AI Performance",
        "text": "Nvidia has announced the B200 Blackwell GPU, delivering 3x the AI training performance of the H100. The new chip features 192GB of HBM3e memory and improved transformer engine for large language models. Cloud providers are already ordering millions of units, though supply constraints are expected through next year. The B200 will power the next generation of AI infrastructure.",
        "topics": ["AI Hardware & Infrastructure", "AI Product Launches"]
    },
    {
        "title": "Synthetic Data Market Grows 200% as Privacy Concerns Rise",
        "text": "The synthetic data generation market has grown 200% year-over-year as companies seek privacy-compliant AI training alternatives. New techniques can generate realistic datasets that preserve statistical properties without exposing real user information. Major enterprises are adopting synthetic data for healthcare, finance, and customer analytics applications where privacy regulations restrict real data use.",
        "topics": ["Data & Privacy", "AI in Business"]
    },
    {
        "title": "AI Writing Assistants Now Used by 60% of Knowledge Workers",
        "text": "A new survey reveals that 60% of knowledge workers regularly use AI writing assistants for email, reports, and documentation. The most common use cases include drafting initial content, editing for clarity, and translating between languages. Workers report average time savings of 2 hours per day, though concerns about over-reliance and skill atrophy persist.",
        "topics": ["AI in Business", "LLMs"]
    },
    {
        "title": "Researchers Achieve Breakthrough in AI Energy Efficiency",
        "text": "MIT researchers have developed a new neural network architecture that reduces AI inference energy consumption by 90%. The 'sparse activation' technique only activates necessary neurons for each task, dramatically cutting power requirements. The breakthrough could enable advanced AI on mobile devices and reduce the environmental impact of large-scale AI deployments.",
        "topics": ["AI Research & Papers", "Edge AI & On-Device"]
    },
    {
        "title": "China Releases Open-Weight Model Rivaling GPT-4",
        "text": "Chinese AI lab Zhipu has released GLM-5, an open-weight model matching GPT-4 performance on Chinese and English benchmarks. The model is available for commercial use with minimal restrictions, challenging Western dominance in frontier AI. Analysts note the rapid pace of Chinese AI development despite export controls on advanced chips.",
        "topics": ["LLMs", "AI Regulation & Policy"]
    },
    {
        "title": "AI Tutors Show 30% Learning Improvement in Schools",
        "text": "A large-scale study across 500 schools shows AI tutoring systems improve student learning outcomes by 30% on average. The personalized AI tutors adapt to individual learning styles and pace, providing immediate feedback on practice problems. Critics raise concerns about screen time and the role of human teachers, while proponents highlight potential to address teacher shortages.",
        "topics": ["AI in Education", "AI Research & Papers"]
    },
    {
        "title": "Venture Funding for AI Startups Reaches $100B in 2025",
        "text": "Global venture funding for AI startups has surpassed $100 billion in 2025, doubling from the previous year. The largest deals focus on foundation models, enterprise AI applications, and autonomous systems. Despite the funding surge, valuations remain high with many companies yet to demonstrate sustainable revenue. Investors increasingly focus on AI infrastructure and vertical applications over general-purpose models.",
        "topics": ["AI Startups & Funding", "AI in Business"]
    }
]


def generate_url(title: str, source: str) -> str:
    """Generate a deterministic mock URL for an article."""
    slug = title.lower().replace(" ", "-")[:50]
    hash_suffix = hashlib.md5(f"{title}{source}".encode()).hexdigest()[:8]
    domain = source.lower().replace(" ", "").replace("'", "")
    return f"https://{domain}.com/articles/{slug}-{hash_suffix}"


def get_mock_articles(since_date: Optional[datetime] = None, limit: int = 50) -> List[Article]:
    """Generate mock articles with realistic timestamps."""
    now = datetime.utcnow()
    articles = []

    for i, mock in enumerate(MOCK_ARTICLES):
        # Generate a random timestamp within the last 24 hours
        hours_ago = random.uniform(0, 24)
        published = now - timedelta(hours=hours_ago)

        # Skip if before since_date
        if since_date and published < since_date:
            continue

        source = random.choice(SOURCES)

        article = Article(
            source=source,
            url=generate_url(mock["title"], source),
            title=mock["title"],
            published=published.isoformat() + "Z",
            text=mock["text"],
            retrieved_at=now.isoformat() + "Z"
        )
        articles.append(article)

        if len(articles) >= limit:
            break

    # Sort by published date (newest first)
    articles.sort(key=lambda x: x.published, reverse=True)

    return articles


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API info."""
    return {
        "service": "MCP Article Server",
        "version": "1.0.0",
        "endpoints": {
            "/articles": "Get articles (with optional ?since=YYYY-MM-DD filter)",
            "/health": "Health check"
        }
    }


@app.get("/articles", response_model=ArticleResponse)
async def get_articles(
    since: Optional[str] = Query(None, description="Filter articles since date (YYYY-MM-DD)"),
    limit: int = Query(50, description="Maximum number of articles to return")
):
    """
    Fetch articles from AI newsletters.

    - **since**: Optional date filter (YYYY-MM-DD format)
    - **limit**: Maximum number of articles (default 50)
    """
    since_date = None
    if since:
        try:
            since_date = datetime.fromisoformat(since.replace("Z", ""))
        except ValueError:
            # Try parsing as date only
            try:
                since_date = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                pass

    articles = get_mock_articles(since_date=since_date, limit=limit)

    return ArticleResponse(
        status="success",
        count=len(articles),
        articles=articles,
        retrieved_at=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "MCP Article Server",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,  # Different port from Node 1
        log_level="info"
    )
