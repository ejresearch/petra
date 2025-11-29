"""
MCP Article Server

Serves AI newsletter articles for the briefing system.
Fetches real articles from RSS feeds of popular AI newsletters.
"""

import asyncio
import aiohttp
import feedparser
import logging
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import html
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Article Server", version="2.0.0")

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
    sources_checked: List[str]
    retrieved_at: str


# ============================================================================
# RSS FEED CONFIGURATION
# ============================================================================

@dataclass
class FeedConfig:
    """Configuration for an RSS feed source."""
    name: str
    url: str
    enabled: bool = True


# AI Newsletter RSS Feeds
RSS_FEEDS = [
    # Major AI Newsletters
    FeedConfig("TLDR AI", "https://tldr.tech/ai/rss"),
    FeedConfig("Import AI", "https://importai.substack.com/feed"),
    FeedConfig("The Neuron", "https://www.theneurondaily.com/feed"),
    FeedConfig("Ben's Bites", "https://bensbites.beehiiv.com/feed"),
    FeedConfig("The Rundown AI", "https://www.therundown.ai/feed"),
    FeedConfig("Superhuman AI", "https://superhumanai.beehiiv.com/feed"),
    FeedConfig("AI Supremacy", "https://aisupremacy.substack.com/feed"),
    FeedConfig("The Algorithm (MIT)", "https://www.technologyreview.com/feed/"),
    FeedConfig("Last Week in AI", "https://lastweekin.ai/feed"),
    FeedConfig("AI Weekly", "https://aiweekly.co/feed"),

    # Substack AI Writers
    FeedConfig("One Useful Thing", "https://www.oneusefulthing.org/feed"),
    FeedConfig("The Batch (DeepLearning.AI)", "https://www.deeplearning.ai/the-batch/feed/"),
    FeedConfig("AI Snake Oil", "https://www.aisnakeoil.com/feed"),
    FeedConfig("Interconnects", "https://www.interconnects.ai/feed"),
    FeedConfig("Semi Analysis", "https://semianalysis.substack.com/feed"),

    # Tech News with AI Coverage
    FeedConfig("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    FeedConfig("Ars Technica AI", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    FeedConfig("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    FeedConfig("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    FeedConfig("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss"),
]


# ============================================================================
# HTML/TEXT UTILITIES
# ============================================================================

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    clean = html.unescape(clean)
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def extract_text(entry: dict) -> str:
    """Extract text content from RSS entry."""
    # Try different fields where content might be
    content = ""

    # Check content:encoded (full article)
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].get('value', '')

    # Check summary/description
    if not content and hasattr(entry, 'summary'):
        content = entry.summary

    # Check description
    if not content and hasattr(entry, 'description'):
        content = entry.description

    # Strip HTML and clean up
    text = strip_html(content)

    # Limit length (keep first ~2000 chars for processing)
    if len(text) > 2000:
        text = text[:2000] + "..."

    return text


def parse_date(entry: dict) -> Optional[datetime]:
    """Parse date from RSS entry."""
    # Try published_parsed first
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6])
        except:
            pass

    # Try updated_parsed
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6])
        except:
            pass

    # Default to now
    return datetime.utcnow()


# ============================================================================
# RSS FETCHER
# ============================================================================

class RSSFetcher:
    """Fetches and parses RSS feeds."""

    def __init__(self, feeds: List[FeedConfig]):
        self.feeds = [f for f in feeds if f.enabled]
        self._cache: dict = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_fetch: Optional[datetime] = None

    async def fetch_feed(self, session: aiohttp.ClientSession, feed: FeedConfig) -> List[Article]:
        """Fetch a single RSS feed."""
        articles = []
        now = datetime.utcnow()

        try:
            logger.info(f"Fetching: {feed.name} from {feed.url}")

            async with session.get(feed.url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {feed.name}: HTTP {response.status}")
                    return articles

                content = await response.text()
                parsed = feedparser.parse(content)

                if parsed.bozo and not parsed.entries:
                    logger.warning(f"Failed to parse {feed.name}: {parsed.bozo_exception}")
                    return articles

                for entry in parsed.entries[:20]:  # Limit entries per feed
                    try:
                        title = strip_html(getattr(entry, 'title', 'Untitled'))
                        url = getattr(entry, 'link', '')
                        text = extract_text(entry)
                        published = parse_date(entry)

                        if not url or not title:
                            continue

                        articles.append(Article(
                            source=feed.name,
                            url=url,
                            title=title,
                            published=published.isoformat() + "Z",
                            text=text if text else f"Read more at {url}",
                            retrieved_at=now.isoformat() + "Z"
                        ))
                    except Exception as e:
                        logger.warning(f"Error parsing entry from {feed.name}: {e}")
                        continue

                logger.info(f"Got {len(articles)} articles from {feed.name}")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {feed.name}")
        except Exception as e:
            logger.error(f"Error fetching {feed.name}: {e}")

        return articles

    async def fetch_all(self, since_date: Optional[datetime] = None, limit: int = 50) -> tuple[List[Article], List[str]]:
        """Fetch all RSS feeds in parallel."""
        all_articles = []
        sources_checked = []

        async with aiohttp.ClientSession(
            headers={"User-Agent": "AI-Briefing-System/1.0"}
        ) as session:
            # Fetch all feeds in parallel
            tasks = [self.fetch_feed(session, feed) for feed in self.feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for feed, result in zip(self.feeds, results):
                sources_checked.append(feed.name)
                if isinstance(result, Exception):
                    logger.error(f"Error from {feed.name}: {result}")
                    continue
                all_articles.extend(result)

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Filter by date if specified
        if since_date:
            filtered = []
            for article in unique_articles:
                try:
                    pub_date = datetime.fromisoformat(article.published.replace("Z", ""))
                    if pub_date >= since_date:
                        filtered.append(article)
                except:
                    filtered.append(article)  # Keep if can't parse date
            unique_articles = filtered

        # Sort by published date (newest first)
        unique_articles.sort(
            key=lambda x: x.published,
            reverse=True
        )

        # Limit results
        unique_articles = unique_articles[:limit]

        logger.info(f"Total: {len(unique_articles)} unique articles from {len(sources_checked)} sources")

        return unique_articles, sources_checked


# Initialize fetcher
fetcher = RSSFetcher(RSS_FEEDS)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API info."""
    return {
        "service": "MCP Article Server",
        "version": "2.0.0",
        "description": "Fetches real AI newsletter articles via RSS",
        "sources": [f.name for f in RSS_FEEDS if f.enabled],
        "endpoints": {
            "/articles": "Get articles (with optional ?since=YYYY-MM-DD filter)",
            "/sources": "List configured RSS sources",
            "/health": "Health check"
        }
    }


@app.get("/sources")
async def get_sources():
    """List configured RSS feed sources."""
    return {
        "sources": [
            {"name": f.name, "url": f.url, "enabled": f.enabled}
            for f in RSS_FEEDS
        ]
    }


@app.get("/articles", response_model=ArticleResponse)
async def get_articles(
    since: Optional[str] = Query(None, description="Filter articles since date (YYYY-MM-DD)"),
    limit: int = Query(50, description="Maximum number of articles to return", le=100)
):
    """
    Fetch articles from AI newsletter RSS feeds.

    - **since**: Optional date filter (YYYY-MM-DD format)
    - **limit**: Maximum number of articles (default 50, max 100)
    """
    since_date = None
    if since:
        try:
            since_date = datetime.fromisoformat(since.replace("Z", ""))
        except ValueError:
            try:
                since_date = datetime.strptime(since, "%Y-%m-%d")
            except ValueError:
                pass

    articles, sources_checked = await fetcher.fetch_all(since_date=since_date, limit=limit)

    return ArticleResponse(
        status="success",
        count=len(articles),
        articles=articles,
        sources_checked=sources_checked,
        retrieved_at=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "MCP Article Server",
        "version": "2.0.0",
        "sources_configured": len([f for f in RSS_FEEDS if f.enabled]),
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
        port=8002,
        log_level="info"
    )
