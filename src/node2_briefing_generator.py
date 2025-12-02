"""
Node 2: Daily Briefing Generator

Generates personalized AI briefings with three sections:
1. The Landscape - Overview of what's happening across AI today
2. Your Top 5 - Personalized article picks based on user topics
3. Three Deep Dives - Hot topics with substantive analysis

Architecture:
- 20 per-site agents process articles in parallel
- Landscape generator synthesizes the big picture
- Top 5 selector ranks by user relevance
- Deep dive generator analyzes hot themes

Can be run standalone or triggered by Enso workflow.
"""

import os
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from prompts import (
    build_site_agent_prompt,
    build_landscape_prompt,
    build_top5_prompt,
    build_deep_dive_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration settings for Node 2."""
    # Article Service (can be local or Render URL)
    article_service_url: str = os.getenv("ARTICLE_SERVICE_URL", "http://localhost:8002")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.1")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Email (Resend - get API key at resend.com)
    resend_api_key: str = os.getenv("RESEND_API_KEY", "")
    from_email: str = os.getenv("FROM_EMAIL", "briefing@resend.dev")  # Use your verified domain

    # Paths
    profiles_path: str = os.getenv("PROFILES_PATH", "user_profiles.jsonl")
    template_path: str = os.getenv("TEMPLATE_PATH", "src/node2_email_template.html")

    # Settings
    max_articles: int = int(os.getenv("MAX_ARTICLES", "50"))
    top_n_articles: int = int(os.getenv("TOP_N_ARTICLES", "5"))
    num_clusters: int = int(os.getenv("NUM_CLUSTERS", "3"))


config = Config()


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UserProfile:
    """User profile from Node 1."""
    email: str
    name: str
    briefing_time: str
    topics: List[str]
    created_at: str
    version: str = "1.0"


@dataclass
class Article:
    """Article from MCP server."""
    source: str
    url: str
    title: str
    published: str
    text: str
    retrieved_at: str


@dataclass
class ProcessedArticle:
    """Article processed by per-site agent."""
    source: str
    url: str
    title: str
    summary: str
    relevance: float
    keywords: List[str]
    why_selected: str = ""  # Added by top-5 selector
    rank: int = 0


@dataclass
class Landscape:
    """The Landscape section - overview of AI today."""
    content: str  # 3-4 paragraphs of prose
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class DeepDive:
    """A single deep dive topic."""
    topic: str
    hook: str
    analysis: str
    related_articles: List[str]


@dataclass
class Briefing:
    """Complete briefing with all three sections."""
    landscape: Landscape
    top_5: List[ProcessedArticle]
    deep_dives: List[DeepDive]
    articles_analyzed: int
    sources_count: int
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class BriefingResult:
    """Result of briefing generation."""
    user_email: str
    articles_fetched: int
    articles_processed: int
    top_articles: int
    email_sent: bool
    status: str
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


# ============================================================================
# USER PROFILE LOADER
# ============================================================================

class ProfileLoader:
    """Load user profiles from storage."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load_profiles(self) -> List[UserProfile]:
        """Load all user profiles."""
        profiles = []

        if not self.path.exists():
            logger.warning(f"Profiles file not found: {self.path}")
            return profiles

        with open(self.path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        profiles.append(UserProfile(**data))
                    except Exception as e:
                        logger.error(f"Error parsing profile: {e}")

        logger.info(f"Loaded {len(profiles)} user profiles")
        return profiles

    def filter_by_time(self, profiles: List[UserProfile], target_time: str) -> List[UserProfile]:
        """Filter profiles by briefing time (HH:MM format)."""
        return [p for p in profiles if p.briefing_time == target_time]


# ============================================================================
# ARTICLE FETCHER
# ============================================================================

class ArticleFetcher:
    """Fetch articles from MCP server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def fetch_articles(self, since_date: Optional[str] = None, limit: int = 50) -> List[Article]:
        """Fetch articles from MCP server."""
        url = f"{self.base_url}/articles"
        params = {"limit": limit}

        if since_date:
            params["since"] = since_date

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"MCP server error: {response.status}")
                        return []

                    data = await response.json()
                    articles = [Article(**a) for a in data.get("articles", [])]
                    logger.info(f"Fetched {len(articles)} articles from MCP")
                    return articles

        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []

    def deduplicate(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles by URL."""
        seen_urls = set()
        unique = []

        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique.append(article)

        logger.info(f"Deduplicated: {len(articles)} -> {len(unique)} articles")
        return unique

    def filter_recent(self, articles: List[Article], hours: int = 24) -> List[Article]:
        """Filter articles from last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        recent = []
        for article in articles:
            try:
                published = datetime.fromisoformat(article.published.replace("Z", ""))
                if published > cutoff:
                    recent.append(article)
            except Exception as e:
                logger.warning(f"Error parsing date: {e}")

        logger.info(f"Filtered to {len(recent)} recent articles")
        return recent

    def group_by_source(self, articles: List[Article]) -> Dict[str, List[Article]]:
        """Group articles by their source for per-site agent processing."""
        grouped: Dict[str, List[Article]] = {}
        for article in articles:
            if article.source not in grouped:
                grouped[article.source] = []
            grouped[article.source].append(article)

        logger.info(f"Grouped into {len(grouped)} sources")
        return grouped


# ============================================================================
# LLM PROCESSOR (New Architecture)
# ============================================================================

class LLMProcessor:
    """
    Process articles using OpenAI LLM with system prompts.

    New architecture:
    1. Per-site agents (20 parallel) - summarize and score articles
    2. Landscape generator - synthesize the big picture
    3. Top 5 selector - rank by user relevance
    4. Deep dive generator - analyze hot themes
    """

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        parse_json: bool = True
    ) -> Optional[Any]:
        """Make an LLM API call with system prompt."""
        if not self.api_key:
            logger.error("OpenAI API key not configured")
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"LLM API error: {response.status} - {error}")
                        return None

                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]

                    if not parse_json:
                        return content.strip()

                    # Parse JSON from response
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return None

    async def process_site(
        self,
        source: str,
        articles: List[Article],
        user_topics: List[str]
    ) -> List[Dict]:
        """
        Per-site agent: process all articles from one source.
        Returns list of processed article dicts.
        """
        if not articles:
            return []

        # Build prompts with user topics injected
        system_prompt, user_prompt = build_site_agent_prompt(
            source=source,
            user_topics=user_topics,
            articles=[{"title": a.title, "text": a.text} for a in articles]
        )

        logger.info(f"Processing {len(articles)} articles from {source}")
        result = await self._call_llm(system_prompt, user_prompt, max_tokens=2000)

        if not result:
            logger.warning(f"No result from site agent for {source}")
            return []

        # Ensure result is a list
        if isinstance(result, dict) and "articles" in result:
            result = result["articles"]

        # Add source to each article
        for article in result:
            article["source"] = source
            # Find original URL
            for orig in articles:
                if orig.title == article.get("title"):
                    article["url"] = orig.url
                    break

        return result

    async def process_all_sites_parallel(
        self,
        articles_by_source: Dict[str, List[Article]],
        user_topics: List[str]
    ) -> List[Dict]:
        """
        Run all per-site agents in parallel.
        Returns combined list of processed articles.
        """
        logger.info(f"Processing {len(articles_by_source)} sources in parallel")

        tasks = [
            self.process_site(source, articles, user_topics)
            for source, articles in articles_by_source.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all results
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Site agent error: {result}")
                continue
            if result:
                all_articles.extend(result)

        logger.info(f"Total processed articles: {len(all_articles)}")
        return all_articles

    async def generate_landscape(
        self,
        source_summaries: Dict[str, List[Dict]],
        user_topics: List[str],
        total_articles: int
    ) -> Optional[str]:
        """
        Generate the Landscape section - overview of AI today.
        Returns prose text (not JSON).
        """
        system_prompt, user_prompt = build_landscape_prompt(
            user_topics=user_topics,
            source_summaries=source_summaries,
            total_articles=total_articles
        )

        logger.info("Generating landscape summary")
        result = await self._call_llm(
            system_prompt,
            user_prompt,
            max_tokens=800,
            temperature=0.8,
            parse_json=False
        )

        return result

    async def select_top_5(
        self,
        articles: List[Dict],
        user_topics: List[str]
    ) -> List[Dict]:
        """
        Select top 5 articles for this user.
        Returns list of 5 articles with why_selected added.
        """
        # Pre-sort by relevance score
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get("relevance", 0),
            reverse=True
        )[:20]  # Send top 20 candidates to LLM

        # Build lookup by title for enrichment
        article_lookup = {a.get("title", ""): a for a in sorted_articles}

        system_prompt, user_prompt = build_top5_prompt(
            user_topics=user_topics,
            articles=sorted_articles
        )

        logger.info("Selecting top 5 articles")
        result = await self._call_llm(system_prompt, user_prompt, max_tokens=1500)

        if result and "top_5" in result:
            # Enrich LLM response with original article data (source, url, keywords)
            enriched = []
            for item in result["top_5"]:
                title = item.get("title", "")
                original = article_lookup.get(title, {})
                enriched.append({
                    **original,  # source, url, keywords, relevance from original
                    **item,      # rank, why_selected from LLM
                })
            return enriched

        # Fallback: just return top 5 by relevance
        logger.warning("Top 5 selection failed, using relevance fallback")
        return sorted_articles[:5]

    async def generate_deep_dives(
        self,
        articles: List[Dict],
        user_topics: List[str]
    ) -> List[Dict]:
        """
        Generate 3 deep dive topics based on what's hot.
        Returns list of 3 deep dive dicts.
        """
        system_prompt, user_prompt = build_deep_dive_prompt(
            user_topics=user_topics,
            articles=articles
        )

        logger.info("Generating deep dives")
        result = await self._call_llm(system_prompt, user_prompt, max_tokens=2000)

        if result and "deep_dives" in result:
            return result["deep_dives"]

        return []


# ============================================================================
# EMAIL SENDER (Resend API)
# ============================================================================

class EmailSender:
    """Send briefing emails via Resend API."""

    def __init__(self, api_key: str, from_email: str):
        self.api_key = api_key
        self.from_email = from_email
        self.api_url = "https://api.resend.com/emails"

    def _load_template(self, template_path: str) -> Template:
        """Load Jinja2 email template."""
        with open(template_path, 'r') as f:
            return Template(f.read())

    def compose_email(
        self,
        template_path: str,
        user: UserProfile,
        articles: List[ProcessedArticle],
        clusters: List,
        briefing_date: str,
        articles_analyzed: int
    ) -> str:
        """Compose HTML email from template (legacy method)."""
        template = self._load_template(template_path)

        # Prepare article data
        articles_data = [
            {
                "rank": a.rank,
                "source": a.source,
                "title": a.title,
                "url": a.url,
                "summary": a.summary,
                "why_selected": a.why_selected,
                "keywords": a.keywords,
            }
            for a in articles
        ]

        return template.render(
            user_name=user.name or "there",
            briefing_date=briefing_date,
            articles_analyzed=articles_analyzed,
            article_count=len(articles),
            top_5_articles=articles_data,
            user_topics=user.topics,
            preferences_url="#",
            unsubscribe_url="#"
        )

    def compose_briefing_email(
        self,
        template_path: str,
        user: UserProfile,
        briefing: 'Briefing',
        briefing_date: str
    ) -> str:
        """
        Compose HTML email with the new briefing structure:
        1. The Landscape
        2. Your Top 5
        3. Three Deep Dives
        """
        template = self._load_template(template_path)

        # Prepare top 5 data
        top_5_data = [
            {
                "rank": a.rank,
                "source": a.source,
                "title": a.title,
                "url": a.url,
                "summary": a.summary,
                "why_selected": a.why_selected,
                "keywords": a.keywords,
            }
            for a in briefing.top_5
        ]

        # Prepare deep dive data
        deep_dives_data = [
            {
                "topic": d.topic,
                "hook": d.hook,
                "analysis": d.analysis,
                "related_articles": d.related_articles
            }
            for d in briefing.deep_dives
        ]

        return template.render(
            user_name=user.name or "there",
            briefing_date=briefing_date,
            # The Landscape
            landscape=briefing.landscape.content,
            # Your Top 5
            top_5_articles=top_5_data,
            # Three Deep Dives
            deep_dives=deep_dives_data,
            # Meta
            articles_analyzed=briefing.articles_analyzed,
            sources_count=briefing.sources_count,
            user_topics=user.topics,
            preferences_url="#",
            unsubscribe_url="#"
        )

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str
    ) -> bool:
        """Send email via Resend API."""
        if not self.api_key:
            logger.warning("RESEND_API_KEY not configured, skipping send")
            return False

        try:
            import requests

            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": self.from_email,
                    "to": [to_email],
                    "subject": subject,
                    "html": html_body
                }
            )

            if response.status_code == 200:
                logger.info(f"Email sent to {to_email}")
                return True
            else:
                logger.error(f"Resend API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


# ============================================================================
# MAIN WORKFLOW (New Architecture)
# ============================================================================

class BriefingGenerator:
    """
    Main workflow orchestrator.

    New flow:
    1. Fetch articles and group by source
    2. Run 20 per-site agents in parallel (with user topics in system prompt)
    3. Generate Landscape summary
    4. Select Top 5 articles
    5. Generate 3 Deep Dives
    6. Compose and send email
    """

    def __init__(self, cfg: Config):
        self.config = cfg
        self.profile_loader = ProfileLoader(cfg.profiles_path)
        self.article_fetcher = ArticleFetcher(cfg.article_service_url)
        self.llm_processor = LLMProcessor(cfg.openai_api_key, cfg.openai_model, cfg.openai_base_url)
        self.email_sender = EmailSender(cfg.resend_api_key, cfg.from_email)

    async def generate_briefing_for_user(
        self,
        user: UserProfile,
        articles_by_source: Dict[str, List[Article]],
        total_articles: int
    ) -> BriefingResult:
        """
        Generate and send briefing for a single user.

        Flow:
        1. Per-site agents process articles (20 parallel)
        2. Generate Landscape
        3. Select Top 5
        4. Generate Deep Dives
        5. Compose and send email
        """
        logger.info(f"Generating briefing for {user.email}")
        logger.info(f"User topics: {user.topics}")

        try:
            # Step 1: Run per-site agents in parallel
            processed_articles = await self.llm_processor.process_all_sites_parallel(
                articles_by_source,
                user.topics
            )

            if not processed_articles:
                return BriefingResult(
                    user_email=user.email,
                    articles_fetched=total_articles,
                    articles_processed=0,
                    top_articles=0,
                    email_sent=False,
                    status="error",
                    error="No articles could be processed"
                )

            # Group processed articles by source for landscape
            processed_by_source: Dict[str, List[Dict]] = {}
            for article in processed_articles:
                source = article.get("source", "Unknown")
                if source not in processed_by_source:
                    processed_by_source[source] = []
                processed_by_source[source].append(article)

            # Step 2: Generate Landscape (runs in parallel with top 5 selection)
            # Step 3: Select Top 5
            landscape_task = self.llm_processor.generate_landscape(
                processed_by_source,
                user.topics,
                total_articles
            )
            top5_task = self.llm_processor.select_top_5(
                processed_articles,
                user.topics
            )

            landscape_text, top_5 = await asyncio.gather(landscape_task, top5_task)

            if not landscape_text:
                landscape_text = "Unable to generate landscape summary."

            # Step 4: Generate Deep Dives (based on all processed articles)
            deep_dives = await self.llm_processor.generate_deep_dives(
                processed_articles,
                user.topics
            )

            # Build the Briefing object
            briefing = Briefing(
                landscape=Landscape(content=landscape_text),
                top_5=[
                    ProcessedArticle(
                        source=a.get("source", ""),
                        url=a.get("url", ""),
                        title=a.get("title", ""),
                        summary=a.get("summary", ""),
                        relevance=a.get("relevance", 0),
                        keywords=a.get("keywords", []),
                        why_selected=a.get("why_selected", ""),
                        rank=a.get("rank", i + 1)
                    )
                    for i, a in enumerate(top_5[:5])
                ],
                deep_dives=[
                    DeepDive(
                        topic=d.get("topic", ""),
                        hook=d.get("hook", ""),
                        analysis=d.get("analysis", ""),
                        related_articles=d.get("related_articles", [])
                    )
                    for d in deep_dives[:3]
                ],
                articles_analyzed=len(processed_articles),
                sources_count=len(articles_by_source)
            )

            # Step 5: Compose email
            briefing_date = datetime.utcnow().strftime("%B %d, %Y")
            html_body = self.email_sender.compose_briefing_email(
                self.config.template_path,
                user,
                briefing,
                briefing_date
            )

            # Step 6: Send email
            subject = f"Your AI Briefing for {briefing_date}"
            email_sent = self.email_sender.send_email(user.email, subject, html_body)

            return BriefingResult(
                user_email=user.email,
                articles_fetched=total_articles,
                articles_processed=len(processed_articles),
                top_articles=len(briefing.top_5),
                email_sent=email_sent,
                status="success" if email_sent else "email_failed"
            )

        except Exception as e:
            logger.error(f"Error generating briefing for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            return BriefingResult(
                user_email=user.email,
                articles_fetched=total_articles,
                articles_processed=0,
                top_articles=0,
                email_sent=False,
                status="error",
                error=str(e)
            )

    async def run(self, target_time: Optional[str] = None) -> List[BriefingResult]:
        """
        Run the complete briefing workflow.

        Args:
            target_time: Optional HH:MM filter. If None, process all users.

        Returns:
            List of BriefingResult for each processed user.
        """
        results = []

        # Step 1: Load user profiles
        profiles = self.profile_loader.load_profiles()

        if not profiles:
            logger.warning("No user profiles found")
            return results

        # Filter by time if specified
        if target_time:
            profiles = self.profile_loader.filter_by_time(profiles, target_time)
            logger.info(f"Filtered to {len(profiles)} users for time {target_time}")

        if not profiles:
            logger.info("No users to process for this time slot")
            return results

        # Step 2: Fetch articles
        since_date = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d")
        articles = await self.article_fetcher.fetch_articles(since_date=since_date)

        if not articles:
            logger.warning("No articles fetched from MCP")
            return results

        # Deduplicate and filter
        articles = self.article_fetcher.deduplicate(articles)
        articles = self.article_fetcher.filter_recent(articles, hours=48)
        total_articles = len(articles)

        # Step 3: Group by source for per-site agents
        articles_by_source = self.article_fetcher.group_by_source(articles)

        # Step 4: Generate briefings for each user
        for user in profiles:
            result = await self.generate_briefing_for_user(
                user,
                articles_by_source,
                total_articles
            )
            results.append(result)
            logger.info(f"Result for {user.email}: {result.status}")

        return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Node 2: Daily Briefing Generator")
    parser.add_argument(
        "--time",
        type=str,
        help="Filter users by briefing time (HH:MM format)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending emails"
    )
    args = parser.parse_args()

    # Create generator
    generator = BriefingGenerator(config)

    # Override email sending for dry run
    if args.dry_run:
        logger.info("DRY RUN MODE - emails will not be sent")
        generator.email_sender.send_email = lambda *args, **kwargs: True

    # Run workflow
    results = await generator.run(target_time=args.time)

    # Print summary
    print("\n" + "=" * 60)
    print("BRIEFING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total users processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.status == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.status != 'success')}")

    for result in results:
        status_icon = "✓" if result.status == "success" else "✗"
        print(f"  {status_icon} {result.user_email}: {result.status}")
        if result.error:
            print(f"      Error: {result.error}")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
