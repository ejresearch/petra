"""
Node 2: Daily Briefing Generator

Complete workflow implementation:
1. Reads user profiles from storage (Google Sheets/JSONL)
2. Fetches AI newsletter articles via MCP server
3. Summarizes & classifies articles using LLM
4. Selects top 5 & clusters them
5. Sends personalized email briefing

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
from dataclasses import dataclass, asdict
from jinja2 import Template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    # MCP Server
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8002")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Email (Gmail)
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")  # App password for Gmail
    from_email: str = os.getenv("FROM_EMAIL", "")

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
    """Article with LLM-generated metadata."""
    source: str
    url: str
    title: str
    published: str
    text: str
    summary: str
    keywords: List[str]
    why_it_matters: str
    matched_topics: List[str]
    relevance_score: float
    final_score: float
    rank: int = 0


@dataclass
class Cluster:
    """Article cluster from LLM."""
    cluster_id: str
    cluster_name: str
    indices: List[int]
    summary: str


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


# ============================================================================
# LLM PROCESSOR
# ============================================================================

class LLMProcessor:
    """Process articles using OpenAI LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[Dict]:
        """Make a single LLM API call."""
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens
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

                    # Parse JSON from response
                    # Handle potential markdown code blocks
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

    async def summarize_article(self, article: Article) -> Optional[Dict]:
        """Summarize a single article."""
        prompt = f"""Summarize this article in 150-200 words. Extract 5 keywords. Provide a one-sentence "Why this matters".

Article Title: {article.title}
Article Text: {article.text}

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "summary": "...",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "why_it_matters": "..."
}}"""

        return await self._call_llm(prompt)

    async def classify_relevance(self, article: Article, summary: str, user_topics: List[str]) -> Optional[Dict]:
        """Classify article relevance to user topics."""
        topics_str = ", ".join(user_topics)

        prompt = f"""Match this article to the user's topics using semantic similarity.

User Topics: {topics_str}
Article Title: {article.title}
Article Summary: {summary}

Return ONLY valid JSON:
{{
  "matched_topics": ["topic1", "topic2"],
  "relevance_score": 0.85
}}

Rules:
- matched_topics should only include topics from the user's list that are relevant
- relevance_score should be between 0.0 and 1.0
- Consider semantic similarity, not just keyword matching"""

        return await self._call_llm(prompt)

    async def cluster_articles(self, articles: List[ProcessedArticle], num_clusters: int = 3) -> List[Cluster]:
        """Cluster articles by theme."""
        articles_text = "\n".join([
            f"- [{i}] {a.title}: {a.summary}"
            for i, a in enumerate(articles)
        ])

        prompt = f"""Group these article summaries into exactly {num_clusters} clusters. Each cluster should represent a distinct theme or trend.

Articles:
{articles_text}

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_01",
      "cluster_name": "Theme Name",
      "indices": [0, 2],
      "summary": "Brief description of what unites these articles..."
    }}
  ]
}}

Rules:
- Create exactly {num_clusters} clusters
- Each article index should appear in exactly one cluster
- cluster_name should be 2-4 words
- summary should be 1-2 sentences"""

        result = await self._call_llm(prompt, max_tokens=800)

        if result and "clusters" in result:
            return [Cluster(**c) for c in result["clusters"]]
        return []

    async def process_articles_parallel(
        self,
        articles: List[Article],
        user_topics: List[str]
    ) -> List[ProcessedArticle]:
        """Process all articles in parallel."""
        processed = []

        # Summarize all articles in parallel
        logger.info(f"Summarizing {len(articles)} articles...")
        summary_tasks = [self.summarize_article(a) for a in articles]
        summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)

        # Classify relevance in parallel
        logger.info("Classifying relevance...")
        classify_tasks = []
        for article, summary_result in zip(articles, summaries):
            if isinstance(summary_result, dict):
                classify_tasks.append(
                    self.classify_relevance(article, summary_result.get("summary", ""), user_topics)
                )
            else:
                classify_tasks.append(asyncio.coroutine(lambda: None)())

        classifications = await asyncio.gather(*classify_tasks, return_exceptions=True)

        # Combine results
        for i, (article, summary_result, class_result) in enumerate(zip(articles, summaries, classifications)):
            if isinstance(summary_result, Exception) or not summary_result:
                continue
            if isinstance(class_result, Exception) or not class_result:
                continue

            # Calculate final score
            base_score = class_result.get("relevance_score", 0.5)
            topic_boost = min(len(class_result.get("matched_topics", [])) * 0.05, 0.15)
            final_score = min(base_score + topic_boost, 1.0)

            processed.append(ProcessedArticle(
                source=article.source,
                url=article.url,
                title=article.title,
                published=article.published,
                text=article.text,
                summary=summary_result.get("summary", ""),
                keywords=summary_result.get("keywords", []),
                why_it_matters=summary_result.get("why_it_matters", ""),
                matched_topics=class_result.get("matched_topics", []),
                relevance_score=base_score,
                final_score=final_score
            ))

        logger.info(f"Processed {len(processed)} articles successfully")
        return processed


# ============================================================================
# EMAIL SENDER
# ============================================================================

class EmailSender:
    """Send briefing emails via SMTP."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email

    def _load_template(self, template_path: str) -> Template:
        """Load Jinja2 email template."""
        with open(template_path, 'r') as f:
            return Template(f.read())

    def compose_email(
        self,
        template_path: str,
        user: UserProfile,
        articles: List[ProcessedArticle],
        clusters: List[Cluster],
        briefing_date: str,
        articles_analyzed: int
    ) -> str:
        """Compose HTML email from template."""
        template = self._load_template(template_path)

        # Prepare article data
        articles_data = [
            {
                "rank": a.rank,
                "source": a.source,
                "title": a.title,
                "url": a.url,
                "summary": a.summary,
                "why_it_matters": a.why_it_matters,
                "keywords": a.keywords,
                "score": a.final_score
            }
            for a in articles
        ]

        # Prepare cluster data
        clusters_data = [
            {
                "cluster_name": c.cluster_name,
                "summary": c.summary
            }
            for c in clusters
        ]

        return template.render(
            user_name=user.name or "there",
            briefing_date=briefing_date,
            articles_analyzed=articles_analyzed,
            article_count=len(articles),
            top_5_articles=articles_data,
            clusters=clusters_data,
            user_topics=user.topics,
            preferences_url="#",  # TODO: Add real URL
            unsubscribe_url="#"   # TODO: Add real URL
        )

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str
    ) -> bool:
        """Send email via SMTP."""
        if not all([self.username, self.password, self.from_email]):
            logger.warning("Email not configured, skipping send")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email

            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, to_email, msg.as_string())

            logger.info(f"Email sent to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class BriefingGenerator:
    """Main workflow orchestrator."""

    def __init__(self, cfg: Config):
        self.config = cfg
        self.profile_loader = ProfileLoader(cfg.profiles_path)
        self.article_fetcher = ArticleFetcher(cfg.mcp_server_url)
        self.llm_processor = LLMProcessor(cfg.openai_api_key, cfg.openai_model, cfg.openai_base_url)
        self.email_sender = EmailSender(
            cfg.smtp_server, cfg.smtp_port,
            cfg.smtp_username, cfg.smtp_password, cfg.from_email
        )

    async def generate_briefing_for_user(
        self,
        user: UserProfile,
        articles: List[Article]
    ) -> BriefingResult:
        """Generate and send briefing for a single user."""
        logger.info(f"Generating briefing for {user.email}")

        try:
            # Step 1: Process articles with LLM
            processed = await self.llm_processor.process_articles_parallel(
                articles[:self.config.max_articles],
                user.topics
            )

            if not processed:
                return BriefingResult(
                    user_email=user.email,
                    articles_fetched=len(articles),
                    articles_processed=0,
                    top_articles=0,
                    email_sent=False,
                    status="error",
                    error="No articles could be processed"
                )

            # Step 2: Sort by score and select top N
            processed.sort(key=lambda x: x.final_score, reverse=True)
            top_articles = processed[:self.config.top_n_articles]

            # Assign ranks
            for i, article in enumerate(top_articles):
                article.rank = i + 1

            # Step 3: Cluster articles
            clusters = await self.llm_processor.cluster_articles(
                top_articles,
                self.config.num_clusters
            )

            # Step 4: Compose email
            briefing_date = datetime.utcnow().strftime("%B %d, %Y")
            html_body = self.email_sender.compose_email(
                self.config.template_path,
                user,
                top_articles,
                clusters,
                briefing_date,
                len(articles)
            )

            # Step 5: Send email
            subject = f"Your Personalized AI Briefing for {briefing_date}"
            email_sent = self.email_sender.send_email(user.email, subject, html_body)

            return BriefingResult(
                user_email=user.email,
                articles_fetched=len(articles),
                articles_processed=len(processed),
                top_articles=len(top_articles),
                email_sent=email_sent,
                status="success" if email_sent else "email_failed"
            )

        except Exception as e:
            logger.error(f"Error generating briefing for {user.email}: {e}")
            return BriefingResult(
                user_email=user.email,
                articles_fetched=len(articles),
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
        articles = self.article_fetcher.filter_recent(articles, hours=24)

        # Step 3: Generate briefings for each user
        for user in profiles:
            result = await self.generate_briefing_for_user(user, articles)
            results.append(result)

            # Log result
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
