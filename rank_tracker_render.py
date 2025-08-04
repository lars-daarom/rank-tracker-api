#!/usr/bin/env python3
"""
Quick fix version - Minimal Rank Tracker
"""

import os
import time
import json
import random
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Database imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

import sqlite3

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DATABASE_URL = os.environ.get('DATABASE_URL')
USE_POSTGRES = DATABASE_URL is not None and POSTGRES_AVAILABLE
RENDER_ENV = os.environ.get('RENDER') is not None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Models
class RankRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=255)
    domain: str = Field(..., min_length=1, max_length=255)
    country: str = Field(default="nl", pattern="^(nl|be|de|uk|us)$")

class RankResponse(BaseModel):
    position: Optional[int] = None
    url: Optional[str] = None
    search_results_count: Optional[int] = None
    timestamp: str
    country: str
    processing_time: float

# Simple Database Functions
def get_db_connection():
    """Simple database connection"""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        conn = sqlite3.connect('rank_tracker.db', timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

def save_rank_simple(keyword: str, domain: str, position: Optional[int], url: Optional[str], country: str):
    """Simple save function"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if USE_POSTGRES:
            cursor.execute("""
                INSERT INTO rank_history (keyword, domain, position, url, country, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (keyword, domain, timestamp) DO NOTHING
            """, (keyword, domain, position, url, country, datetime.now()))
        else:
            cursor.execute("""
                INSERT OR IGNORE INTO rank_history 
                (keyword, domain, position, url, country, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (keyword, domain, position, url, country, datetime.now().isoformat()))
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"💾 Saved: {keyword} -> {position or 'Not found'}")
        
    except Exception as e:
        logger.error(f"Save failed: {e}")

# Google Scraper
class SimpleGoogleScraper:
    def __init__(self):
        self.session = requests.Session()
        self.last_request = 0
        self.delay = 5  # 5 seconds between requests
        
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def get_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
    
    def build_url(self, keyword: str, country: str = 'nl'):
        """Build simple Google URL"""
        url = f"https://www.google.com/search?q={quote_plus(keyword)}"
        
        if country != 'us':
            country_lang = {'nl': 'nl', 'be': 'nl', 'de': 'de', 'uk': 'en'}
            if country in country_lang:
                url += f"&hl={country_lang[country]}"
        
        logger.info(f"🌐 URL: {url}")
        return url
    
    async def rate_limit(self):
        """Wait between requests"""
        now = time.time()
        elapsed = now - self.last_request
        
        if elapsed < self.delay:
            wait_time = self.delay - elapsed + random.uniform(1, 3)
            logger.info(f"⏳ Waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request = time.time()
    
    def extract_results(self, html: str) -> List[Dict[str, Any]]:
        """Extract search results with debugging"""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        position = 1
        
        # Debug: Let's see what elements exist
        all_divs = soup.select('div')
        g_divs = soup.select('div.g')
        h3_elements = soup.select('h3')
        all_links = soup.select('a[href]')
        http_links = soup.select('a[href^="http"]')
        
        logger.info(f"🔍 Debug counts - Total divs: {len(all_divs)}, div.g: {len(g_divs)}, h3: {len(h3_elements)}, all links: {len(all_links)}, http links: {len(http_links)}")
        
        # Sample some elements to understand structure
        if h3_elements:
            logger.info(f"📝 First h3 text: {h3_elements[0].get_text()[:50]}...")
        
        if all_links:
            sample_hrefs = [link.get('href', '')[:50] for link in all_links[:5]]
            logger.info(f"🔗 Sample hrefs: {sample_hrefs}")
        
        # Try various extraction strategies
        strategies = [
            # Strategy 1: Look for h3 elements and find their parent links
            ('h3_parent_links', lambda: self.extract_h3_links(soup)),
            
            # Strategy 2: Find all external links and filter
            ('external_links', lambda: self.extract_external_links(soup)),
            
            # Strategy 3: Look for common Google result patterns
            ('result_patterns', lambda: self.extract_result_patterns(soup)),
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                strategy_results = strategy_func()
                logger.info(f"🎯 Strategy '{strategy_name}' found {len(strategy_results)} results")
                
                if strategy_results:
                    results = strategy_results
                    break
                    
            except Exception as e:
                logger.error(f"Strategy '{strategy_name}' failed: {e}")
                continue
        
        logger.info(f"✅ Final extraction: {len(results)} results")
        return results
    
    def extract_h3_links(self, soup) -> List[Dict[str, Any]]:
        """Extract by finding h3 elements and their parent links"""
        results = []
        position = 1
        
        h3_elements = soup.select('h3')
        
        for h3 in h3_elements:
            # Look for a link that contains or is near this h3
            link = None
            
            # Check if h3 is inside a link
            link = h3.find_parent('a')
            
            # If not, look for a link near the h3
            if not link:
                # Look in parent container
                parent = h3.find_parent()
                if parent:
                    link = parent.select_one('a[href^="http"]')
            
            # Check siblings
            if not link and h3.parent:
                link = h3.parent.select_one('a[href^="http"]')
            
            if link:
                href = link.get('href', '')
                
                # Skip Google's own links
                if any(skip in href.lower() for skip in [
                    'google.', 'youtube.com/results', '/aclk?', 'googleadservices', 'accounts.google'
                ]):
                    continue
                
                title = h3.get_text().strip()
                if len(title) < 5:
                    continue
                
                try:
                    domain = urlparse(href).netloc.replace('www.', '').lower()
                    if domain:
                        results.append({
                            'position': position,
                            'url': href,
                            'title': title,
                            'domain': domain
                        })
                        position += 1
                        
                        if len(results) >= 20:  # Max 20 results
                            break
                except:
                    continue
        
        return results
    
    def extract_external_links(self, soup) -> List[Dict[str, Any]]:
        """Extract by finding all external links and filtering"""
        results = []
        position = 1
        
        links = soup.select('a[href^="http"]')
        
        for link in links:
            href = link.get('href', '')
            
            # Skip unwanted links
            if any(skip in href.lower() for skip in [
                'google.', 'youtube.com/results', '/aclk?', 'googleadservices', 
                'accounts.google', 'maps.google', 'translate.google'
            ]):
                continue
            
            # Get title from link text or nearby h3
            title = link.get_text().strip()
            
            if not title or len(title) < 5:
                # Look for h3 in parent or children
                parent = link.find_parent()
                if parent:
                    h3 = parent.select_one('h3')
                    if h3:
                        title = h3.get_text().strip()
            
            if not title or len(title) < 5:
                continue
            
            try:
                domain = urlparse(href).netloc.replace('www.', '').lower()
                if domain:
                    results.append({
                        'position': position,
                        'url': href,
                        'title': title,
                        'domain': domain
                    })
                    position += 1
                    
                    if len(results) >= 20:
                        break
            except:
                continue
        
        return results
    
    def extract_result_patterns(self, soup) -> List[Dict[str, Any]]:
        """Extract using common Google result patterns"""
        results = []
        position = 1
        
        # Look for containers that might hold results
        containers = soup.select('div[data-hveid], div[data-async-context], div.g, div[jscontroller]')
        
        for container in containers:
            # Look for external links in this container
            links = container.select('a[href^="http"]')
            
            for link in links:
                href = link.get('href', '')
                
                # Skip unwanted
                if any(skip in href.lower() for skip in [
                    'google.', 'youtube.com/results', '/aclk?', 'googleadservices'
                ]):
                    continue
                
                # Must have a reasonable title
                title = link.get_text().strip()
                
                # Look for h3 near this link
                if not title or len(title) < 10:
                    h3 = container.select_one('h3')
                    if h3:
                        title = h3.get_text().strip()
                
                if title and len(title) >= 10:
                    try:
                        domain = urlparse(href).netloc.replace('www.', '').lower()
                        if domain:
                            results.append({
                                'position': position,
                                'url': href,
                                'title': title,
                                'domain': domain
                            })
                            position += 1
                            
                            if len(results) >= 20:
                                return results
                    except:
                        continue
        
        return results
    
    async def search(self, keyword: str, country: str = 'nl') -> List[Dict[str, Any]]:
        """Perform Google search"""
        await self.rate_limit()
        
        url = self.build_url(keyword, country)
        headers = self.get_headers()
        
        try:
            logger.info(f"🔍 Searching: '{keyword}' in {country}")
            
            response = self.session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            # Check encoding
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            html = response.text
            logger.info(f"📡 Status: {response.status_code}, Length: {len(html)}")
            
            # Check for blocking
            if any(block in html.lower() for block in ['unusual traffic', 'captcha', 'blocked']):
                logger.warning("🚫 Google blocking detected")
                return []
                
            # Basic HTML check
            if not ('<!DOCTYPE html' in html or '<html' in html):
                logger.warning("⚠️ Invalid HTML response")
                return []
                
            results = self.extract_results(html)
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def find_domain_position(self, results: List[Dict[str, Any]], target_domain: str) -> tuple[Optional[int], Optional[str]]:
        """Find domain in results"""
        target = target_domain.replace('www.', '').lower()
        
        for result in results:
            if target in result['domain'] or result['domain'] in target:
                logger.info(f"✅ Found {target} at position {result['position']}")
                return result['position'], result['url']
        
        logger.info(f"❌ {target} not found in results")
        return None, None

# Global scraper
scraper = SimpleGoogleScraper()

# FastAPI App
app = FastAPI(title="🔍 Rank Tracker API", version="2.0.1")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Rank Tracker API</title></head>
    <body style="font-family: Arial; margin: 40px; text-align: center;">
        <h1>🔍 Rank Tracker API</h1>
        <p>Professional SEO rank tracking voor daar-om.nl</p>
        <p><strong>Status:</strong> <span style="color: green;">OPERATIONAL</span></p>
        <p><strong>Database:</strong> """ + ("PostgreSQL" if USE_POSTGRES else "SQLite") + """</p>
        <hr>
        <p><a href="/docs">📚 API Documentation</a></p>
        <p><a href="/health">❤️ Health Check</a></p>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    """Health check"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_type": "PostgreSQL" if USE_POSTGRES else "SQLite",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {e}")

@app.post("/check-rank", response_model=RankResponse)
@limiter.limit("30/minute")
async def check_rank(request: Request, rank_request: RankRequest):
    """Check rank for keyword"""
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Request: '{rank_request.keyword}' for {rank_request.domain}")
        
        # Search Google
        results = await scraper.search(rank_request.keyword, rank_request.country)
        
        # Find domain
        position, url = scraper.find_domain_position(results, rank_request.domain)
        
        processing_time = time.time() - start_time
        
        # Save to database
        save_rank_simple(
            rank_request.keyword,
            rank_request.domain,
            position,
            url,
            rank_request.country
        )
        
        response = RankResponse(
            position=position,
            url=url,
            search_results_count=len(results),
            timestamp=datetime.now().isoformat(),
            country=rank_request.country,
            processing_time=round(processing_time, 3)
        )
        
        status = "✅" if position else "❌"
        logger.info(f"{status} Completed: {rank_request.keyword} -> {position or 'Not found'}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/keywords")
@limiter.limit("60/minute")
async def get_keywords(request: Request):
    """Get tracked keywords"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT keyword, domain, MAX(timestamp) as last_check,
                   (SELECT position FROM rank_history rh2 
                    WHERE rh2.keyword = rh1.keyword AND rh2.domain = rh1.domain 
                    ORDER BY timestamp DESC LIMIT 1) as latest_position
            FROM rank_history rh1
            GROUP BY keyword, domain
            ORDER BY last_check DESC
            LIMIT 50
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        keywords = []
        for row in rows:
            keywords.append({
                'keyword': row['keyword'] if 'keyword' in row.keys() else row[0],
                'domain': row['domain'] if 'domain' in row.keys() else row[1],
                'last_check': row['last_check'] if 'last_check' in row.keys() else row[2],
                'latest_position': row['latest_position'] if 'latest_position' in row.keys() else row[3]
            })
        
        return keywords
        
    except Exception as e:
        logger.error(f"Keywords error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    
    print("🚀 Starting Minimal Rank Tracker API")
    print(f"📊 Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print(f"🌍 Port: {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
