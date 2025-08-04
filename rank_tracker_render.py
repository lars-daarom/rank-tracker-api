#!/usr/bin/env python3
"""
Professional Rank Tracker API - Optimized voor Render.com
SEO rank tracking tool voor daar-om.nl

Features:
- Multi-country Google scraping (NL, BE, DE, UK, US)
- PostgreSQL/SQLite dual database support
- Rate limiting en anti-blocking measures
- Comprehensive analytics en logging
- iOS SwiftUI app compatible
- Production ready met error handling
"""

import os
import time
import json
import random
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rank_tracker.log') if not os.environ.get('RENDER') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
USE_POSTGRES = DATABASE_URL is not None and POSTGRES_AVAILABLE
API_KEY = os.environ.get('API_KEY')  # Optional API key protection
RENDER_ENV = os.environ.get('RENDER') is not None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security (optioneel)
security = HTTPBearer(auto_error=False) if API_KEY else None

# Pydantic Models
class RankRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=255, description="Zoekwoord om te checken")
    domain: str = Field(..., min_length=1, max_length=255, description="Domein om te zoeken")
    country: str = Field(default="nl", pattern="^(nl|be|de|uk|us)$", description="Land code")

class RankResponse(BaseModel):
    position: Optional[int] = Field(None, description="Positie in zoekresultaten (null = niet gevonden)")
    url: Optional[str] = Field(None, description="URL van gevonden pagina")
    search_results_count: Optional[int] = Field(None, description="Totaal aantal zoekresultaten")
    timestamp: str = Field(..., description="Timestamp van de check")
    country: str = Field(..., description="Land waarin gezocht is")
    processing_time: float = Field(..., description="Verwerkingstijd in seconden")

class RankHistory(BaseModel):
    keyword: str
    domain: str
    position: Optional[int]
    url: Optional[str]
    country: str
    timestamp: str

class AnalyticsResponse(BaseModel):
    total_checks: int
    unique_keywords: int
    unique_domains: int
    avg_response_time: float
    success_rate: float
    top_keywords: List[Dict[str, Any]]
    country_distribution: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

class KeywordSummary(BaseModel):
    keyword: str
    domain: str
    latest_position: Optional[int]
    previous_position: Optional[int]
    trend: str  # 'up', 'down', 'stable', 'new'
    country: str
    last_check: str
    total_checks: int

# Database Connection Manager
class DatabaseManager:
    def __init__(self):
        self.use_postgres = USE_POSTGRES
        
    def get_connection(self):
        """Get database connection"""
        if self.use_postgres:
            return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        else:
            conn = sqlite3.connect('rank_tracker.db', timeout=30.0)
            conn.row_factory = sqlite3.Row
            return conn
    
    def execute_query(self, query: str, params: tuple = (), fetch: str = None):
        """Execute query met error handling"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            
            if fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'all':
                result = cursor.fetchall()
            else:
                result = None
                
            conn.commit()
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise HTTPException(status_code=500, detail="Database error")

# Global database manager
db = DatabaseManager()

# Google Scraping Class
class GoogleRankTracker:
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_delay = 4 if RENDER_ENV else 2  # Conservative voor production
        
        # Uitgebreide user agents pool
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        # Google domains en languages
        self.google_config = {
            'nl': {'domain': 'google.nl', 'lang': 'nl', 'gl': 'NL'},
            'be': {'domain': 'google.be', 'lang': 'nl', 'gl': 'BE'},
            'de': {'domain': 'google.de', 'lang': 'de', 'gl': 'DE'},
            'uk': {'domain': 'google.co.uk', 'lang': 'en', 'gl': 'GB'},
            'us': {'domain': 'google.com', 'lang': 'en', 'gl': 'US'}
        }
    
    def get_random_headers(self) -> Dict[str, str]:
        """Generate random headers om detectie te voorkomen"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Sec-GPC': '1'
        }
    
    def clean_domain(self, domain: str) -> str:
        """Clean en normalize domain"""
        if domain.startswith(('http://', 'https://')):
            domain = urlparse(domain).netloc
        return domain.replace('www.', '').lower().strip()
    
    def build_google_url(self, keyword: str, country: str = 'nl', num_results: int = 100) -> str:
        """Build Google search URL met country-specific settings"""
        config = self.google_config.get(country, self.google_config['nl'])
        
        base_url = f"https://www.{config['domain']}/search"
        params = {
            'q': keyword,
            'num': num_results,
            'hl': config['lang'],
            'gl': config['gl'],
            'start': 0,
            'pws': '0',  # Disable personalization
            'safe': 'off',
            'filter': '0'  # No duplicate filtering
        }
        
        query_string = '&'.join([f"{k}={quote_plus(str(v))}" for k, v in params.items()])
        return f"{base_url}?{query_string}"
    
    async def rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        if time_diff < self.request_delay:
            sleep_time = self.request_delay - time_diff + random.uniform(0.5, 1.5)  # Add jitter
            logger.info(f"⏳ Rate limiting: waiting {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def extract_search_results(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract search results van Google HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Check if we got blocked first
        if any(indicator in html_content.lower() for indicator in [
            'unusual traffic', 'captcha', 'blocked'
        ]):
            logger.warning("🚫 Google blocking detected in HTML content")
            return results
        
        # Comprehensive selectors for different Google layouts (2024/2025)
        selectors = [
            # Modern layouts
            'div.g div.yuRUbf',           # Current primary
            'div.g:has(div.yuRUbf)',      # Alternative modern
            'div[data-hveid] div.yuRUbf', # Data attribute version
            
            # Classic layouts  
            '.tF2Cxc',                    # Classic container
            'div.g .rc',                  # Older classic
            
            # Fallback broader selectors
            'div.g:has(h3)',              # Any div.g with h3
            'div[data-hveid]:has(h3)',    # Data divs with h3
            'div.g',                      # Broadest fallback
            
            # Link-based selectors
            'div:has(> a[href^="http"]):has(h3)', # Divs with external links and h3
        ]
        
        elements = []
        used_selector = None
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                # Filter out empty or ad results
                valid_elements = []
                for elem in elements:
                    # Must have a proper link
                    link = elem.select_one('a[href]')
                    if link and link.get('href'):
                        href = link.get('href')
                        # Skip ads and Google's own links
                        if not any(skip in href.lower() for skip in [
                            '/aclk?', 'googleadservices', '/search?', 'accounts.google'
                        ]):
                            valid_elements.append(elem)
                
                if valid_elements and len(valid_elements) >= 3:
                    elements = valid_elements
                    used_selector = selector
                    logger.info(f"🎯 Using selector '{selector}' - found {len(elements)} valid elements")
                    break
                    
            except Exception as e:
                logger.warning(f"Selector '{selector}' failed: {e}")
                continue
        
        if not elements:
            logger.warning("❌ No search results found with any selector")
            # Log more detailed HTML sample for debugging
            sample_html = html_content[:1000] if len(html_content) > 1000 else html_content
            logger.warning(f"HTML sample (first 1000 chars): {sample_html}")
            
            # Try to find any links at all for debugging
            all_links = soup.select('a[href]')
            logger.info(f"🔍 Found {len(all_links)} total links in page")
            
            # Look for common Google elements
            google_divs = soup.select('div.g')
            logger.info(f"🔍 Found {len(google_divs)} div.g elements")
            
            return results
        
        position = 1
        for element in elements[:100]:  # Max 100 results
            try:
                # Strategy 1: yuRUbf container (modern Google)
                link_container = element.select_one('div.yuRUbf a') or element.select_one('.yuRUbf a')
                
                # Strategy 2: Direct link in element
                if not link_container:
                    link_container = element.select_one('a[href]')
                
                # Strategy 3: First valid link in element
                if not link_container:
                    all_links = element.select('a[href]')
                    for link in all_links:
                        href = link.get('href', '')
                        if href.startswith('http') and 'google.' not in href:
                            link_container = link
                            break
                
                if not link_container:
                    continue
                
                url = link_container.get('href', '')
                
                # Clean and validate URL
                if url.startswith('/url?q='):
                    try:
                        from urllib.parse import unquote, parse_qs, urlparse as parse_url
                        parsed = parse_url(url)
                        if parsed.query:
                            query_params = parse_qs(parsed.query)
                            if 'q' in query_params:
                                url = unquote(query_params['q'][0])
                    except:
                        continue
                
                # Skip invalid URLs
                if not url.startswith('http') or any(skip in url.lower() for skip in [
                    'google.', 'youtube.com/results', 'maps.google', '/aclk?', 
                    'googleadservices', 'accounts.google', '/search'
                ]):
                    continue
                
                # Extract title - multiple strategies
                title = None
                
                # Strategy 1: h3 tag (most common)
                title_elem = element.select_one('h3')
                if title_elem:
                    title = title_elem.get_text().strip()
                
                # Strategy 2: Link text if no h3
                if not title and link_container:
                    title = link_container.get_text().strip()
                
                # Strategy 3: aria-label or other attributes
                if not title:
                    for attr in ['aria-label', 'title']:
                        if link_container.get(attr):
                            title = link_container.get(attr).strip()
                            break
                
                if not title:
                    title = "No title found"
                
                # Extract snippet/description
                snippet = ""
                snippet_selectors = [
                    '.VwiC3b',      # Modern snippet
                    '.s',           # Classic snippet  
                    '.st',          # Alternative snippet
                    'span:contains("...")',  # Ellipsis indicator
                    '.IsZvec'       # Another modern class
                ]
                
                for snippet_sel in snippet_selectors:
                    snippet_elem = element.select_one(snippet_sel)
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()[:200]
                        break
                
                # Extract domain
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.replace('www.', '').lower()
                except:
                    continue
                
                if domain and title != "No title found":  # Only add quality results
                    results.append({
                        'position': position,
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'domain': domain
                    })
                    position += 1
                
            except Exception as e:
                logger.error(f"Error extracting result at position {position}: {e}")
                continue
        
        logger.info(f"✅ Successfully extracted {len(results)} valid search results")
        return results
    
    async def search_google(self, keyword: str, country: str = 'nl') -> List[Dict[str, Any]]:
        """Perform Google search met anti-blocking measures"""
        await self.rate_limit()
        
        url = self.build_google_url(keyword, country)
        headers = self.get_random_headers()
        
        logger.info(f"🔍 Searching Google: '{keyword}' in {country.upper()}")
        logger.info(f"🌐 URL: {url[:100]}...")
        
        try:
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=15,
                allow_redirects=True
            )
            response.raise_for_status()
            
            logger.info(f"📡 Response status: {response.status_code}")
            logger.info(f"📏 Response length: {len(response.text)} characters")
            
            # Check for blocking indicators
            content_lower = response.text.lower()
            blocking_indicators = ['unusual traffic', 'automated queries', 'captcha', 'blocked', 'our systems have detected']
            found_indicators = [indicator for indicator in blocking_indicators if indicator in content_lower]
            
            if found_indicators:
                logger.warning(f"⚠️ Google blocking indicators found: {found_indicators}")
                # Don't raise exception yet, try to parse anyway
            
            # Log some sample content for debugging
            sample_content = response.text[:2000] if len(response.text) > 2000 else response.text
            logger.info(f"📄 HTML sample (first 2000 chars): {sample_content}")
            
            # Check if we got actual search results
            has_search_content = any(indicator in content_lower for indicator in [
                'search results', 'about', 'results', 'web', 'images'
            ])
            
            if not has_search_content:
                logger.warning(f"⚠️ Response doesn't look like search results page")
                
            # Try to find common Google search page elements
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for search form (indicates we're on a search page)
            search_form = soup.select_one('form[action="/search"]')
            logger.info(f"🔍 Search form found: {search_form is not None}")
            
            # Look for result stats
            result_stats = soup.select_one('#result-stats') or soup.select_one('.result-stats')
            if result_stats:
                logger.info(f"📊 Result stats found: {result_stats.get_text()[:100]}")
            
            # Count different element types
            div_g_count = len(soup.select('div.g'))
            yuRUbf_count = len(soup.select('.yuRUbf'))
            tF2Cxc_count = len(soup.select('.tF2Cxc'))
            all_links = len(soup.select('a[href]'))
            
            logger.info(f"🧮 Element counts - div.g: {div_g_count}, .yuRUbf: {yuRUbf_count}, .tF2Cxc: {tF2Cxc_count}, links: {all_links}")
            
            results = self.extract_search_results(response.text)
            
            if not results:
                logger.warning(f"🔍 Geen zoekresultaten gevonden voor '{keyword}' in {country}")
                
                # Additional debugging: save HTML to check manually
                if len(response.text) > 1000:
                    # In development, we could save this to a file
                    logger.info("💡 Consider saving response HTML for manual inspection")
            else:
                logger.info(f"✅ Successfully found {len(results)} results")
            
            return results
            
        except requests.exceptions.Timeout:
            logger.error("⏰ Google search timeout")
            raise HTTPException(status_code=408, detail="Search timeout - probeer opnieuw")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Network error during Google search: {e}")
            raise HTTPException(status_code=503, detail="Netwerkfout bij zoeken")
            
        except Exception as e:
            logger.error(f"❌ Unexpected error during Google search: {e}")
            raise HTTPException(status_code=500, detail="Onverwachte fout bij zoeken")
    
    def find_domain_position(self, results: List[Dict[str, Any]], target_domain: str) -> tuple[Optional[int], Optional[str]]:
        """Find domain position in search results"""
        target_domain = self.clean_domain(target_domain)
        
        logger.info(f"🎯 Looking for domain: {target_domain}")
        
        for result in results:
            result_domain = result.get('domain', '')
            
            # Exact match
            if target_domain == result_domain:
                logger.info(f"✅ Found exact match at position {result['position']}: {result['url']}")
                return result['position'], result['url']
            
            # Subdomain match (e.g., blog.daar-om.nl matches daar-om.nl)
            if target_domain in result_domain or result_domain in target_domain:
                logger.info(f"✅ Found subdomain match at position {result['position']}: {result['url']}")
                return result['position'], result['url']
        
        logger.info(f"❌ Domain '{target_domain}' not found in search results")
        return None, None

# Global tracker instance
tracker = GoogleRankTracker()

# Database Functions
def save_rank_to_db(keyword: str, domain: str, position: Optional[int], 
                   url: Optional[str], country: str, processing_time: float):
    """Save rank result naar database"""
    try:
        if USE_POSTGRES:
            query = """
                INSERT INTO rank_history (keyword, domain, position, url, country, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (keyword, domain, timestamp) DO NOTHING
            """
            params = (keyword, domain, position, url, country, datetime.now())
        else:
            query = """
                INSERT OR IGNORE INTO rank_history 
                (keyword, domain, position, url, country, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (keyword, domain, position, url, country, datetime.now().isoformat())
        
        db.execute_query(query, params)
        logger.info(f"💾 Saved rank: {keyword} -> {position or 'Not found'}")
        
    except Exception as e:
        logger.error(f"Failed to save rank to database: {e}")

def log_analytics(endpoint: str, keyword: str, domain: str, country: str, 
                 response_time: float, status_code: int):
    """Log analytics data"""
    try:
        if USE_POSTGRES:
            query = """
                INSERT INTO analytics (endpoint, keyword, domain, country, response_time, status_code, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (endpoint, keyword, domain, country, response_time, status_code, datetime.now())
        else:
            query = """
                INSERT INTO analytics (endpoint, keyword, domain, country, response_time, status_code, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (endpoint, keyword, domain, country, response_time, status_code, datetime.now().isoformat())
        
        db.execute_query(query, params)
        
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")

# API Key verification (optioneel)
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if configured"""
    if API_KEY and (not credentials or credentials.credentials != API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return credentials.credentials if credentials else None

# Application Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown"""
    logger.info("🚀 Starting Professional Rank Tracker API")
    logger.info(f"📊 Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    logger.info(f"🔒 API Key Protection: {'Enabled' if API_KEY else 'Disabled'}")
    logger.info(f"🌍 Environment: {'Production (Render)' if RENDER_ENV else 'Development'}")
    
    # Initialize database
    try:
        from setup_db import setup_database
        setup_database()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    logger.info("👋 Shutting down Rank Tracker API")

# FastAPI Application
app = FastAPI(
    title="🔍 Professional Rank Tracker API",
    description="Professional SEO rank tracking voor daar-om.nl CRO specialists",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not RENDER_ENV else [
        "https://daar-om.nl",
        "https://www.daar-om.nl", 
        "https://app.daar-om.nl"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )
    
    return response

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """API root met HTML dashboard"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rank Tracker API - daar-om.nl</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; border-radius: 10px; text-align: center; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                     gap: 20px; margin: 30px 0; }}
            .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .endpoints {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }}
            .endpoint {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Professional Rank Tracker API</h1>
            <p>SEO rank tracking voor daar-om.nl CRO specialists</p>
            <p>Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'} | 
               Environment: {'Production' if RENDER_ENV else 'Development'}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>📊 Status</h3>
                <p style="color: green; font-weight: bold;">OPERATIONAL</p>
            </div>
            <div class="stat-card">
                <h3>🌍 Supported Countries</h3>
                <p>NL, BE, DE, UK, US</p>
            </div>
            <div class="stat-card">
                <h3>⚡ Version</h3>
                <p>v2.0.0</p>
            </div>
        </div>
        
        <div class="endpoints">
            <h2>🔗 API Endpoints</h2>
            <div class="endpoint"><strong>POST /check-rank</strong> - Check keyword position</div>
            <div class="endpoint"><strong>GET /rank-history</strong> - Get historical data</div>
            <div class="endpoint"><strong>GET /keywords</strong> - List tracked keywords</div>
            <div class="endpoint"><strong>GET /analytics</strong> - Usage analytics</div>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>GET /docs</strong> - Interactive API documentation</div>
        </div>
        
        <p style="text-align: center; margin-top: 40px; color: #6c757d;">
            Gemaakt door <strong>daar-om.nl</strong> 🚀
        </p>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint voor monitoring"""
    try:
        # Test database connection
        if USE_POSTGRES:
            query = "SELECT 1 as test"
        else:
            query = "SELECT 1 as test"
            
        result = db.execute_query(query, fetch='one')
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_type": "PostgreSQL" if USE_POSTGRES else "SQLite",
            "environment": "production" if RENDER_ENV else "development",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/check-rank", response_model=RankResponse)
@limiter.limit("30/minute")
async def check_rank(
    request: Request, 
    rank_request: RankRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(verify_api_key) if API_KEY else None
):
    """Check domain position voor keyword"""
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Rank check request: '{rank_request.keyword}' for {rank_request.domain} in {rank_request.country}")
        
        # Search Google
        results = await tracker.search_google(rank_request.keyword, rank_request.country)
        
        # Find domain position
        position, url = tracker.find_domain_position(results, rank_request.domain)
        
        processing_time = time.time() - start_time
        
        # Save to database (background task voor performance)
        background_tasks.add_task(
            save_rank_to_db,
            rank_request.keyword,
            rank_request.domain,
            position,
            url,
            rank_request.country,
            processing_time
        )
        
        # Log analytics (background task)
        background_tasks.add_task(
            log_analytics,
            "check-rank",
            rank_request.keyword,
            rank_request.domain,
            rank_request.country,
            processing_time,
            200
        )
        
        response = RankResponse(
            position=position,
            url=url,
            search_results_count=len(results),
            timestamp=datetime.now().isoformat(),
            country=rank_request.country,
            processing_time=round(processing_time, 3)
        )
        
        status_emoji = "✅" if position else "❌"
        logger.info(f"{status_emoji} Rank check completed: {rank_request.keyword} -> {position or 'Not found'} ({processing_time:.2f}s)")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (zoals rate limiting)
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log failed analytics
        background_tasks.add_task(
            log_analytics,
            "check-rank",
            rank_request.keyword,
            rank_request.domain,
            rank_request.country,
            processing_time,
            500
        )
        
        logger.error(f"❌ Rank check failed for '{rank_request.keyword}': {e}")
        raise HTTPException(status_code=500, detail=f"Rank check failed: {str(e)}")

@app.get("/rank-history")
@limiter.limit("60/minute")
async def get_rank_history(
    request: Request,
    keyword: str,
    domain: str,
    limit: int = 50,
    days: int = 30,
    api_key: Optional[str] = Depends(verify_api_key) if API_KEY else None
):
    """Get rank history voor keyword/domain"""
    try:
        limit = min(limit, 100)  # Max 100 records
        days = min(days, 365)    # Max 1 year
        
        if USE_POSTGRES:
            query = """
                SELECT keyword, domain, position, url, country, timestamp
                FROM rank_history 
                WHERE keyword = %s AND domain = %s 
                AND timestamp > %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            since_date = datetime.now() - timedelta(days=days)
            params = (keyword, domain, since_date, limit)
        else:
            query = """
                SELECT keyword, domain, position, url, country, timestamp
                FROM rank_history 
                WHERE keyword = ? AND domain = ?
                AND datetime(timestamp) > datetime('now', '-{} days')
                ORDER BY timestamp DESC
                LIMIT ?
            """.format(days)
            params = (keyword, domain, limit)
        
        rows = db.execute_query(query, params, fetch='all')
        
        history = []
        for row in rows:
            history.append(RankHistory(
                keyword=row['keyword'],
                domain=row['domain'],
                position=row['position'],
                url=row['url'],
                country=row['country'],
                timestamp=str(row['timestamp'])
            ))
        
        logger.info(f"📊 Retrieved {len(history)} history records for {keyword} @ {domain}")
        return history
        
    except Exception as e:
        logger.error(f"Error getting rank history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rank history")

@app.get("/keywords")
@limiter.limit("30/minute") 
async def get_tracked_keywords(
    request: Request,
    limit: int = 50,
    api_key: Optional[str] = Depends(verify_api_key) if API_KEY else None
):
    """Get all tracked keywords met latest positions en trends"""
    try:
        if USE_POSTGRES:
            query = """
                WITH latest_ranks AS (
                    SELECT DISTINCT ON (keyword, domain) 
                           keyword, domain, position, country, timestamp,
                           ROW_NUMBER() OVER (PARTITION BY keyword, domain ORDER BY timestamp DESC) as rn
                    FROM rank_history 
                    ORDER BY keyword, domain, timestamp DESC
                ),
                previous_ranks AS (
                    SELECT keyword, domain, position as prev_position
                    FROM rank_history r1
                    WHERE timestamp = (
                        SELECT MAX(timestamp) 
                        FROM rank_history r2 
                        WHERE r2.keyword = r1.keyword 
                        AND r2.domain = r1.domain 
                        AND r2.timestamp < (
                            SELECT MAX(timestamp) 
                            FROM rank_history r3 
                            WHERE r3.keyword = r1.keyword 
                            AND r3.domain = r1.domain
                        )
                    )
                ),
                check_counts AS (
                    SELECT keyword, domain, COUNT(*) as total_checks
                    FROM rank_history
                    GROUP BY keyword, domain
                )
                SELECT 
                    l.keyword, l.domain, l.position as latest_position,
                    p.prev_position as previous_position, l.country, l.timestamp as last_check,
                    c.total_checks,
                    CASE 
                        WHEN p.prev_position IS NULL THEN 'new'
                        WHEN l.position IS NULL AND p.prev_position IS NOT NULL THEN 'down'
                        WHEN l.position IS NOT NULL AND p.prev_position IS NULL THEN 'up'
                        WHEN l.position < p.prev_position THEN 'up'
                        WHEN l.position > p.prev_position THEN 'down'
                        ELSE 'stable'
                    END as trend
                FROM latest_ranks l
                LEFT JOIN previous_ranks p ON l.keyword = p.keyword AND l.domain = p.domain
                LEFT JOIN check_counts c ON l.keyword = c.keyword AND l.domain = c.domain
                WHERE l.rn = 1
                ORDER BY l.timestamp DESC
                LIMIT %s
            """
            params = (limit,)
        else:
            # Simplified SQLite query
            query = """
                SELECT keyword, domain,
                       MAX(timestamp) as last_check,
                       (SELECT position FROM rank_history rh2 
                        WHERE rh2.keyword = rh1.keyword 
                        AND rh2.domain = rh1.domain 
                        ORDER BY timestamp DESC LIMIT 1) as latest_position,
                       COUNT(*) as total_checks,
                       country
                FROM rank_history rh1
                GROUP BY keyword, domain
                ORDER BY last_check DESC
                LIMIT ?
            """
            params = (limit,)
        
        rows = db.execute_query(query, params, fetch='all')
        
        keywords = []
        for row in rows:
            if USE_POSTGRES:
                keywords.append(KeywordSummary(
                    keyword=row['keyword'],
                    domain=row['domain'],
                    latest_position=row['latest_position'],
                    previous_position=row.get('previous_position'),
                    trend=row.get('trend', 'stable'),
                    country=row['country'],
                    last_check=str(row['last_check']),
                    total_checks=row['total_checks']
                ))
            else:
                keywords.append({
                    'keyword': row['keyword'],
                    'domain': row['domain'],
                    'latest_position': row['latest_position'],
                    'previous_position': None,
                    'trend': 'stable',
                    'country': row.get('country', 'nl'),
                    'last_check': row['last_check'],
                    'total_checks': row['total_checks']
                })
        
        logger.info(f"📋 Retrieved {len(keywords)} tracked keywords")
        return keywords
        
    except Exception as e:
        logger.error(f"Error getting tracked keywords: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve keywords")

@app.get("/analytics", response_model=AnalyticsResponse)
@limiter.limit("10/minute")
async def get_analytics(
    request: Request,
    days: int = 30,
    api_key: Optional[str] = Depends(verify_api_key) if API_KEY else None
):
    """Get comprehensive analytics"""
    try:
        days = min(days, 365)  # Max 1 year
        
        # Total checks
        total_query = "SELECT COUNT(*) as count FROM rank_history"
        if USE_POSTGRES:
            total_query += " WHERE timestamp > %s"
            since_date = datetime.now() - timedelta(days=days)
            total_params = (since_date,)
        else:
            total_query += " WHERE datetime(timestamp) > datetime('now', '-{} days')".format(days)
            total_params = ()
            
        total_result = db.execute_query(total_query, total_params, fetch='one')
        total_checks = total_result['count'] if total_result else 0
        
        # Unique keywords and domains
        unique_query = """
            SELECT COUNT(DISTINCT keyword) as keywords, 
                   COUNT(DISTINCT domain) as domains 
            FROM rank_history
        """
        if USE_POSTGRES:
            unique_query += " WHERE timestamp > %s"
            unique_params = (since_date,)
        else:
            unique_query += " WHERE datetime(timestamp) > datetime('now', '-{} days')".format(days)
            unique_params = ()
            
        unique_result = db.execute_query(unique_query, unique_params, fetch='one')
        unique_keywords = unique_result['keywords'] if unique_result else 0
        unique_domains = unique_result['domains'] if unique_result else 0
        
        # Average response time en success rate
        if USE_POSTGRES:
            perf_query = """
                SELECT AVG(response_time) as avg_time,
                       COUNT(CASE WHEN status_code = 200 THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM analytics 
                WHERE timestamp > %s AND response_time IS NOT NULL
            """
            perf_params = (since_date,)
        else:
            perf_query = """
                SELECT AVG(response_time) as avg_time,
                       COUNT(CASE WHEN status_code = 200 THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM analytics 
                WHERE datetime(timestamp) > datetime('now', '-{} days') 
                AND response_time IS NOT NULL
            """.format(days)
            perf_params = ()
            
        try:
            perf_result = db.execute_query(perf_query, perf_params, fetch='one')
            avg_response_time = float(perf_result['avg_time'] or 0)
            success_rate = float(perf_result['success_rate'] or 100)
        except:
            avg_response_time = 0.0
            success_rate = 100.0
        
        # Top keywords
        if USE_POSTGRES:
            top_query = """
                SELECT keyword, COUNT(*) as checks,
                       AVG(CASE WHEN position IS NOT NULL THEN position END) as avg_position
                FROM rank_history 
                WHERE timestamp > %s
                GROUP BY keyword 
                ORDER BY checks DESC 
                LIMIT 10
            """
            top_params = (since_date,)
        else:
            top_query = """
                SELECT keyword, COUNT(*) as checks,
                       AVG(CASE WHEN position IS NOT NULL THEN position END) as avg_position
                FROM rank_history 
                WHERE datetime(timestamp) > datetime('now', '-{} days')
                GROUP BY keyword 
                ORDER BY checks DESC 
                LIMIT 10
            """.format(days)
            top_params = ()
            
        top_results = db.execute_query(top_query, top_params, fetch='all')
        top_keywords = []
        for row in top_results:
            top_keywords.append({
                "keyword": row['keyword'],
                "checks": row['checks'],
                "avg_position": round(float(row['avg_position'] or 0), 1)
            })
        
        # Country distribution
        if USE_POSTGRES:
            country_query = """
                SELECT country, COUNT(*) as count
                FROM rank_history 
                WHERE timestamp > %s
                GROUP BY country
                ORDER BY count DESC
            """
            country_params = (since_date,)
        else:
            country_query = """
                SELECT country, COUNT(*) as count
                FROM rank_history 
                WHERE datetime(timestamp) > datetime('now', '-{} days')
                GROUP BY country
                ORDER BY count DESC
            """.format(days)
            country_params = ()
            
        country_results = db.execute_query(country_query, country_params, fetch='all')
        country_distribution = {}
        for row in country_results:
            country_distribution[row['country']] = row['count']
        
        # Recent activity
        if USE_POSTGRES:
            recent_query = """
                SELECT keyword, domain, position, timestamp
                FROM rank_history 
                WHERE timestamp > %s
                ORDER BY timestamp DESC 
                LIMIT 10
            """
            recent_params = (since_date,)
        else:
            recent_query = """
                SELECT keyword, domain, position, timestamp
                FROM rank_history 
                WHERE datetime(timestamp) > datetime('now', '-{} days')
                ORDER BY timestamp DESC 
                LIMIT 10
            """.format(days)
            recent_params = ()
            
        recent_results = db.execute_query(recent_query, recent_params, fetch='all')
        recent_activity = []
        for row in recent_results:
            recent_activity.append({
                "keyword": row['keyword'],
                "domain": row['domain'],
                "position": row['position'],
                "timestamp": str(row['timestamp'])
            })
        
        analytics_response = AnalyticsResponse(
            total_checks=total_checks,
            unique_keywords=unique_keywords,
            unique_domains=unique_domains,
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            top_keywords=top_keywords,
            country_distribution=country_distribution,
            recent_activity=recent_activity
        )
        
        logger.info(f"📈 Analytics generated: {total_checks} total checks, {unique_keywords} keywords")
        return analytics_response
        
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Check /docs voor beschikbare endpoints",
            "available_endpoints": [
                "/check-rank", "/rank-history", "/keywords", 
                "/analytics", "/health", "/docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Er is een onverwachte fout opgetreden"
        }
    )

# Main Application Entry Point
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"
    
    print("=" * 60)
    print("🚀 STARTING PROFESSIONAL RANK TRACKER API")
    print("=" * 60)
    print(f"🌍 Server: http://{host}:{port}")
    print(f"📊 Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print(f"🔒 API Key: {'Required' if API_KEY else 'Not required'}")
    print(f"📝 Documentation: http://{host}:{port}/docs")
    print(f"💡 Environment: {'Production (Render)' if RENDER_ENV else 'Development'}")
    print("=" * 60)
    
    uvicorn.run(
        "rank_tracker_render:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        access_log=True,
        reload=False  # Disable reload in production
    )
