#!/usr/bin/env python3
"""
Database setup script voor Rank Tracker API
Werkt met zowel PostgreSQL (production) als SQLite (development)
"""

import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_postgresql():
    """Setup PostgreSQL database voor Render.com production"""
    try:
        import psycopg2
        from psycopg2.extras import DictCursor
        
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise Exception("DATABASE_URL environment variable not found")
        
        logger.info("🐘 Setting up PostgreSQL database...")
        
        # Connect naar PostgreSQL
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Create rank_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rank_history (
                id SERIAL PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                domain VARCHAR(255) NOT NULL,
                position INTEGER,
                url TEXT,
                country VARCHAR(10) DEFAULT 'nl',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                search_volume INTEGER,
                CONSTRAINT unique_keyword_domain_timestamp 
                UNIQUE(keyword, domain, timestamp)
            );
        """)
        
        # Create indexes voor performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_keyword_domain 
            ON rank_history(keyword, domain);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON rank_history(timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_country 
            ON rank_history(country);
        """)
        
        # Create analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id SERIAL PRIMARY KEY,
                endpoint VARCHAR(100),
                keyword VARCHAR(255),
                domain VARCHAR(255),
                country VARCHAR(10),
                response_time REAL,
                status_code INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Analytics indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_timestamp 
            ON analytics(timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_endpoint 
            ON analytics(endpoint);
        """)
        
        # Create user preferences table (voor toekomstige features)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE,
                default_country VARCHAR(10) DEFAULT 'nl',
                notification_email VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create keyword alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keyword_alerts (
                id SERIAL PRIMARY KEY,
                keyword VARCHAR(255) NOT NULL,
                domain VARCHAR(255) NOT NULL,
                target_position INTEGER,
                alert_type VARCHAR(50) DEFAULT 'position_change',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_keyword_domain_alert 
                UNIQUE(keyword, domain, alert_type)
            );
        """)
        
        # Commit all changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ PostgreSQL database setup complete!")
        logger.info("📊 Created tables: rank_history, analytics, user_preferences, keyword_alerts")
        
        return True
        
    except ImportError:
        logger.error("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"❌ PostgreSQL setup failed: {e}")
        return False

def setup_sqlite():
    """Setup SQLite database voor local development"""
    try:
        import sqlite3
        
        logger.info("🗃️ Setting up SQLite database...")
        
        conn = sqlite3.connect('rank_tracker.db')
        cursor = conn.cursor()
        
        # Create rank_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rank_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                domain TEXT NOT NULL,
                position INTEGER,
                url TEXT,
                country TEXT DEFAULT 'nl',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                search_volume INTEGER,
                UNIQUE(keyword, domain, timestamp) ON CONFLICT IGNORE
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_keyword_domain 
            ON rank_history(keyword, domain)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON rank_history(timestamp DESC)
        ''')
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                keyword TEXT,
                domain TEXT,
                country TEXT,
                response_time REAL,
                status_code INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                default_country TEXT DEFAULT 'nl',
                notification_email TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create keyword alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                domain TEXT NOT NULL,
                target_position INTEGER,
                alert_type TEXT DEFAULT 'position_change',
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(keyword, domain, alert_type) ON CONFLICT IGNORE
            )
        ''')
        
        # Insert some sample data voor development
        cursor.execute('''
            INSERT OR IGNORE INTO user_preferences (user_id, default_country, notification_email)
            VALUES ('daar-om-default', 'nl', 'info@daar-om.nl')
        ''')
        
        # Sample keyword alerts
        sample_keywords = [
            ('online marketing bureau', 'daar-om.nl', 5),
            ('seo specialist', 'daar-om.nl', 3),
            ('website optimalisatie', 'daar-om.nl', 10)
        ]
        
        for keyword, domain, target_pos in sample_keywords:
            cursor.execute('''
                INSERT OR IGNORE INTO keyword_alerts (keyword, domain, target_position)
                VALUES (?, ?, ?)
            ''', (keyword, domain, target_pos))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ SQLite database setup complete!")
        logger.info("📊 Created tables: rank_history, analytics, user_preferences, keyword_alerts")
        logger.info("🎯 Added sample data voor daar-om.nl")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SQLite setup failed: {e}")
        return False

def verify_database():
    """Verify database setup door test queries uit te voeren"""
    try:
        database_url = os.environ.get('DATABASE_URL')
        
        if database_url:
            # Test PostgreSQL
            import psycopg2
            from psycopg2.extras import DictCursor
            
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            # Test query
            cursor.execute("SELECT COUNT(*) as count FROM rank_history")
            result = cursor.fetchone()
            
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ PostgreSQL verification successful!")
            logger.info(f"📊 Tables found: {[table['table_name'] for table in tables]}")
            logger.info(f"📈 Rank history records: {result['count']}")
            
        else:
            # Test SQLite
            import sqlite3
            
            conn = sqlite3.connect('rank_tracker.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM rank_history")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ SQLite verification successful!")
            logger.info(f"📊 Tables found: {[table[0] for table in tables]}")
            logger.info(f"📈 Rank history records: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False

def setup_database():
    """Main database setup function"""
    logger.info("🚀 Starting database setup...")
    
    # Check if we're on Render.com (PostgreSQL) or local (SQLite)
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url:
        logger.info("🔍 Detected PostgreSQL environment (DATABASE_URL found)")
        success = setup_postgresql()
    else:
        logger.info("🔍 Detected local environment (no DATABASE_URL)")
        success = setup_sqlite()
    
    if success:
        logger.info("🔍 Verifying database setup...")
        verify_success = verify_database()
        
        if verify_success:
            logger.info("🎉 Database setup and verification complete!")
            logger.info("💡 Ready for rank tracking!")
        else:
            logger.warning("⚠️ Database setup complete but verification failed")
    else:
        logger.error("❌ Database setup failed!")
        exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 RANK TRACKER DATABASE SETUP")
    print("=" * 60)
    
    setup_database()
    
    print("=" * 60)
    print("✅ Setup completed!")
    print("🚀 Ready to start rank tracking API!")
    print("=" * 60)
