# 🔍 Professional Rank Tracker API

Professional SEO rank tracking tool voor **daar-om.nl**.

## 🚀 Features

- ✅ **Real-time Google rank checking**
- ✅ **Multi-country support** (NL, BE, DE, UK, US)
- ✅ **PostgreSQL database** voor historische data
- ✅ **Rate limiting** en anti-blocking measures
- ✅ **iOS SwiftUI app** compatible
- ✅ **Professional API** met FastAPI
- ✅ **Analytics dashboard**

## 🛠️ Tech Stack

- **Backend:** Python + FastAPI
- **Database:** PostgreSQL (production) / SQLite (development)
- **Frontend:** iOS SwiftUI
- **Hosting:** Render.com
- **Scraping:** BeautifulSoup + Requests

## 📱 API Endpoints

- `POST /check-rank` - Check keyword position
- `GET /rank-history` - Get historical data
- `GET /keywords` - List tracked keywords
- `GET /analytics` - Usage statistics
- `GET /health` - Health check

## 🚀 Quick Deploy

### Render.com (Aanbevolen)
1. Fork deze repo
2. Maak PostgreSQL database op Render
3. Deploy web service met deze repo
4. Update iOS app baseURL

### Local Development
```bash
pip install -r requirements.txt
python setup_db.py
python rank_tracker_render.py
