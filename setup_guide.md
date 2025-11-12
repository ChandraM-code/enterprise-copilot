# Quick Setup Guide

Follow these steps to get the application running quickly:

## Step 1: Start Redis and Qdrant

### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

This will start both Redis and Qdrant in the background.

### Option B: Using Individual Docker Commands

```bash
# Start Redis
docker run -d --name redis-cache -p 6379:6379 redis:latest

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Option C: Install Locally

**Redis:**
- Windows: Download from https://redis.io/download
- Linux: `sudo apt-get install redis-server && sudo service redis-server start`
- Mac: `brew install redis && brew services start redis`

**Qdrant:**
- Follow instructions at https://qdrant.tech/documentation/quick-start/

## Step 2: Verify Services are Running

```bash
# Check Redis
redis-cli ping
# Should return: PONG

# Check Qdrant
curl http://localhost:6333/healthz
# Should return: OK
```

## Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Configure API Keys

Create a `.env` file in the project root:

```env
# At least one of these is required:
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Override defaults
DEFAULT_LLM=openai
REDIS_HOST=localhost
QDRANT_HOST=localhost
```

## Step 5: Start the Application

```bash
python main.py
```

The application will start on http://localhost:8000

## Step 6: Test the Application

### Option A: Use the Test Script

```bash
python test_api.py
```

### Option B: Open Interactive Docs

Visit http://localhost:8000/docs in your browser

### Option C: Use cURL

```bash
# Add a document
curl -X POST "http://localhost:8000/api/documents" \
  -H "Content-Type: application/json" \
  -d '{"content": "Python is a programming language", "metadata": {"topic": "programming"}}'

# Query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

## Troubleshooting

### Issue: "Could not connect to Redis"
**Solution:**
```bash
# Check if Redis is running
docker ps | grep redis
# Or
redis-cli ping
```

### Issue: "Could not connect to Qdrant"
**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant
# Or
curl http://localhost:6333/healthz
```

### Issue: "OpenAI API key not configured"
**Solution:**
- Make sure you have created a `.env` file
- Verify the API key is correct
- Ensure the `.env` file is in the project root directory

### Issue: "Module not found"
**Solution:**
```bash
# Make sure you're in the virtual environment
# Then reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

1. Add your own documents to the RAG cache
2. Test different types of queries
3. Experiment with cache thresholds
4. Try different LLM providers

## Stopping the Application

```bash
# Stop the application: Ctrl+C

# Stop Docker services
docker-compose down

# Or stop individual containers
docker stop redis-cache qdrant
```

## Need Help?

- Check the main README.md for detailed documentation
- Visit http://localhost:8000/docs for API documentation
- Run the test script: `python test_api.py`

