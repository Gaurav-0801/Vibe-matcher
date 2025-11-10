# Vibe Matcher Setup Guide

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Clone the Repository
\`\`\`bash
git clone <your-repo-url>
cd vibe-matcher
\`\`\`

### 2. Create Virtual Environment (Recommended)
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install pandas numpy scikit-learn matplotlib requests python-dotenv
\`\`\`

### 4. Configure OpenAI API Key

**Option A: Using .env file (Recommended for development)**

1. Copy `.env.example` to `.env`:
   \`\`\`bash
   cp .env.example .env
   \`\`\`

2. Add your OpenAI API key to `.env`:
   \`\`\`
   OPENAI_API_KEY=sk-proj-your-actual-api-key-here
   \`\`\`

3. Get your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)

**Option B: System Environment Variable**
\`\`\`bash
export OPENAI_API_KEY=sk-proj-your-api-key-here  # macOS/Linux
set OPENAI_API_KEY=sk-proj-your-api-key-here     # Windows
\`\`\`

### 5. Run the Vibe Matcher
\`\`\`bash
python scripts/vibe_matcher.py
\`\`\`

## Output Files
After running, you'll find:
- `vibe_matcher_results.png` - Visual analytics dashboard
- `vibe_matcher_reflection.txt` - Strategic insights and recommendations

## Security Notes
- ⚠️ **NEVER commit `.env` to version control** (it's in `.gitignore`)
- Use `.env.example` as a template for team members
- Rotate your API key regularly if exposed
- Monitor API usage at https://platform.openai.com/account/usage/limits

## Troubleshooting

### "OpenAI API key not found" Error
- Verify `.env` file exists in project root
- Check API key is correctly formatted: `sk-proj-...`
- Ensure `.env` file has proper permissions

### "Rate limited" Warnings
- API quota exceeded - check usage at OpenAI dashboard
- Script automatically retries with exponential backoff
- Consider batching queries or upgrading API plan

### Module Not Found Errors
- Run: `pip install -r requirements.txt`
- Or manually install: `pip install pandas numpy scikit-learn matplotlib requests python-dotenv`
