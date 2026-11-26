# JanSahayak — Dynamic Uttarakhand Government Service Navigator

JanSahayak is a conversational, source-grounded research assistant. It is **not restricted to a pre-entered list of government services**. For each user request it:

1. Uses Gemini to understand the citizen's intent and plan searches.
2. Searches the live web using the `ddgs` metasearch package (no separate search API key).
3. Filters search results to HTTPS `.gov.in` and `.nic.in` domains.
4. Opens the official pages, extracts text and relevant same-domain links, and follows selected links.
5. Uses Gemini to prepare a clear guide using only the retrieved official-page evidence.
6. Displays sources, search queries and retrieval status so the user can inspect the research.

The application has **no manually maintained service catalogue, no 12-service cap, and no service-name allowlist**. Each request is researched dynamically. A run can search up to 8 targeted queries, retrieve up to 8 primary official pages, and follow up to 8 relevant official links. These are per-request retrieval limits to control latency, not a limit on which services citizens can ask about.

## Important limitations

- `ddgs` is an open-source metasearch library, not a guaranteed government or commercial search API. Search engines may throttle, block, change their interfaces or return incomplete results. It is suitable for a student prototype, not a service-level guarantee.
- The strict HTTPS `.gov.in` / `.nic.in` allowlist reduces exposure to unofficial sites but can exclude legitimate government systems hosted on other domains. Such sites must be independently reviewed before adding them to a trusted-domain policy.
- A search result is not proof that a page supports a requirement. Exact eligibility, mandatory documents, fees and timelines are only presented as confirmed when the retrieved official page says so.
- Some government pages use JavaScript, CAPTCHA or scanned PDFs and may not be extractable by this prototype. In that case JanSahayak should disclose the limitation rather than guess.
- Gemini free-tier availability, quotas and data-use terms can change. Avoid entering Aadhaar numbers, OTPs, passwords, bank details or private identity documents.
- JanSahayak is independent and not an official Government of Uttarakhand service. It does not submit applications, make payments or replace advice from the relevant authority.

## Requirements

- Python 3.10+ (tested target: Python 3.11)
- Gemini API key: <https://aistudio.google.com/apikey>
- Internet connection

No OpenAI API key, Tavily key, SerpAPI key, government credentials or manually populated service database is required.

## Windows PowerShell installation

Open the extracted project folder (the folder containing `app.py`) in VS Code. Then run:

```powershell
# Create environment (use python, not py, if the Python launcher is missing)
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, run this in the same terminal, then activate again:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

# Install required packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Create the local key file
notepad .env
```

Paste into `.env` (replace the placeholder; do not add quotation marks):

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Save with Ctrl+S. Then launch:

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`. Keep the terminal running while using the app.

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` with the same two environment variables, then:

```bash
python -m streamlit run app.py
```

## Test the research helpers (offline)

```powershell
python -m unittest discover -s tests -v
```

These tests do not call Gemini or the live web search provider.

## Suggested live test prompts

1. `I need to complain about garbage collection in Roorkee.`
2. `I want to apply for a domicile certificate in Haridwar.`
3. `How do I complain about an electricity supply problem in Dehradun?`
4. `My pension application is delayed. Find the official grievance route in Uttarakhand.`
5. `I need to register a birth in my municipality. What is the process?`

For a local issue, enter the town/district in the sidebar or include it in the message.

## Architecture

`app.py` (Streamlit UI) → `jansahayak_agent.py` (plan/search/retrieve/synthesise orchestrator) → `ddgs` live search + `requests`/BeautifulSoup/PyPDF retrieval → Gemini guide synthesis.

The service catalogue from the earlier prototype is intentionally removed. A future cache can store previously researched pages with timestamps, but the cache must never restrict which services can be searched.

## Sources for API integration

- Gemini Python SDK and `generate_content`: <https://ai.google.dev/api/generate-content>
- DDGS package and search interface: <https://pypi.org/project/ddgs/>
- Gemini API pricing and free-tier terms: <https://ai.google.dev/gemini-api/docs/pricing>
