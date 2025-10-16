import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

# ---------------------------------------------
# STEP 1: Define base index URL and headers
# ---------------------------------------------
INDEX_URL = "https://www.investopedia.com/terms-beginning-with-z-4769376"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FinanceChatbotBot/1.0)"}

# ---------------------------------------------
# STEP 2: Collect all term links
# ---------------------------------------------
print("🔍 Collecting term URLs...")

res = requests.get(INDEX_URL, headers=HEADERS)
soup = BeautifulSoup(res.text, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a.get("href")
    if isinstance(href, (list, tuple)):
        href = href[0] if href else None
    if href is None:
        continue
    href = str(href)
    if href.startswith("https://www.investopedia.com/terms/") and href.endswith(".asp"):
        links.append(href)

links = list(set(links))
print(f"✅ Found {len(links)} term links.")

# ---------------------------------------------
# STEP 3: Scrape each linked page
# ---------------------------------------------
data = []

for url in tqdm(links, desc="📘 Scraping term pages"):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        page = BeautifulSoup(response.text, "html.parser")

        h1 = page.find("h1")
        term_name = h1.get_text(strip=True) if h1 else None
        if not term_name:
            continue

        question = f"What is {term_name} and how does it work?"

        paragraphs = page.find_all("p")
        answer_texts = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            if not text or len(text) < 50:
                continue

            # Skip author bios or attribution lines
            if any(phrase in text for phrase in [" is a ", " holds a ", " editor", "Investopedia /"]):
                continue

            answer_texts.append(text)

        
        if not answer_texts:
            continue
        answer = " ".join(answer_texts)
        if len(answer) > 100:
            data.append({"question": question, "answer": answer})

        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        continue

# ---------------------------------------------
# STEP 4: Save results to CSV
# ---------------------------------------------
df = pd.DataFrame(data)
df.drop_duplicates(subset="question", inplace=True)
df.to_csv("finance_qa_z.csv", index=False, encoding="utf-8")

print(f"\n✅ Scraping complete! Saved {len(df)} Q&A pairs to finance_qa_z.csv.")
