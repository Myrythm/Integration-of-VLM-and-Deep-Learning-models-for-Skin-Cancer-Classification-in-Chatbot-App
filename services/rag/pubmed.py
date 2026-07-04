import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import urlopen

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_QUERY = (
    '("skin neoplasms"[MeSH] OR melanoma OR "basal cell" OR "squamous cell") '
    'AND English AND free full text[Filter]'
)
RATE_LIMIT_SECONDS = 0.4


def _esearch(query: str, max_results: int) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "json"}
    url = f"{ESEARCH_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> str:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    url = f"{EFETCH_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="")
        abstract = " ".join(
            "".join(t.itertext()) for t in article.findall(".//AbstractText")
        )
        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//PubDate/Year", default="")
        month = article.findtext(".//PubDate/Month", default="")
        if month:
            month_lower = month.lower()[:3]
            month_map = {
                "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
                "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
            }
            if month_lower in month_map:
                month = month_map[month_lower]
            elif month.isdigit():
                month = month.zfill(2)
            else:
                month = ""
        publish_date = f"{year}-{month}" if year and month else year
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        out.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publish_date": publish_date,
            "url": url,
        })
    return out


def fetch_pubmed_abstracts(query: str = DEFAULT_QUERY, max_results: int = 500) -> list[dict]:
    pmids = _esearch(query, max_results)
    if not pmids:
        return []
    
    results = []
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        time.sleep(RATE_LIMIT_SECONDS)
        batch_pmids = pmids[i:i + batch_size]
        xml_text = _efetch(batch_pmids)
        results.extend(parse_pubmed_xml(xml_text))
    return results
