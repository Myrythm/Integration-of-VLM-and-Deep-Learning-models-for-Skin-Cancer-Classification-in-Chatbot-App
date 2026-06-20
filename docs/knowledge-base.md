# Knowledge Base

## Sources

| Source | License | Notes |
|--------|---------|-------|
| [AAD](https://www.aad.org/public) | Free for educational use | ~150 patient-facing pages on skin cancer types |
| [MedlinePlus](https://medlineplus.gov) | Public domain (NIH) | ~50 skin cancer topic pages |
| [DermNet NZ](https://dermnetnz.org) | CC BY-NC-ND | ~400 patient-facing pages, non-commercial only |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov) | Abstracts: fair use | Filtered: skin neoplasms AND patient education AND free full text |

## Ingestion

```bash
# One-time setup
python -m utils.rag.ingestion --source all --rebuild

# Or fetch PubMed first
python data/knowledge_base/pubmed/fetch_pubmed.py --limit 500
python -m utils.rag.ingestion --source pubmed
```

## Updates

- **Guidelines**: Manual re-download when source pages change (rare).
- **PubMed**: Re-run `fetch_pubmed.py` periodically. Append to existing JSON files; idempotent ingestion via SHA256 chunk IDs.
