from unittest.mock import patch

from utils.rag.pubmed import fetch_pubmed_abstracts, parse_pubmed_xml


SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Melanoma patient education trial</ArticleTitle>
        <Abstract>
          <AbstractText>This study evaluated patient education materials for early melanoma detection.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Journal of Dermatological Education</Title>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
              <Month>06</Month>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml() -> None:
    results = parse_pubmed_xml(SAMPLE_XML)
    assert len(results) == 1
    assert results[0]["pmid"] == "12345678"
    assert "patient education" in results[0]["abstract"].lower()
    assert results[0]["title"] == "Melanoma patient education trial"


def test_fetch_pubmed_abstracts_uses_api() -> None:
    with patch("utils.rag.pubmed._esearch") as mock_search, \
         patch("utils.rag.pubmed._efetch") as mock_fetch:
        mock_search.return_value = ["12345678", "87654321"]
        mock_fetch.return_value = SAMPLE_XML

        results = fetch_pubmed_abstracts("melanoma education", max_results=2)

    assert len(results) == 1
    mock_search.assert_called_once()
    mock_fetch.assert_called_once()
