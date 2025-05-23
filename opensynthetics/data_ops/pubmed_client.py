"""PubMed API client for OpenSynthetics."""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.exceptions import APIError, ProcessingError
from opensynthetics.data_ops.pdf_processor import ProcessedPDF, ScientificPDFProcessor


class PubMedPaper(BaseModel):
    """Represents a PubMed paper."""
    
    pmid: str = Field(..., description="PubMed ID")
    title: str = Field("", description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field("", description="Paper abstract")
    keywords: List[str] = Field(default_factory=list, description="MeSH keywords")
    journal: str = Field("", description="Journal name")
    publication_date: Optional[datetime] = Field(None, description="Publication date")
    doi: Optional[str] = Field(None, description="DOI if available")
    pmc_id: Optional[str] = Field(None, description="PMC ID for full text")
    publication_types: List[str] = Field(default_factory=list, description="Publication types")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    grant_list: List[str] = Field(default_factory=list, description="Grant numbers")
    affiliation: str = Field("", description="Author affiliation")
    language: str = Field("english", description="Publication language")


class PubMedSearchQuery(BaseModel):
    """Search query parameters for PubMed."""
    
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results", ge=1, le=10000)
    start: int = Field(0, description="Start index for pagination", ge=0)
    date_from: Optional[datetime] = Field(None, description="Filter papers from this date")
    date_to: Optional[datetime] = Field(None, description="Filter papers to this date")
    publication_types: List[str] = Field(default_factory=list, description="Filter by publication types")
    languages: List[str] = Field(default_factory=list, description="Filter by languages")
    mesh_terms: List[str] = Field(default_factory=list, description="Filter by MeSH terms")
    journals: List[str] = Field(default_factory=list, description="Filter by journal names")


class PubMedClient:
    """Client for interacting with PubMed APIs."""
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 0.34) -> None:
        """Initialize the PubMed client.
        
        Args:
            api_key: NCBI API key for increased rate limits
            rate_limit_delay: Delay between API requests in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay if not api_key else 0.1
        self.last_request_time = 0.0
        self.pdf_processor = ScientificPDFProcessor()
        
        # Tool identifier for NCBI
        self.tool = "OpenSynthetics"
        self.email = "contact@opensynthetics.org"
        
        # Publication type mappings
        self.publication_types = {
            "Journal Article": "Research Article",
            "Review": "Review Article", 
            "Meta-Analysis": "Meta-Analysis",
            "Clinical Trial": "Clinical Trial",
            "Randomized Controlled Trial": "RCT",
            "Case Reports": "Case Report",
            "Letter": "Letter",
            "Editorial": "Editorial"
        }

    def search(self, query: Union[str, PubMedSearchQuery]) -> List[PubMedPaper]:
        """Search PubMed papers.
        
        Args:
            query: Search query string or PubMedSearchQuery object
            
        Returns:
            List of PubMedPaper objects
            
        Raises:
            APIError: If the API request fails
        """
        if isinstance(query, str):
            search_query = PubMedSearchQuery(query=query)
        else:
            search_query = query
            
        try:
            # First, get PMIDs from search
            pmids = self._search_pmids(search_query)
            
            if not pmids:
                logger.info("No papers found for query")
                return []
            
            # Then fetch detailed information
            papers = self._fetch_paper_details(pmids)
            
            logger.info(f"Retrieved {len(papers)} papers from PubMed")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            raise APIError(f"PubMed search error: {e}")

    def get_paper_by_pmid(self, pmid: str) -> Optional[PubMedPaper]:
        """Get a specific paper by PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMedPaper object or None if not found
        """
        try:
            papers = self._fetch_paper_details([pmid])
            return papers[0] if papers else None
            
        except Exception as e:
            logger.error(f"Error getting paper {pmid}: {e}")
            return None

    def download_full_text(self, paper: PubMedPaper) -> Optional[ProcessedPDF]:
        """Download and process full text PDF if available.
        
        Args:
            paper: PubMedPaper object
            
        Returns:
            ProcessedPDF object or None if not available
            
        Raises:
            ProcessingError: If download or processing fails
        """
        try:
            # Try PMC full text first
            if paper.pmc_id:
                pdf_url = f"{self.PMC_BASE_URL}/{paper.pmc_id}/pdf/"
                try:
                    processed_pdf = self.pdf_processor.process_pdf_from_url(pdf_url)
                    
                    # Enhance metadata with PubMed information
                    processed_pdf.metadata.pubmed_id = paper.pmid
                    processed_pdf.metadata.title = paper.title
                    processed_pdf.metadata.authors = paper.authors
                    processed_pdf.metadata.abstract = paper.abstract
                    processed_pdf.metadata.doi = paper.doi
                    processed_pdf.metadata.journal = paper.journal
                    processed_pdf.metadata.keywords = paper.keywords
                    processed_pdf.metadata.publication_date = paper.publication_date.isoformat() if paper.publication_date else None
                    
                    return processed_pdf
                    
                except Exception as e:
                    logger.warning(f"Failed to download PMC full text for {paper.pmid}: {e}")
            
            # Try DOI-based PDF download
            if paper.doi:
                # This would require integration with publisher APIs
                # For now, we'll return None
                logger.info(f"DOI found for {paper.pmid} but publisher API integration not implemented")
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading full text for {paper.pmid}: {e}")
            raise ProcessingError(f"Failed to download PubMed full text: {e}")

    def search_and_process(
        self, 
        query: Union[str, PubMedSearchQuery],
        download_full_text: bool = False
    ) -> List[Dict[str, Any]]:
        """Search papers and optionally download full text.
        
        Args:
            query: Search query
            download_full_text: Whether to attempt full text download
            
        Returns:
            List of papers with optional processed PDF data
        """
        papers = self.search(query)
        results = []
        
        for paper in papers:
            result = {
                "paper": paper,
                "processed_pdf": None,
                "training_segments": None
            }
            
            if download_full_text:
                try:
                    processed_pdf = self.download_full_text(paper)
                    if processed_pdf:
                        result["processed_pdf"] = processed_pdf
                        result["training_segments"] = self.pdf_processor.extract_training_segments(processed_pdf)
                        
                        logger.info(f"Successfully processed {paper.pmid} - {len(result['training_segments'])} training segments")
                    
                except Exception as e:
                    logger.warning(f"Failed to process full text for {paper.pmid}: {e}")
            
            # Create training segments from abstract if no full text
            if not result["training_segments"] and paper.abstract:
                result["training_segments"] = self._create_abstract_segments(paper)
                    
            results.append(result)
            
        return results

    def get_recent_papers(
        self, 
        subject_areas: List[str], 
        days_back: int = 7,
        max_results: int = 50
    ) -> List[PubMedPaper]:
        """Get recent papers from specified subject areas.
        
        Args:
            subject_areas: List of MeSH terms or keywords
            days_back: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of recent PubMedPaper objects
        """
        # Build subject query
        subject_query = " OR ".join([f'"{area}"[MeSH Terms]' for area in subject_areas])
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = PubMedSearchQuery(
            query=f"({subject_query})",
            max_results=max_results,
            date_from=start_date,
            date_to=end_date
        )
        
        return self.search(query)

    def _search_pmids(self, search_query: PubMedSearchQuery) -> List[str]:
        """Search for PMIDs using ESearch."""
        self._respect_rate_limit()
        
        params = {
            "db": "pubmed",
            "term": self._build_search_term(search_query),
            "retmax": str(search_query.max_results),
            "retstart": str(search_query.start),
            "retmode": "xml",
            "tool": self.tool,
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        logger.info(f"Searching PubMed with term: {params['term']}")
        
        response = requests.get(self.ESEARCH_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        pmids = []
        
        for id_elem in root.findall(".//Id"):
            if id_elem.text:
                pmids.append(id_elem.text)
        
        logger.info(f"Found {len(pmids)} PMIDs")
        return pmids

    def _fetch_paper_details(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch detailed paper information using EFetch."""
        if not pmids:
            return []
        
        # Process in batches to avoid URL length limits
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            self._respect_rate_limit()
            
            params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml",
                "rettype": "abstract",
                "tool": self.tool,
                "email": self.email
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            logger.info(f"Fetching details for {len(batch_pmids)} papers")
            
            response = requests.get(self.EFETCH_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_paper_details(response.content)
            all_papers.extend(papers)
        
        return all_papers

    def _parse_paper_details(self, xml_content: bytes) -> List[PubMedPaper]:
        """Parse paper details from EFetch XML response."""
        try:
            root = ET.fromstring(xml_content)
            papers = []
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_single_article(article)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
                    
            return papers
            
        except Exception as e:
            logger.error(f"Error parsing paper details: {e}")
            return []

    def _parse_single_article(self, article_elem: ET.Element) -> PubMedPaper:
        """Parse a single article from XML."""
        paper = PubMedPaper(pmid="")
        
        # PMID
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text:
            paper.pmid = pmid_elem.text
        
        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        if title_elem is not None and title_elem.text:
            paper.title = title_elem.text.strip()
        
        # Abstract
        abstract_elems = article_elem.findall(".//AbstractText")
        if abstract_elems:
            abstract_parts = []
            for abs_elem in abstract_elems:
                if abs_elem.text:
                    label = abs_elem.get("Label", "")
                    text = abs_elem.text.strip()
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            paper.abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            first_name = author.find("ForeName")
            if last_name is not None and first_name is not None:
                authors.append(f"{first_name.text} {last_name.text}")
            elif last_name is not None:
                authors.append(last_name.text)
        paper.authors = authors
        
        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        if journal_elem is not None and journal_elem.text:
            paper.journal = journal_elem.text
        
        # Publication date
        pub_date = article_elem.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            
            if year is not None and year.text:
                try:
                    year_val = int(year.text)
                    month_val = 1
                    day_val = 1
                    
                    if month is not None and month.text:
                        try:
                            month_val = int(month.text)
                        except ValueError:
                            # Handle month names
                            month_names = {
                                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                            }
                            month_val = month_names.get(month.text, 1)
                    
                    if day is not None and day.text:
                        try:
                            day_val = int(day.text)
                        except ValueError:
                            day_val = 1
                    
                    paper.publication_date = datetime(year_val, month_val, day_val)
                except ValueError:
                    pass
        
        # DOI
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi" and article_id.text:
                paper.doi = article_id.text
            elif article_id.get("IdType") == "pmc" and article_id.text:
                paper.pmc_id = article_id.text
        
        # MeSH terms
        mesh_terms = []
        for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)
        paper.mesh_terms = mesh_terms
        
        # Keywords
        keywords = []
        for keyword in article_elem.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)
        paper.keywords = keywords
        
        # Publication types
        pub_types = []
        for pub_type in article_elem.findall(".//PublicationType"):
            if pub_type.text:
                pub_types.append(pub_type.text)
        paper.publication_types = pub_types
        
        # Affiliation (first author's affiliation)
        affiliation_elem = article_elem.find(".//Affiliation")
        if affiliation_elem is not None and affiliation_elem.text:
            paper.affiliation = affiliation_elem.text
        
        # Language
        language_elem = article_elem.find(".//Language")
        if language_elem is not None and language_elem.text:
            paper.language = language_elem.text.lower()
        
        return paper

    def _build_search_term(self, search_query: PubMedSearchQuery) -> str:
        """Build the search term for PubMed."""
        terms = []
        
        # Main query
        if search_query.query:
            terms.append(search_query.query)
        
        # Date filters
        if search_query.date_from or search_query.date_to:
            date_filter = ""
            if search_query.date_from and search_query.date_to:
                start_date = search_query.date_from.strftime("%Y/%m/%d")
                end_date = search_query.date_to.strftime("%Y/%m/%d")
                date_filter = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            elif search_query.date_from:
                start_date = search_query.date_from.strftime("%Y/%m/%d")
                date_filter = f'"{start_date}"[Date - Publication] : 3000[Date - Publication]'
            elif search_query.date_to:
                end_date = search_query.date_to.strftime("%Y/%m/%d")
                date_filter = f'1800[Date - Publication] : "{end_date}"[Date - Publication]'
            
            if date_filter:
                terms.append(date_filter)
        
        # Publication types
        if search_query.publication_types:
            pub_type_terms = [f'"{pt}"[Publication Type]' for pt in search_query.publication_types]
            terms.append(f"({' OR '.join(pub_type_terms)})")
        
        # Languages
        if search_query.languages:
            lang_terms = [f'"{lang}"[Language]' for lang in search_query.languages]
            terms.append(f"({' OR '.join(lang_terms)})")
        
        # MeSH terms
        if search_query.mesh_terms:
            mesh_terms = [f'"{term}"[MeSH Terms]' for term in search_query.mesh_terms]
            terms.append(f"({' OR '.join(mesh_terms)})")
        
        # Journals
        if search_query.journals:
            journal_terms = [f'"{journal}"[Journal]' for journal in search_query.journals]
            terms.append(f"({' OR '.join(journal_terms)})")
        
        return " AND ".join(terms) if terms else "*"

    def _create_abstract_segments(self, paper: PubMedPaper) -> List[Dict[str, Any]]:
        """Create training segments from abstract only."""
        segments = []
        
        if paper.abstract:
            # Abstract summary segment
            segments.append({
                "type": "abstract",
                "title": paper.title,
                "content": paper.abstract,
                "metadata": {
                    "pmid": paper.pmid,
                    "authors": paper.authors,
                    "journal": paper.journal,
                    "doi": paper.doi,
                    "mesh_terms": paper.mesh_terms,
                    "publication_types": paper.publication_types
                }
            })
            
            # Q&A pair from abstract
            segments.append({
                "type": "qa_pair",
                "question": f"What is the main finding of the study titled '{paper.title}'?",
                "answer": paper.abstract,
                "metadata": {
                    "pmid": paper.pmid,
                    "journal": paper.journal,
                    "synthetic": True
                }
            })
        
        return segments

    def _respect_rate_limit(self) -> None:
        """Implement rate limiting for NCBI APIs."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def generate_medical_training_dataset(
        self, 
        medical_areas: List[str],
        max_papers: int = 200,
        include_full_text: bool = False
    ) -> Dict[str, Any]:
        """Generate a comprehensive medical training dataset.
        
        Args:
            medical_areas: List of medical subject areas or MeSH terms
            max_papers: Maximum number of papers to process
            include_full_text: Whether to attempt full text download
            
        Returns:
            Dictionary containing medical training dataset
        """
        logger.info(f"Generating medical training dataset from {len(medical_areas)} areas")
        
        dataset = {
            "metadata": {
                "source": "pubmed",
                "medical_areas": medical_areas,
                "generated_at": datetime.now().isoformat(),
                "total_papers": 0,
                "total_segments": 0
            },
            "papers": [],
            "training_segments": []
        }
        
        papers_per_area = max_papers // len(medical_areas)
        
        for area in medical_areas:
            try:
                logger.info(f"Processing medical area: {area}")
                
                query = PubMedSearchQuery(
                    query=f'"{area}"[MeSH Terms]',
                    max_results=papers_per_area
                )
                
                results = self.search_and_process(query, download_full_text=include_full_text)
                
                for result in results:
                    paper_data = {
                        "pmid": result["paper"].pmid,
                        "title": result["paper"].title,
                        "authors": result["paper"].authors,
                        "journal": result["paper"].journal,
                        "medical_area": area,
                        "mesh_terms": result["paper"].mesh_terms,
                        "publication_types": result["paper"].publication_types,
                        "processed": result["processed_pdf"] is not None
                    }
                    
                    dataset["papers"].append(paper_data)
                    
                    # Add training segments
                    if result["training_segments"]:
                        dataset["training_segments"].extend(result["training_segments"])
                
                logger.info(f"Processed {len(results)} papers from {area}")
                
            except Exception as e:
                logger.error(f"Error processing medical area {area}: {e}")
                continue
        
        dataset["metadata"]["total_papers"] = len(dataset["papers"])
        dataset["metadata"]["total_segments"] = len(dataset["training_segments"])
        
        logger.info(f"Generated medical dataset with {dataset['metadata']['total_papers']} papers and {dataset['metadata']['total_segments']} training segments")
        
        return dataset 