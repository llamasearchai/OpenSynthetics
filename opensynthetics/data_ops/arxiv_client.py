"""arXiv API client for OpenSynthetics."""

import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import feedparser
import requests
from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.exceptions import APIError, ProcessingError
from opensynthetics.data_ops.pdf_processor import ProcessedPDF, ScientificPDFProcessor


class ArxivPaper(BaseModel):
    """Represents an arXiv paper."""
    
    id: str = Field(..., description="arXiv ID (e.g., 2301.00001)")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field("", description="Paper abstract")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    primary_category: str = Field("", description="Primary category")
    published: Optional[datetime] = Field(None, description="Publication date")
    updated: Optional[datetime] = Field(None, description="Last update date")
    doi: Optional[str] = Field(None, description="DOI if available")
    journal_ref: Optional[str] = Field(None, description="Journal reference")
    pdf_url: str = Field("", description="PDF download URL")
    entry_url: str = Field("", description="arXiv entry page URL")
    comment: Optional[str] = Field(None, description="Author comments")


class ArxivSearchQuery(BaseModel):
    """Search query parameters for arXiv."""
    
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=2000)
    start: int = Field(0, description="Start index for pagination", ge=0)
    sort_by: str = Field("relevance", description="Sort order: relevance, lastUpdatedDate, submittedDate")
    sort_order: str = Field("descending", description="Sort direction: ascending, descending")
    categories: List[str] = Field(default_factory=list, description="Filter by categories")
    date_from: Optional[datetime] = Field(None, description="Filter papers from this date")
    date_to: Optional[datetime] = Field(None, description="Filter papers to this date")


class ArxivClient:
    """Client for interacting with the arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"
    PDF_BASE_URL = "https://arxiv.org/pdf"
    
    def __init__(self, rate_limit_delay: float = 3.0) -> None:
        """Initialize the arXiv client.
        
        Args:
            rate_limit_delay: Delay between API requests in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.pdf_processor = ScientificPDFProcessor()
        
        # arXiv category mappings
        self.category_mappings = {
            # Computer Science
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            
            # Physics
            "physics.med-ph": "Medical Physics",
            "physics.bio-ph": "Biological Physics",
            "quant-ph": "Quantum Physics",
            
            # Mathematics
            "math.ST": "Statistics Theory",
            "math.PR": "Probability",
            "math.CO": "Combinatorics",
            
            # Biology
            "q-bio.BM": "Biomolecules",
            "q-bio.CB": "Cell Behavior",
            "q-bio.GN": "Genomics",
            "q-bio.QM": "Quantitative Methods",
            
            # Economics
            "econ.EM": "Econometrics",
            "econ.TH": "Theoretical Economics",
        }

    def search(self, query: Union[str, ArxivSearchQuery]) -> List[ArxivPaper]:
        """Search arXiv papers.
        
        Args:
            query: Search query string or ArxivSearchQuery object
            
        Returns:
            List of ArxivPaper objects
            
        Raises:
            APIError: If the API request fails
        """
        if isinstance(query, str):
            search_query = ArxivSearchQuery(query=query)
        else:
            search_query = query
            
        try:
            self._respect_rate_limit()
            
            # Build query parameters
            params = self._build_query_params(search_query)
            
            logger.info(f"Searching arXiv with query: {search_query.query}")
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            papers = self._parse_search_response(response.text)
            
            logger.info(f"Found {len(papers)} papers on arXiv")
            return papers
            
        except requests.RequestException as e:
            logger.error(f"arXiv API request failed: {e}")
            raise APIError(f"Failed to search arXiv: {e}")
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise APIError(f"arXiv search error: {e}")

    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv ID (e.g., "2301.00001" or "cs.AI/0001001")
            
        Returns:
            ArxivPaper object or None if not found
        """
        try:
            # Clean the arXiv ID
            clean_id = self._clean_arxiv_id(arxiv_id)
            
            query = ArxivSearchQuery(
                query=f"id:{clean_id}",
                max_results=1
            )
            
            papers = self.search(query)
            return papers[0] if papers else None
            
        except Exception as e:
            logger.error(f"Error getting paper {arxiv_id}: {e}")
            return None

    def download_pdf(self, paper: ArxivPaper) -> ProcessedPDF:
        """Download and process PDF for an arXiv paper.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            ProcessedPDF object
            
        Raises:
            ProcessingError: If download or processing fails
        """
        try:
            self._respect_rate_limit()
            
            pdf_url = paper.pdf_url or f"{self.PDF_BASE_URL}/{paper.id}.pdf"
            
            logger.info(f"Downloading PDF for paper {paper.id}")
            
            processed_pdf = self.pdf_processor.process_pdf_from_url(pdf_url)
            
            # Enhance metadata with arXiv information
            processed_pdf.metadata.arxiv_id = paper.id
            processed_pdf.metadata.title = paper.title
            processed_pdf.metadata.authors = paper.authors
            processed_pdf.metadata.abstract = paper.abstract
            processed_pdf.metadata.doi = paper.doi
            processed_pdf.metadata.publication_date = paper.published.isoformat() if paper.published else None
            
            return processed_pdf
            
        except Exception as e:
            logger.error(f"Error downloading PDF for {paper.id}: {e}")
            raise ProcessingError(f"Failed to download arXiv PDF: {e}")

    def search_and_download(
        self, 
        query: Union[str, ArxivSearchQuery],
        download_pdfs: bool = True
    ) -> List[Dict[str, Any]]:
        """Search papers and optionally download PDFs.
        
        Args:
            query: Search query
            download_pdfs: Whether to download and process PDFs
            
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
            
            if download_pdfs:
                try:
                    processed_pdf = self.download_pdf(paper)
                    result["processed_pdf"] = processed_pdf
                    result["training_segments"] = self.pdf_processor.extract_training_segments(processed_pdf)
                    
                    logger.info(f"Successfully processed {paper.id} - {len(result['training_segments'])} training segments")
                    
                except Exception as e:
                    logger.warning(f"Failed to process PDF for {paper.id}: {e}")
                    
            results.append(result)
            
        return results

    def get_recent_papers(
        self, 
        categories: List[str], 
        days_back: int = 7,
        max_results: int = 50
    ) -> List[ArxivPaper]:
        """Get recent papers from specified categories.
        
        Args:
            categories: List of arXiv categories (e.g., ["cs.AI", "cs.LG"])
            days_back: Number of days to look back
            max_results: Maximum number of results
            
        Returns:
            List of recent ArxivPaper objects
        """
        # Build category query
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = ArxivSearchQuery(
            query=category_query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
            date_from=start_date,
            date_to=end_date
        )
        
        return self.search(query)

    def _build_query_params(self, search_query: ArxivSearchQuery) -> Dict[str, str]:
        """Build query parameters for arXiv API."""
        params = {
            "search_query": self._build_search_string(search_query),
            "start": str(search_query.start),
            "max_results": str(search_query.max_results),
            "sortBy": search_query.sort_by,
            "sortOrder": search_query.sort_order
        }
        
        return params

    def _build_search_string(self, search_query: ArxivSearchQuery) -> str:
        """Build the search string for arXiv API."""
        query_parts = []
        
        # Main query
        if search_query.query:
            # Handle different query types
            if ":" in search_query.query:
                # Already formatted query (e.g., "cat:cs.AI")
                query_parts.append(search_query.query)
            else:
                # Text search in title and abstract
                escaped_query = quote(search_query.query)
                query_parts.append(f"all:{escaped_query}")
        
        # Category filters
        if search_query.categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in search_query.categories])
            query_parts.append(f"({category_query})")
        
        # Date filters (arXiv uses submittedDate)
        if search_query.date_from or search_query.date_to:
            date_query = "submittedDate:["
            
            if search_query.date_from:
                date_query += search_query.date_from.strftime("%Y%m%d%H%M")
            else:
                date_query += "*"
                
            date_query += " TO "
            
            if search_query.date_to:
                date_query += search_query.date_to.strftime("%Y%m%d%H%M")
            else:
                date_query += "*"
                
            date_query += "]"
            query_parts.append(date_query)
        
        return " AND ".join(query_parts) if query_parts else "all:*"

    def _parse_search_response(self, response_text: str) -> List[ArxivPaper]:
        """Parse arXiv API response."""
        try:
            feed = feedparser.parse(response_text)
            papers = []
            
            for entry in feed.entries:
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue
                    
            return papers
            
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
            raise APIError(f"Failed to parse arXiv response: {e}")

    def _parse_entry(self, entry: Any) -> ArxivPaper:
        """Parse a single arXiv entry."""
        # Extract arXiv ID from URL
        arxiv_id = entry.id.split('/')[-1]
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]  # Remove version number
        
        # Parse authors
        authors = []
        if hasattr(entry, 'authors'):
            authors = [author.name for author in entry.authors]
        elif hasattr(entry, 'author'):
            authors = [entry.author]
        
        # Parse categories
        categories = []
        primary_category = ""
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                if hasattr(tag, 'term'):
                    categories.append(tag.term)
            if categories:
                primary_category = categories[0]
        
        # Parse dates
        published = None
        updated = None
        if hasattr(entry, 'published_parsed'):
            published = datetime(*entry.published_parsed[:6])
        if hasattr(entry, 'updated_parsed'):
            updated = datetime(*entry.updated_parsed[:6])
        
        # Extract DOI from links
        doi = None
        pdf_url = ""
        entry_url = entry.id
        
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.get('title') == 'pdf':
                    pdf_url = link.href
                elif link.get('title') == 'doi':
                    doi = link.href.split('/')[-1] if '/' in link.href else link.href
        
        # Extract comment
        comment = getattr(entry, 'arxiv_comment', None)
        
        # Extract journal reference
        journal_ref = getattr(entry, 'arxiv_journal_ref', None)
        
        return ArxivPaper(
            id=arxiv_id,
            title=entry.title,
            authors=authors,
            abstract=entry.summary,
            categories=categories,
            primary_category=primary_category,
            published=published,
            updated=updated,
            doi=doi,
            journal_ref=journal_ref,
            pdf_url=pdf_url,
            entry_url=entry_url,
            comment=comment
        )

    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and normalize arXiv ID."""
        # Remove arXiv: prefix if present
        if arxiv_id.startswith('arXiv:'):
            arxiv_id = arxiv_id[6:]
        
        # Remove version number if present
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]
        
        return arxiv_id

    def _respect_rate_limit(self) -> None:
        """Implement rate limiting for arXiv API."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def get_category_info(self, category: str) -> Dict[str, str]:
        """Get information about an arXiv category.
        
        Args:
            category: arXiv category code
            
        Returns:
            Dictionary with category information
        """
        return {
            "code": category,
            "name": self.category_mappings.get(category, "Unknown Category"),
            "url": f"https://arxiv.org/list/{category}/recent"
        }

    def generate_training_dataset(
        self, 
        categories: List[str],
        max_papers: int = 100,
        include_abstracts: bool = True,
        include_full_text: bool = True
    ) -> Dict[str, Any]:
        """Generate a comprehensive training dataset from arXiv papers.
        
        Args:
            categories: List of arXiv categories to include
            max_papers: Maximum number of papers to process
            include_abstracts: Whether to include abstract-based training data
            include_full_text: Whether to download and process full PDFs
            
        Returns:
            Dictionary containing training dataset
        """
        logger.info(f"Generating training dataset from {len(categories)} categories")
        
        dataset = {
            "metadata": {
                "source": "arxiv",
                "categories": categories,
                "generated_at": datetime.now().isoformat(),
                "total_papers": 0,
                "total_segments": 0
            },
            "papers": [],
            "training_segments": []
        }
        
        papers_per_category = max_papers // len(categories)
        
        for category in categories:
            try:
                logger.info(f"Processing category: {category}")
                
                query = ArxivSearchQuery(
                    query=f"cat:{category}",
                    max_results=papers_per_category,
                    sort_by="relevance"
                )
                
                results = self.search_and_download(query, download_pdfs=include_full_text)
                
                for result in results:
                    paper_data = {
                        "arxiv_id": result["paper"].id,
                        "title": result["paper"].title,
                        "authors": result["paper"].authors,
                        "category": category,
                        "abstract": result["paper"].abstract,
                        "processed": result["processed_pdf"] is not None
                    }
                    
                    dataset["papers"].append(paper_data)
                    
                    # Add abstract-based training data
                    if include_abstracts and result["paper"].abstract:
                        dataset["training_segments"].append({
                            "type": "abstract_summary",
                            "input": f"Summarize this research paper: {result['paper'].title}",
                            "output": result["paper"].abstract,
                            "metadata": {
                                "arxiv_id": result["paper"].id,
                                "category": category,
                                "authors": result["paper"].authors
                            }
                        })
                    
                    # Add full-text training segments
                    if result["training_segments"]:
                        dataset["training_segments"].extend(result["training_segments"])
                
                logger.info(f"Processed {len(results)} papers from {category}")
                
            except Exception as e:
                logger.error(f"Error processing category {category}: {e}")
                continue
        
        dataset["metadata"]["total_papers"] = len(dataset["papers"])
        dataset["metadata"]["total_segments"] = len(dataset["training_segments"])
        
        logger.info(f"Generated dataset with {dataset['metadata']['total_papers']} papers and {dataset['metadata']['total_segments']} training segments")
        
        return dataset 