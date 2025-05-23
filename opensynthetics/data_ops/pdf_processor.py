"""Scientific PDF processing module for OpenSynthetics."""

import io
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import requests
from loguru import logger
from pydantic import BaseModel, Field

from opensynthetics.core.exceptions import ProcessingError


class PDFMetadata(BaseModel):
    """Metadata extracted from scientific PDFs."""
    
    title: Optional[str] = Field(None, description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier")
    pubmed_id: Optional[str] = Field(None, description="PubMed identifier")
    journal: Optional[str] = Field(None, description="Journal name")
    publication_date: Optional[str] = Field(None, description="Publication date")
    page_count: int = Field(0, description="Number of pages")
    language: str = Field("en", description="Document language")


class PDFSection(BaseModel):
    """Represents a section of a scientific paper."""
    
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    page_number: int = Field(..., description="Page number where section starts")
    section_type: str = Field("content", description="Type: abstract, introduction, methods, results, conclusion, references")


class ProcessedPDF(BaseModel):
    """Complete processed PDF document."""
    
    metadata: PDFMetadata = Field(..., description="Document metadata")
    sections: List[PDFSection] = Field(default_factory=list, description="Document sections")
    full_text: str = Field("", description="Complete extracted text")
    figures_count: int = Field(0, description="Number of figures")
    tables_count: int = Field(0, description="Number of tables")
    references_count: int = Field(0, description="Number of references")
    file_size: int = Field(0, description="File size in bytes")
    processing_time: float = Field(0.0, description="Processing time in seconds")


class ScientificPDFProcessor:
    """Advanced processor for scientific PDF documents."""

    def __init__(self) -> None:
        """Initialize the PDF processor."""
        self.section_patterns = {
            "abstract": [
                r"(?i)^abstract\s*$",
                r"(?i)^summary\s*$",
                r"(?i)^executive\s+summary\s*$"
            ],
            "introduction": [
                r"(?i)^introduction\s*$",
                r"(?i)^1\.?\s*introduction\s*$",
                r"(?i)^background\s*$"
            ],
            "methods": [
                r"(?i)^methods?\s*$",
                r"(?i)^methodology\s*$",
                r"(?i)^materials?\s+and\s+methods?\s*$",
                r"(?i)^experimental\s+design\s*$"
            ],
            "results": [
                r"(?i)^results?\s*$",
                r"(?i)^findings\s*$",
                r"(?i)^results?\s+and\s+discussion\s*$"
            ],
            "discussion": [
                r"(?i)^discussion\s*$",
                r"(?i)^analysis\s*$"
            ],
            "conclusion": [
                r"(?i)^conclusions?\s*$",
                r"(?i)^summary\s+and\s+conclusions?\s*$",
                r"(?i)^final\s+remarks\s*$"
            ],
            "references": [
                r"(?i)^references?\s*$",
                r"(?i)^bibliography\s*$",
                r"(?i)^works?\s+cited\s*$"
            ]
        }

    def process_pdf_file(self, file_path: Union[str, Path]) -> ProcessedPDF:
        """Process a PDF file from disk.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ProcessedPDF: Processed document data
            
        Raises:
            ProcessingError: If PDF processing fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise ProcessingError(f"PDF file not found: {file_path}")
                
            with open(file_path, "rb") as f:
                pdf_data = f.read()
                
            return self.process_pdf_data(
                pdf_data, 
                filename=file_path.name,
                file_size=file_path.stat().st_size
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            raise ProcessingError(f"Failed to process PDF file: {e}")

    def process_pdf_data(
        self, 
        pdf_data: bytes, 
        filename: str = "document.pdf",
        file_size: Optional[int] = None
    ) -> ProcessedPDF:
        """Process PDF from binary data.
        
        Args:
            pdf_data: PDF binary data
            filename: Original filename
            file_size: File size in bytes
            
        Returns:
            ProcessedPDF: Processed document data
            
        Raises:
            ProcessingError: If PDF processing fails
        """
        import time
        start_time = time.time()
        
        try:
            # Open PDF document
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Extract basic metadata
            metadata = self._extract_metadata(doc, filename)
            
            # Extract text and identify sections
            sections, full_text = self._extract_text_and_sections(doc)
            
            # Analyze document structure
            figures_count = self._count_figures(doc)
            tables_count = self._count_tables(full_text)
            references_count = self._count_references(sections)
            
            # Create processed document
            processed = ProcessedPDF(
                metadata=metadata,
                sections=sections,
                full_text=full_text,
                figures_count=figures_count,
                tables_count=tables_count,
                references_count=references_count,
                file_size=file_size or len(pdf_data),
                processing_time=time.time() - start_time
            )
            
            doc.close()
            logger.info(f"Successfully processed PDF: {filename} ({len(sections)} sections)")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing PDF data: {e}")
            raise ProcessingError(f"Failed to process PDF: {e}")

    def process_pdf_from_url(self, url: str) -> ProcessedPDF:
        """Download and process PDF from URL.
        
        Args:
            url: URL to PDF file
            
        Returns:
            ProcessedPDF: Processed document data
            
        Raises:
            ProcessingError: If download or processing fails
        """
        try:
            logger.info(f"Downloading PDF from URL: {url}")
            
            headers = {
                "User-Agent": "OpenSynthetics PDF Processor 1.0"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            filename = self._extract_filename_from_url(url)
            
            return self.process_pdf_data(
                response.content,
                filename=filename,
                file_size=len(response.content)
            )
            
        except requests.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            raise ProcessingError(f"Failed to download PDF: {e}")
        except Exception as e:
            logger.error(f"Error processing PDF from URL {url}: {e}")
            raise ProcessingError(f"Failed to process PDF from URL: {e}")

    def _extract_metadata(self, doc: fitz.Document, filename: str) -> PDFMetadata:
        """Extract metadata from PDF document."""
        metadata = PDFMetadata()
        
        # Basic document metadata
        pdf_meta = doc.metadata
        metadata.page_count = len(doc)
        
        # Extract title from metadata or first page
        if pdf_meta.get("title"):
            metadata.title = pdf_meta["title"].strip()
        else:
            metadata.title = self._extract_title_from_text(doc)
            
        # Extract authors
        if pdf_meta.get("author"):
            metadata.authors = self._parse_authors(pdf_meta["author"])
        else:
            metadata.authors = self._extract_authors_from_text(doc)
            
        # Extract identifiers and other metadata from text
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        metadata.doi = self._extract_doi(first_page_text)
        metadata.arxiv_id = self._extract_arxiv_id(first_page_text)
        metadata.pubmed_id = self._extract_pubmed_id(first_page_text)
        
        # Extract abstract
        metadata.abstract = self._extract_abstract(doc)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(doc)
        
        return metadata

    def _extract_text_and_sections(self, doc: fitz.Document) -> Tuple[List[PDFSection], str]:
        """Extract text and identify document sections."""
        sections = []
        full_text = ""
        current_section = None
        current_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            full_text += page_text + "\n"
            
            # Split into lines and process
            lines = page_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line is a section header
                section_type = self._identify_section_type(line)
                
                if section_type:
                    # Save previous section
                    if current_section and current_content:
                        sections.append(PDFSection(
                            title=current_section,
                            content='\n'.join(current_content),
                            page_number=page_num,
                            section_type=section_type
                        ))
                    
                    # Start new section
                    current_section = line
                    current_content = []
                else:
                    # Add to current section content
                    if current_section:
                        current_content.append(line)
                    
        # Add final section
        if current_section and current_content:
            sections.append(PDFSection(
                title=current_section,
                content='\n'.join(current_content),
                page_number=len(doc) - 1,
                section_type="content"
            ))
            
        return sections, full_text

    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify if a line is a section header and return its type."""
        line = line.strip()
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line):
                    return section_type
                    
        return None

    def _extract_title_from_text(self, doc: fitz.Document) -> Optional[str]:
        """Extract title from document text."""
        if len(doc) == 0:
            return None
            
        first_page = doc[0].get_text()
        lines = first_page.split('\n')
        
        # Look for title in first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 20 and not line.isupper() and '.' not in line:
                return line
                
        return None

    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of individual authors."""
        if not author_string:
            return []
            
        # Common separators
        separators = [',', ';', ' and ', '&']
        authors = [author_string]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend([a.strip() for a in author.split(sep)])
            authors = new_authors
            
        return [a for a in authors if a and len(a) > 2]

    def _extract_authors_from_text(self, doc: fitz.Document) -> List[str]:
        """Extract authors from document text."""
        # This is a simplified implementation
        # In practice, this would need more sophisticated NLP
        return []

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        doi_pattern = r'(?:doi:|DOI:)?\s*(10\.\d+/[^\s]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text."""
        arxiv_pattern = r'arXiv:(\d+\.\d+)'
        match = re.search(arxiv_pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_pubmed_id(self, text: str) -> Optional[str]:
        """Extract PubMed ID from text."""
        pmid_pattern = r'PMID:?\s*(\d+)'
        match = re.search(pmid_pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_abstract(self, doc: fitz.Document) -> Optional[str]:
        """Extract abstract from document."""
        full_text = ""
        for page_num in range(min(3, len(doc))):  # Check first 3 pages
            full_text += doc[page_num].get_text()
            
        # Look for abstract section
        abstract_pattern = r'(?i)abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|\Z))'
        match = re.search(abstract_pattern, full_text, re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract if len(abstract) > 50 else None
            
        return None

    def _extract_keywords(self, doc: fitz.Document) -> List[str]:
        """Extract keywords from document."""
        full_text = ""
        for page_num in range(min(2, len(doc))):
            full_text += doc[page_num].get_text()
            
        # Look for keywords section
        keywords_pattern = r'(?i)keywords?\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|1\.|\Z))'
        match = re.search(keywords_pattern, full_text, re.DOTALL)
        
        if match:
            keywords_text = match.group(1).strip()
            # Split keywords by common separators
            keywords = re.split(r'[,;]|\sand\s', keywords_text)
            return [k.strip() for k in keywords if k.strip()]
            
        return []

    def _count_figures(self, doc: fitz.Document) -> int:
        """Count figures in document."""
        figure_count = 0
        for page in doc:
            # Count image objects
            figure_count += len(page.get_images())
            
            # Also look for "Figure" references in text
            text = page.get_text()
            figure_refs = re.findall(r'(?i)figure\s+\d+', text)
            figure_count += len(set(figure_refs))
            
        return figure_count

    def _count_tables(self, text: str) -> int:
        """Count tables in document text."""
        table_refs = re.findall(r'(?i)table\s+\d+', text)
        return len(set(table_refs))

    def _count_references(self, sections: List[PDFSection]) -> int:
        """Count references in document."""
        for section in sections:
            if section.section_type == "references":
                # Count numbered references
                ref_pattern = r'^\[\d+\]|\(\d+\)|\d+\.'
                lines = section.content.split('\n')
                ref_count = sum(1 for line in lines if re.match(ref_pattern, line.strip()))
                return ref_count
                
        return 0

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        return filename if filename.endswith('.pdf') else 'document.pdf'

    def extract_training_segments(self, processed_pdf: ProcessedPDF) -> List[Dict[str, Any]]:
        """Extract segments suitable for LLM training data.
        
        Args:
            processed_pdf: Processed PDF document
            
        Returns:
            List of training segments with context
        """
        segments = []
        
        # Add abstract as a summary segment
        if processed_pdf.metadata.abstract:
            segments.append({
                "type": "abstract",
                "title": processed_pdf.metadata.title or "Untitled",
                "content": processed_pdf.metadata.abstract,
                "metadata": {
                    "authors": processed_pdf.metadata.authors,
                    "doi": processed_pdf.metadata.doi,
                    "section": "abstract"
                }
            })
        
        # Add each section as training data
        for section in processed_pdf.sections:
            if len(section.content.strip()) > 100:  # Only substantial content
                segments.append({
                    "type": "section",
                    "title": section.title,
                    "content": section.content,
                    "metadata": {
                        "section_type": section.section_type,
                        "page_number": section.page_number,
                        "paper_title": processed_pdf.metadata.title,
                        "authors": processed_pdf.metadata.authors
                    }
                })
        
        # Add Q&A pairs based on sections
        for i, section in enumerate(processed_pdf.sections):
            if section.section_type in ["methods", "results", "discussion"]:
                segments.append({
                    "type": "qa_pair",
                    "question": f"What are the key points in the {section.section_type} section?",
                    "answer": section.content[:1000] + "..." if len(section.content) > 1000 else section.content,
                    "metadata": {
                        "paper_title": processed_pdf.metadata.title,
                        "section_type": section.section_type,
                        "synthetic": True
                    }
                })
        
        return segments 