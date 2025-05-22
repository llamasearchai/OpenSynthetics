"""Advanced data generation strategies for OpenSynthetics."""

import json
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, validator

from opensynthetics.core.config import Config
from opensynthetics.core.exceptions import GenerationError
from opensynthetics.datagen.engine import GenerationStrategy
from opensynthetics.llm_core.agents.generator import GeneratorAgent


class ConversationalDataParams(BaseModel):
    """Parameters for conversational data generation."""
    
    domain: str = Field(..., description="Conversation domain (e.g., customer_service, technical_support, casual)")
    count: int = Field(..., description="Number of conversations to generate", gt=0)
    turns_per_conversation: int = Field(default=6, description="Number of turns per conversation", ge=2, le=20)
    style: str = Field(default="professional", description="Conversation style")
    
    @validator('domain')
    def validate_domain(cls, v):
        """Validate domain."""
        valid_domains = ['customer_service', 'technical_support', 'casual', 'educational', 'sales', 'medical']
        if v.lower() not in valid_domains:
            raise ValueError(f"Domain must be one of: {', '.join(valid_domains)}")
        return v.lower()
    
    @validator('style')
    def validate_style(cls, v):
        """Validate style."""
        valid_styles = ['professional', 'casual', 'formal', 'friendly', 'technical']
        if v.lower() not in valid_styles:
            raise ValueError(f"Style must be one of: {', '.join(valid_styles)}")
        return v.lower()


class ResearchPaperParams(BaseModel):
    """Parameters for research paper generation."""
    
    field: str = Field(..., description="Research field (e.g., machine_learning, biology, physics)")
    count: int = Field(..., description="Number of papers to generate", gt=0)
    paper_type: str = Field(default="full", description="Type of paper content to generate")
    complexity: str = Field(default="intermediate", description="Complexity level")
    
    @validator('field')
    def validate_field(cls, v):
        """Validate research field."""
        valid_fields = [
            'machine_learning', 'computer_science', 'biology', 'physics', 'chemistry',
            'mathematics', 'engineering', 'medicine', 'psychology', 'economics'
        ]
        if v.lower() not in valid_fields:
            raise ValueError(f"Field must be one of: {', '.join(valid_fields)}")
        return v.lower()
    
    @validator('paper_type')
    def validate_paper_type(cls, v):
        """Validate paper type."""
        valid_types = ['abstract', 'introduction', 'full', 'methodology', 'results']
        if v.lower() not in valid_types:
            raise ValueError(f"Paper type must be one of: {', '.join(valid_types)}")
        return v.lower()
    
    @validator('complexity')
    def validate_complexity(cls, v):
        """Validate complexity level."""
        valid_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        if v.lower() not in valid_levels:
            raise ValueError(f"Complexity must be one of: {', '.join(valid_levels)}")
        return v.lower()


class CodeGenerationParams(BaseModel):
    """Parameters for code generation."""
    
    language: str = Field(..., description="Programming language")
    count: int = Field(..., description="Number of code snippets to generate", gt=0)
    problem_type: str = Field(..., description="Type of programming problem")
    difficulty: str = Field(default="medium", description="Difficulty level")
    include_tests: bool = Field(default=True, description="Whether to include unit tests")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate programming language."""
        valid_languages = [
            'python', 'javascript', 'java', 'cpp', 'c', 'go', 'rust',
            'typescript', 'swift', 'kotlin', 'ruby', 'php', 'scala'
        ]
        if v.lower() not in valid_languages:
            raise ValueError(f"Language must be one of: {', '.join(valid_languages)}")
        return v.lower()
    
    @validator('problem_type')
    def validate_problem_type(cls, v):
        """Validate problem type."""
        valid_types = [
            'algorithms', 'data_structures', 'web_development', 'machine_learning',
            'system_design', 'database', 'api_development', 'utilities'
        ]
        if v.lower() not in valid_types:
            raise ValueError(f"Problem type must be one of: {', '.join(valid_types)}")
        return v.lower()
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        """Validate difficulty level."""
        valid_levels = ['easy', 'medium', 'hard', 'expert']
        if v.lower() not in valid_levels:
            raise ValueError(f"Difficulty must be one of: {', '.join(valid_levels)}")
        return v.lower()


class MultiModalDataParams(BaseModel):
    """Parameters for multi-modal data generation."""
    
    modalities: List[str] = Field(..., description="List of modalities to include")
    count: int = Field(..., description="Number of multi-modal samples to generate", gt=0)
    use_case: str = Field(..., description="Use case for the multi-modal data")
    alignment_quality: str = Field(default="high", description="Quality of alignment between modalities")
    
    @validator('modalities')
    def validate_modalities(cls, v):
        """Validate modalities."""
        valid_modalities = ['text', 'image_description', 'audio_description', 'video_description', 'code']
        if not v or len(v) < 2:
            raise ValueError("At least 2 modalities must be specified")
        
        for modality in v:
            if modality.lower() not in valid_modalities:
                raise ValueError(f"Each modality must be one of: {', '.join(valid_modalities)}")
        
        return [m.lower() for m in v]
    
    @validator('use_case')
    def validate_use_case(cls, v):
        """Validate use case."""
        valid_cases = [
            'content_creation', 'education', 'entertainment', 'research',
            'accessibility', 'translation', 'summarization'
        ]
        if v.lower() not in valid_cases:
            raise ValueError(f"Use case must be one of: {', '.join(valid_cases)}")
        return v.lower()


class ConversationalDataStrategy(GenerationStrategy):
    """Strategy for generating conversational data."""
    
    parameter_model = ConversationalDataParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate conversational data.
        
        Returns:
            List[Dict[str, Any]]: Generated conversations
            
        Raises:
            GenerationError: If generation fails
        """
        domain = self.parameters["domain"]
        count = self.parameters["count"]
        turns_per_conversation = self.parameters["turns_per_conversation"]
        style = self.parameters["style"]
        
        try:
            agent = GeneratorAgent(config=self.config)
            conversations = []
            
            for i in range(count):
                prompt = f"""
                Generate a realistic {style} conversation in the {domain} domain with exactly {turns_per_conversation} turns.
                
                Format the output as JSON:
                {{
                    "id": "conv_{i+1}",
                    "domain": "{domain}",
                    "style": "{style}",
                    "turns": [
                        {{"speaker": "user", "message": "user message"}},
                        {{"speaker": "assistant", "message": "assistant response"}},
                        // ... continue for {turns_per_conversation} turns
                    ],
                    "metadata": {{
                        "topic": "conversation topic",
                        "resolution": "resolved/unresolved/escalated",
                        "sentiment": "positive/neutral/negative",
                        "generated_timestamp": "{time.time()}"
                    }}
                }}
                
                Make the conversation natural and contextually appropriate for {domain}.
                Return only valid JSON with no additional text.
                """
                
                logger.info(f"Generating {domain} conversation #{i+1}")
                response = agent.generate(prompt=prompt)
                
                try:
                    # Parse JSON response
                    conversation_json = response.strip()
                    if conversation_json.find('{') > 0:
                        conversation_json = conversation_json[conversation_json.find('{'):]
                    if conversation_json.rfind('}') < len(conversation_json) - 1:
                        conversation_json = conversation_json[:conversation_json.rfind('}')+1]
                    
                    conversation = json.loads(conversation_json)
                    
                    # Validate structure
                    if "turns" not in conversation or len(conversation["turns"]) != turns_per_conversation:
                        raise ValueError("Invalid conversation structure")
                    
                    conversations.append(conversation)
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing conversation JSON: {e}")
                    # Create fallback conversation
                    conversation = {
                        "id": f"conv_{i+1}",
                        "domain": domain,
                        "style": style,
                        "turns": [
                            {"speaker": "user", "message": "Hello, I need help."},
                            {"speaker": "assistant", "message": "I'd be happy to help you. What can I assist you with today?"}
                        ],
                        "metadata": {
                            "topic": "general_inquiry",
                            "resolution": "unresolved",
                            "sentiment": "neutral",
                            "generated_timestamp": time.time(),
                            "error": f"JSON parse error: {str(e)}"
                        }
                    }
                    conversations.append(conversation)
            
            logger.info(f"Generated {len(conversations)} {domain} conversations")
            return conversations
            
        except Exception as e:
            logger.error(f"Error generating conversational data: {e}")
            raise GenerationError(f"Failed to generate conversational data: {e}")


class ResearchPaperStrategy(GenerationStrategy):
    """Strategy for generating research paper content."""
    
    parameter_model = ResearchPaperParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate research paper content.
        
        Returns:
            List[Dict[str, Any]]: Generated research papers
            
        Raises:
            GenerationError: If generation fails
        """
        field = self.parameters["field"]
        count = self.parameters["count"]
        paper_type = self.parameters["paper_type"]
        complexity = self.parameters["complexity"]
        
        try:
            agent = GeneratorAgent(config=self.config)
            papers = []
            
            for i in range(count):
                prompt = f"""
                Generate a {complexity}-level research paper {paper_type} in the field of {field}.
                
                Format the output as JSON:
                {{
                    "id": "paper_{i+1}",
                    "title": "paper title",
                    "authors": ["Author 1", "Author 2"],
                    "abstract": "paper abstract (150-250 words)",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "field": "{field}",
                    "complexity": "{complexity}",
                    {"content": "full paper content (if paper_type is full)," if paper_type == "full" else ""}
                    {"introduction": "introduction section," if paper_type in ["introduction", "full"] else ""}
                    {"methodology": "methodology section," if paper_type in ["methodology", "full"] else ""}
                    {"results": "results section," if paper_type in ["results", "full"] else ""}
                    "references": [
                        "Reference 1 in proper academic format",
                        "Reference 2 in proper academic format"
                    ],
                    "metadata": {{
                        "publication_year": 2024,
                        "journal": "appropriate journal name",
                        "doi": "10.1000/example.doi",
                        "generated_timestamp": "{time.time()}"
                    }}
                }}
                
                Make the content scientifically accurate and well-structured for {field}.
                Return only valid JSON with no additional text.
                """
                
                logger.info(f"Generating {field} research paper #{i+1}")
                response = agent.generate(prompt=prompt)
                
                try:
                    # Parse JSON response
                    paper_json = response.strip()
                    if paper_json.find('{') > 0:
                        paper_json = paper_json[paper_json.find('{'):]
                    if paper_json.rfind('}') < len(paper_json) - 1:
                        paper_json = paper_json[:paper_json.rfind('}')+1]
                    
                    paper = json.loads(paper_json)
                    papers.append(paper)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing paper JSON: {e}")
                    # Create fallback paper
                    paper = {
                        "id": f"paper_{i+1}",
                        "title": f"Research in {field.replace('_', ' ').title()}",
                        "authors": ["AI Generated"],
                        "abstract": response[:500] + "..." if len(response) > 500 else response,
                        "keywords": [field, "research", "analysis"],
                        "field": field,
                        "complexity": complexity,
                        "references": [],
                        "metadata": {
                            "publication_year": 2024,
                            "journal": f"Journal of {field.replace('_', ' ').title()}",
                            "doi": f"10.1000/generated.{i+1}",
                            "generated_timestamp": time.time(),
                            "error": f"JSON parse error: {str(e)}"
                        }
                    }
                    papers.append(paper)
            
            logger.info(f"Generated {len(papers)} {field} research papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error generating research papers: {e}")
            raise GenerationError(f"Failed to generate research papers: {e}")


class CodeGenerationStrategy(GenerationStrategy):
    """Strategy for generating code snippets and programming problems."""
    
    parameter_model = CodeGenerationParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate code snippets.
        
        Returns:
            List[Dict[str, Any]]: Generated code snippets
            
        Raises:
            GenerationError: If generation fails
        """
        language = self.parameters["language"]
        count = self.parameters["count"]
        problem_type = self.parameters["problem_type"]
        difficulty = self.parameters["difficulty"]
        include_tests = self.parameters["include_tests"]
        
        try:
            agent = GeneratorAgent(config=self.config)
            code_snippets = []
            
            for i in range(count):
                prompt = f"""
                Generate a {difficulty}-level {language} programming solution for a {problem_type} problem.
                
                Format the output as JSON:
                {{
                    "id": "code_{i+1}",
                    "problem_description": "Clear description of the problem to solve",
                    "language": "{language}",
                    "problem_type": "{problem_type}",
                    "difficulty": "{difficulty}",
                    "solution": "complete working code solution",
                    {"test_cases": "unit tests for the solution"," if include_tests else ""}
                    "explanation": "explanation of the approach and algorithm",
                    "time_complexity": "time complexity analysis",
                    "space_complexity": "space complexity analysis",
                    "example_usage": "example of how to use the code",
                    "metadata": {{
                        "lines_of_code": "estimated number of lines",
                        "generated_timestamp": "{time.time()}"
                    }}
                }}
                
                Ensure the code is syntactically correct and follows {language} best practices.
                Return only valid JSON with no additional text.
                """
                
                logger.info(f"Generating {language} {problem_type} code #{i+1}")
                response = agent.generate(prompt=prompt)
                
                try:
                    # Parse JSON response
                    code_json = response.strip()
                    if code_json.find('{') > 0:
                        code_json = code_json[code_json.find('{'):]
                    if code_json.rfind('}') < len(code_json) - 1:
                        code_json = code_json[:code_json.rfind('}')+1]
                    
                    code_snippet = json.loads(code_json)
                    code_snippets.append(code_snippet)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing code JSON: {e}")
                    # Create fallback code snippet
                    code_snippet = {
                        "id": f"code_{i+1}",
                        "problem_description": f"Generate a {problem_type} solution in {language}",
                        "language": language,
                        "problem_type": problem_type,
                        "difficulty": difficulty,
                        "solution": response,
                        "explanation": "Code generation response",
                        "time_complexity": "O(n)",
                        "space_complexity": "O(1)",
                        "example_usage": "See solution above",
                        "metadata": {
                            "lines_of_code": "unknown",
                            "generated_timestamp": time.time(),
                            "error": f"JSON parse error: {str(e)}"
                        }
                    }
                    code_snippets.append(code_snippet)
            
            logger.info(f"Generated {len(code_snippets)} {language} code snippets")
            return code_snippets
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise GenerationError(f"Failed to generate code: {e}")


class MultiModalDataStrategy(GenerationStrategy):
    """Strategy for generating multi-modal data."""
    
    parameter_model = MultiModalDataParams
    
    def generate(self) -> List[Dict[str, Any]]:
        """Generate multi-modal data.
        
        Returns:
            List[Dict[str, Any]]: Generated multi-modal data
            
        Raises:
            GenerationError: If generation fails
        """
        modalities = self.parameters["modalities"]
        count = self.parameters["count"]
        use_case = self.parameters["use_case"]
        alignment_quality = self.parameters["alignment_quality"]
        
        try:
            agent = GeneratorAgent(config=self.config)
            multimodal_data = []
            
            for i in range(count):
                prompt = f"""
                Generate a multi-modal data sample for {use_case} with {alignment_quality} alignment between modalities.
                Include the following modalities: {', '.join(modalities)}
                
                Format the output as JSON:
                {{
                    "id": "multimodal_{i+1}",
                    "use_case": "{use_case}",
                    "modalities": {modalities},
                    "alignment_quality": "{alignment_quality}",
                    {"text": "text content"," if 'text' in modalities else ""}
                    {"image_description": "detailed image description"," if 'image_description' in modalities else ""}
                    {"audio_description": "audio content description"," if 'audio_description' in modalities else ""}
                    {"video_description": "video content description"," if 'video_description' in modalities else ""}
                    {"code": "relevant code snippet"," if 'code' in modalities else ""}
                    "alignment_score": "0.0-1.0 score indicating how well modalities align",
                    "semantic_relationships": ["relationship1", "relationship2"],
                    "metadata": {{
                        "complexity": "simple/medium/complex",
                        "target_audience": "intended audience",
                        "generated_timestamp": "{time.time()}"
                    }}
                }}
                
                Ensure all modalities are semantically aligned and support the {use_case} use case.
                Return only valid JSON with no additional text.
                """
                
                logger.info(f"Generating multi-modal data #{i+1} for {use_case}")
                response = agent.generate(prompt=prompt)
                
                try:
                    # Parse JSON response
                    data_json = response.strip()
                    if data_json.find('{') > 0:
                        data_json = data_json[data_json.find('{'):]
                    if data_json.rfind('}') < len(data_json) - 1:
                        data_json = data_json[:data_json.rfind('}')+1]
                    
                    data_sample = json.loads(data_json)
                    multimodal_data.append(data_sample)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing multi-modal JSON: {e}")
                    # Create fallback data sample
                    data_sample = {
                        "id": f"multimodal_{i+1}",
                        "use_case": use_case,
                        "modalities": modalities,
                        "alignment_quality": alignment_quality,
                        "content": response,
                        "alignment_score": 0.5,
                        "semantic_relationships": ["generated", "content"],
                        "metadata": {
                            "complexity": "medium",
                            "target_audience": "general",
                            "generated_timestamp": time.time(),
                            "error": f"JSON parse error: {str(e)}"
                        }
                    }
                    multimodal_data.append(data_sample)
            
            logger.info(f"Generated {len(multimodal_data)} multi-modal data samples")
            return multimodal_data
            
        except Exception as e:
            logger.error(f"Error generating multi-modal data: {e}")
            raise GenerationError(f"Failed to generate multi-modal data: {e}") 