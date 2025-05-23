"""LLM training utilities for OpenSynthetics with QLoRA and advanced fine-tuning support."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. QLoRA features will be disabled.")

from opensynthetics.core.exceptions import TrainingError


class TrainingDataFormat(BaseModel):
    """Configuration for training data formats."""
    
    format_type: str = Field(..., description="Format type: alpaca, sharegpt, instruction, completion")
    instruction_key: str = Field("instruction", description="Key for instruction text")
    input_key: Optional[str] = Field("input", description="Key for input text (optional)")
    output_key: str = Field("output", description="Key for output/response text")
    system_message: Optional[str] = Field(None, description="System message template")
    max_length: int = Field(2048, description="Maximum sequence length")


class QLoRAConfig(BaseModel):
    """Configuration for QLoRA fine-tuning."""
    
    r: int = Field(16, description="LoRA rank", ge=1, le=256)
    alpha: int = Field(32, description="LoRA alpha parameter", ge=1)
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA adaptation"
    )
    dropout: float = Field(0.1, description="LoRA dropout", ge=0.0, le=1.0)
    bias: str = Field("none", description="LoRA bias type: none, all, lora_only")
    task_type: str = Field("CAUSAL_LM", description="Task type for PEFT")
    
    # Quantization settings
    load_in_4bit: bool = Field(True, description="Load model in 4-bit")
    load_in_8bit: bool = Field(False, description="Load model in 8-bit")
    bnb_4bit_compute_dtype: str = Field("bfloat16", description="Compute dtype for 4-bit")
    bnb_4bit_quant_type: str = Field("nf4", description="Quantization type")
    bnb_4bit_use_double_quant: bool = Field(True, description="Use double quantization")


class FineTuningConfig(BaseModel):
    """Configuration for fine-tuning parameters."""
    
    model_name: str = Field(..., description="Base model name or path")
    output_dir: str = Field(..., description="Output directory for trained model")
    
    # Training parameters
    num_train_epochs: int = Field(3, description="Number of training epochs", ge=1)
    per_device_train_batch_size: int = Field(4, description="Training batch size per device", ge=1)
    per_device_eval_batch_size: int = Field(4, description="Evaluation batch size per device", ge=1)
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps", ge=1)
    learning_rate: float = Field(2e-4, description="Learning rate", gt=0.0)
    weight_decay: float = Field(0.01, description="Weight decay", ge=0.0)
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm", gt=0.0)
    
    # Scheduler and optimizer
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler")
    warmup_ratio: float = Field(0.1, description="Warmup ratio", ge=0.0, le=1.0)
    optimizer: str = Field("adamw_torch", description="Optimizer type")
    
    # Logging and evaluation
    logging_steps: int = Field(10, description="Logging frequency", ge=1)
    eval_steps: int = Field(100, description="Evaluation frequency", ge=1)
    save_steps: int = Field(500, description="Save frequency", ge=1)
    evaluation_strategy: str = Field("steps", description="Evaluation strategy")
    save_strategy: str = Field("steps", description="Save strategy")
    
    # Advanced settings
    fp16: bool = Field(False, description="Use FP16 training")
    bf16: bool = Field(True, description="Use BF16 training")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")
    dataloader_num_workers: int = Field(4, description="Number of dataloader workers", ge=0)
    remove_unused_columns: bool = Field(False, description="Remove unused columns")
    group_by_length: bool = Field(True, description="Group sequences by length")
    
    # QLoRA settings
    use_qlora: bool = Field(True, description="Use QLoRA for parameter-efficient fine-tuning")
    qlora_config: Optional[QLoRAConfig] = Field(default_factory=QLoRAConfig, description="QLoRA configuration")
    
    # Data settings
    data_format: TrainingDataFormat = Field(default_factory=TrainingDataFormat, description="Training data format")


class ScientificDataFormatter:
    """Formatter for converting scientific paper data to training formats."""
    
    def __init__(self, format_config: TrainingDataFormat):
        """Initialize the formatter.
        
        Args:
            format_config: Configuration for data formatting
        """
        self.format_config = format_config
        
    def format_paper_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format paper segments into training examples.
        
        Args:
            segments: List of paper segments with training data
            
        Returns:
            List of formatted training examples
        """
        formatted_examples = []
        
        for segment in segments:
            try:
                if segment.get("type") == "abstract":
                    example = self._format_abstract_example(segment)
                elif segment.get("type") == "section":
                    example = self._format_section_example(segment)
                elif segment.get("type") == "qa_pair":
                    example = self._format_qa_example(segment)
                else:
                    continue
                
                if example:
                    formatted_examples.append(example)
                    
            except Exception as e:
                logger.warning(f"Failed to format segment: {e}")
                continue
        
        return formatted_examples
    
    def _format_abstract_example(self, segment: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format abstract segment into training example."""
        title = segment.get("title", "")
        content = segment.get("content", "")
        
        if not content:
            return None
        
        if self.format_config.format_type == "alpaca":
            return {
                "instruction": "Summarize the key findings and contributions of this research paper.",
                "input": f"Title: {title}",
                "output": content
            }
        elif self.format_config.format_type == "instruction":
            return {
                "instruction": f"Provide a comprehensive abstract for the research paper titled: {title}",
                "response": content
            }
        elif self.format_config.format_type == "completion":
            prompt = f"Research Paper Title: {title}\n\nAbstract: "
            return {
                "prompt": prompt,
                "completion": content
            }
        
        return None
    
    def _format_section_example(self, segment: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format section segment into training example."""
        title = segment.get("title", "")
        content = segment.get("content", "")
        metadata = segment.get("metadata", {})
        section_type = metadata.get("section_type", "content")
        paper_title = metadata.get("paper_title", "")
        
        if not content or len(content) < 100:
            return None
        
        # Truncate very long content
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        if self.format_config.format_type == "alpaca":
            instruction = f"Explain the {section_type} section of this research paper."
            input_text = f"Paper: {paper_title}\nSection: {title}"
            return {
                "instruction": instruction,
                "input": input_text,
                "output": content
            }
        elif self.format_config.format_type == "instruction":
            return {
                "instruction": f"Describe the {section_type} of the research paper '{paper_title}'",
                "response": content
            }
        
        return None
    
    def _format_qa_example(self, segment: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format Q&A segment into training example."""
        question = segment.get("question", "")
        answer = segment.get("answer", "")
        
        if not question or not answer:
            return None
        
        if self.format_config.format_type == "alpaca":
            return {
                "instruction": question,
                "input": "",
                "output": answer
            }
        elif self.format_config.format_type == "instruction":
            return {
                "instruction": question,
                "response": answer
            }
        elif self.format_config.format_type == "sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ]
            }
        
        return None
    
    def create_conversation_examples(
        self, 
        papers: List[Dict[str, Any]], 
        conversation_types: List[str] = None
    ) -> List[Dict[str, str]]:
        """Create conversation-style training examples from papers.
        
        Args:
            papers: List of paper data dictionaries
            conversation_types: Types of conversations to generate
            
        Returns:
            List of conversation examples
        """
        if conversation_types is None:
            conversation_types = ["summary", "explanation", "analysis", "comparison"]
        
        examples = []
        
        for paper in papers:
            try:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                authors = paper.get("authors", [])
                
                if not title or not abstract:
                    continue
                
                # Generate different types of conversations
                for conv_type in conversation_types:
                    example = self._generate_conversation(
                        title, abstract, authors, conv_type
                    )
                    if example:
                        examples.append(example)
                        
            except Exception as e:
                logger.warning(f"Failed to create conversation for paper: {e}")
                continue
        
        return examples
    
    def _generate_conversation(
        self, 
        title: str, 
        abstract: str, 
        authors: List[str], 
        conv_type: str
    ) -> Optional[Dict[str, str]]:
        """Generate a specific type of conversation."""
        author_str = ", ".join(authors[:3])  # First 3 authors
        
        if conv_type == "summary":
            question = f"Can you summarize the paper '{title}' by {author_str}?"
            answer = f"This paper presents {abstract}"
        elif conv_type == "explanation":
            question = f"What is the main contribution of '{title}'?"
            answer = abstract
        elif conv_type == "analysis":
            question = f"What are the key insights from this research: {title}?"
            answer = f"The key insights from this research include: {abstract}"
        else:
            return None
        
        if self.format_config.format_type == "sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ]
            }
        else:
            return {
                "instruction": question,
                "response": answer
            }


class LLMTrainer:
    """Advanced LLM trainer with QLoRA and scientific data support."""
    
    def __init__(self, config: FineTuningConfig):
        """Initialize the trainer.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.formatter = ScientificDataFormatter(config.data_format)
        self.model = None
        self.tokenizer = None
        
        if config.use_qlora and not PEFT_AVAILABLE:
            raise TrainingError("PEFT library required for QLoRA training")
    
    def prepare_model_and_tokenizer(self) -> None:
        """Prepare model and tokenizer for training."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare quantization config for QLoRA
        if self.config.use_qlora:
            bnb_config = self._create_bnb_config()
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Apply LoRA
            lora_config = self._create_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            logger.info("Model prepared with QLoRA configuration")
        else:
            # Standard full fine-tuning
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            logger.info("Model prepared for full fine-tuning")
    
    def _create_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes quantization config."""
        qlora_config = self.config.qlora_config
        
        # Map dtype strings to torch dtypes
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        
        compute_dtype = dtype_mapping.get(
            qlora_config.bnb_4bit_compute_dtype, 
            torch.bfloat16
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=qlora_config.load_in_4bit,
            load_in_8bit=qlora_config.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        )
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        qlora_config = self.config.qlora_config
        
        # Map task type string to TaskType enum
        task_type_mapping = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "QUESTION_ANS": TaskType.QUESTION_ANS
        }
        
        task_type = task_type_mapping.get(qlora_config.task_type, TaskType.CAUSAL_LM)
        
        return LoraConfig(
            r=qlora_config.r,
            lora_alpha=qlora_config.alpha,
            target_modules=qlora_config.target_modules,
            lora_dropout=qlora_config.dropout,
            bias=qlora_config.bias,
            task_type=task_type,
        )
    
    def prepare_scientific_dataset(
        self, 
        scientific_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Prepare scientific paper data for training.
        
        Args:
            scientific_data: Scientific paper dataset
            output_path: Path to save formatted dataset
            
        Returns:
            Path to formatted dataset
        """
        logger.info("Preparing scientific dataset for training")
        
        # Extract training segments
        training_segments = scientific_data.get("training_segments", [])
        papers = scientific_data.get("papers", [])
        
        # Format segments
        formatted_examples = self.formatter.format_paper_segments(training_segments)
        
        # Add conversation examples
        conversation_examples = self.formatter.create_conversation_examples(papers)
        formatted_examples.extend(conversation_examples)
        
        # Generate synthetic examples
        synthetic_examples = self._generate_synthetic_examples(papers)
        formatted_examples.extend(synthetic_examples)
        
        logger.info(f"Generated {len(formatted_examples)} training examples")
        
        # Save dataset
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "training_dataset.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training dataset saved to: {output_path}")
        return output_path
    
    def _generate_synthetic_examples(self, papers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate synthetic training examples from papers."""
        synthetic_examples = []
        
        for paper in papers:
            try:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                authors = paper.get("authors", [])
                
                if not title or not abstract:
                    continue
                
                # Generate research questions
                examples = [
                    {
                        "instruction": "Generate a research question that this paper might address.",
                        "input": f"Title: {title}",
                        "output": f"How can we {abstract.split('.')[0].lower()}?"
                    },
                    {
                        "instruction": "What methodology would be appropriate for this research?",
                        "input": f"Research topic: {title}",
                        "output": f"Based on the abstract, this research uses {abstract.split('.')[1] if '.' in abstract else 'empirical analysis'}."
                    },
                    {
                        "instruction": "Identify the key contributions of this research.",
                        "input": f"Paper: {title}",
                        "output": f"The key contributions include: {abstract}"
                    }
                ]
                
                synthetic_examples.extend(examples)
                
            except Exception as e:
                logger.warning(f"Failed to generate synthetic examples: {e}")
                continue
        
        return synthetic_examples[:100]  # Limit synthetic examples
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments for Transformers Trainer."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            optim=self.config.optimizer,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            group_by_length=self.config.group_by_length,
            report_to="none",  # Disable wandb by default
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    
    def generate_training_script(
        self, 
        dataset_path: str, 
        script_path: Optional[str] = None
    ) -> str:
        """Generate a complete training script.
        
        Args:
            dataset_path: Path to training dataset
            script_path: Path to save training script
            
        Returns:
            Path to generated training script
        """
        if script_path is None:
            script_path = os.path.join(self.config.output_dir, "train.py")
        
        script_content = f'''#!/usr/bin/env python3
"""
Generated training script for OpenSynthetics scientific LLM fine-tuning.
Generated with QLoRA: {self.config.use_qlora}
Base model: {self.config.model_name}
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
{"from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training" if self.config.use_qlora else ""}

def main():
    # Model configuration
    model_name = "{self.config.model_name}"
    output_dir = "{self.config.output_dir}"
    dataset_path = "{dataset_path}"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    {'# QLoRA Configuration' if self.config.use_qlora else '# Standard fine-tuning'}
    {self._generate_model_loading_code()}
    
    # Load and prepare dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        # Format based on data structure
        if "instruction" in examples:
            if "input" in examples and examples["input"]:
                text = f"### Instruction:\\n{{examples['instruction']}}\\n\\n### Input:\\n{{examples['input']}}\\n\\n### Response:\\n{{examples['output']}}"
            else:
                text = f"### Instruction:\\n{{examples['instruction']}}\\n\\n### Response:\\n{{examples['output']}}"
        else:
            text = examples["text"]
        
        return tokenizer(
            text,
            truncation=True,
            max_length={self.config.data_format.max_length},
            padding=False,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs={self.config.num_train_epochs},
        per_device_train_batch_size={self.config.per_device_train_batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        learning_rate={self.config.learning_rate},
        weight_decay={self.config.weight_decay},
        lr_scheduler_type="{self.config.lr_scheduler_type}",
        warmup_ratio={self.config.warmup_ratio},
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        bf16={str(self.config.bf16).lower()},
        gradient_checkpointing={str(self.config.gradient_checkpointing).lower()},
        dataloader_num_workers={self.config.dataloader_num_workers},
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed. Model saved to: {{output_dir}}")

if __name__ == "__main__":
    main()
'''

        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Training script generated: {script_path}")
        return script_path
    
    def _generate_model_loading_code(self) -> str:
        """Generate model loading code for the training script."""
        if self.config.use_qlora:
            qlora_config = self.config.qlora_config
            return f'''
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit={str(qlora_config.load_in_4bit).lower()},
        bnb_4bit_compute_dtype=torch.{qlora_config.bnb_4bit_compute_dtype},
        bnb_4bit_quant_type="{qlora_config.bnb_4bit_quant_type}",
        bnb_4bit_use_double_quant={str(qlora_config.bnb_4bit_use_double_quant).lower()},
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r={qlora_config.r},
        lora_alpha={qlora_config.alpha},
        target_modules={qlora_config.target_modules},
        lora_dropout={qlora_config.dropout},
        bias="{qlora_config.bias}",
        task_type=TaskType.{qlora_config.task_type},
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    '''
        else:
            return '''
    # Load model for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    '''
    
    def create_training_pipeline(
        self, 
        scientific_data: Dict[str, Any],
        run_training: bool = False
    ) -> Dict[str, str]:
        """Create complete training pipeline.
        
        Args:
            scientific_data: Scientific paper dataset
            run_training: Whether to run training immediately
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Creating complete training pipeline")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Prepare dataset
        dataset_path = self.prepare_scientific_dataset(scientific_data)
        
        # Generate training script
        script_path = self.generate_training_script(dataset_path)
        
        # Generate configuration file
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.dict(), f, indent=2, default=str)
        
        # Generate README
        readme_path = self._generate_readme()
        
        # Generate requirements file
        requirements_path = self._generate_requirements()
        
        result = {
            "dataset": dataset_path,
            "script": script_path,
            "config": config_path,
            "readme": readme_path,
            "requirements": requirements_path,
            "output_dir": self.config.output_dir
        }
        
        if run_training:
            logger.info("Starting training process...")
            # This would typically be run in a separate process
            # For now, we'll just log the command to run
            logger.info(f"To start training, run: python {script_path}")
        
        logger.info("Training pipeline created successfully")
        return result
    
    def _generate_readme(self) -> str:
        """Generate README file for the training setup."""
        readme_path = os.path.join(self.config.output_dir, "README.md")
        
        readme_content = f"""# Scientific LLM Fine-tuning with OpenSynthetics

## Overview

This directory contains a complete fine-tuning setup generated by OpenSynthetics for training language models on scientific literature.

## Configuration

- **Base Model**: {self.config.model_name}
- **Training Method**: {"QLoRA (Parameter-Efficient)" if self.config.use_qlora else "Full Fine-tuning"}
- **Data Format**: {self.config.data_format.format_type}
- **Training Epochs**: {self.config.num_train_epochs}
- **Batch Size**: {self.config.per_device_train_batch_size}
- **Learning Rate**: {self.config.learning_rate}

## Files

- `train.py` - Main training script
- `training_dataset.json` - Formatted training data
- `training_config.json` - Complete configuration
- `requirements.txt` - Python dependencies

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Run training:
```bash
python train.py
```

## Hardware Requirements

{'- GPU: 8GB+ VRAM (QLoRA enables training on smaller GPUs)' if self.config.use_qlora else '- GPU: 24GB+ VRAM for full fine-tuning'}
- RAM: 16GB+ system memory
- Storage: 10GB+ for model and datasets

## Model Usage

After training, load your fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
{'from peft import PeftModel' if self.config.use_qlora else ''}

tokenizer = AutoTokenizer.from_pretrained("{self.config.output_dir}")
{'base_model = AutoModelForCausalLM.from_pretrained("' + self.config.model_name + '")' if self.config.use_qlora else ''}
{'model = PeftModel.from_pretrained(base_model, "' + self.config.output_dir + '")' if self.config.use_qlora else 'model = AutoModelForCausalLM.from_pretrained("' + self.config.output_dir + '")'}

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Generated by OpenSynthetics

This training setup was automatically generated by OpenSynthetics, an advanced platform for scientific data synthesis and LLM training.
"""
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        return readme_path
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt file."""
        requirements_path = os.path.join(self.config.output_dir, "requirements.txt")
        
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
        ]
        
        if self.config.use_qlora:
            requirements.append("peft>=0.6.0")
        
        requirements.extend([
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "tqdm>=4.65.0",
            "loguru>=0.7.0",
        ])
        
        with open(requirements_path, "w") as f:
            f.write("\n".join(requirements))
        
        return requirements_path 