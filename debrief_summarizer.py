"""
Flying Debrief Summarizer - Python Module
==========================================

Modern NLP module for summarizing flight instructor debriefs.

Usage:
    from debrief_summarizer import DebriefSummarizer, ModelConfig

    config = ModelConfig.bart_large_cnn()
    summarizer = DebriefSummarizer(config)
    summary = summarizer.summarize(debrief_text)

Author: Updated for 2025
License: MIT
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for a summarization model"""
    name: str
    model_id: str
    max_input_length: int
    max_output_length: int
    min_output_length: int
    description: str

    @staticmethod
    def bart_large_cnn():
        """Best balance of quality and speed"""
        return ModelConfig(
            name="BART-Large-CNN",
            model_id="facebook/bart-large-cnn",
            max_input_length=1024,
            max_output_length=300,
            min_output_length=100,
            description="State-of-the-art for general summarization"
        )

    @staticmethod
    def flan_t5_large():
        """Instruction-tuned, better context understanding"""
        return ModelConfig(
            name="FLAN-T5-Large",
            model_id="google/flan-t5-large",
            max_input_length=512,
            max_output_length=300,
            min_output_length=100,
            description="Instruction-tuned model with better context understanding"
        )

    @staticmethod
    def pegasus_xsum():
        """Concise summaries"""
        return ModelConfig(
            name="Pegasus-XSum",
            model_id="google/pegasus-xsum",
            max_input_length=512,
            max_output_length=300,
            min_output_length=100,
            description="Updated Pegasus model for concise summaries"
        )


@dataclass
class GenerationParams:
    """Parameters for text generation"""
    length_penalty: float = 2.0
    num_beams: int = 4
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    temperature: float = 1.0

    @staticmethod
    def balanced():
        """Standard balanced configuration"""
        return GenerationParams(
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    @staticmethod
    def concise():
        """For shorter, more concise summaries"""
        return GenerationParams(
            length_penalty=4.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    @staticmethod
    def detailed():
        """For longer, more detailed summaries"""
        return GenerationParams(
            length_penalty=1.0,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    @staticmethod
    def high_quality():
        """Best quality (slower)"""
        return GenerationParams(
            length_penalty=2.0,
            num_beams=8,
            early_stopping=True,
            no_repeat_ngram_size=3
        )


class DebriefSummarizer:
    """Modern summarization engine for flying debriefs"""

    def __init__(
        self,
        model_config: ModelConfig,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize the summarizer.

        Args:
            model_config: ModelConfig instance specifying the model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_fp16: Use FP16 precision on GPU for 2x speedup
        """
        self.config = model_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading {self.config.name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ {self.config.name} loaded successfully on {self.device}")
        if self.use_fp16:
            print(f"  Using FP16 precision for faster inference")

    def summarize(
        self,
        text: str,
        params: Optional[GenerationParams] = None,
        add_instruction: bool = False
    ) -> str:
        """
        Generate summary with given parameters.

        Args:
            text: Input text to summarize
            params: Generation parameters (defaults to balanced)
            add_instruction: Add instruction prefix for T5 models

        Returns:
            Generated summary text
        """
        if params is None:
            params = GenerationParams.balanced()

        # Add instruction prefix for instruction-tuned models
        if add_instruction and "flan" in self.config.model_id.lower():
            text = (
                f"Summarize this flight instructor debrief, "
                f"highlighting key points and action items: {text}"
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_output_length,
                min_length=self.config.min_output_length,
                length_penalty=params.length_penalty,
                num_beams=params.num_beams,
                early_stopping=params.early_stopping,
                no_repeat_ngram_size=params.no_repeat_ngram_size,
                temperature=params.temperature,
            )

        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def batch_summarize(
        self,
        texts: List[str],
        params: Optional[GenerationParams] = None,
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate summaries for multiple texts.

        Args:
            texts: List of input texts
            params: Generation parameters
            batch_size: Batch size for processing

        Returns:
            List of generated summaries
        """
        if params is None:
            params = GenerationParams.balanced()

        summaries = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                max_length=self.config.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_output_length,
                    min_length=self.config.min_output_length,
                    length_penalty=params.length_penalty,
                    num_beams=params.num_beams,
                    early_stopping=params.early_stopping,
                    no_repeat_ngram_size=params.no_repeat_ngram_size,
                )

            batch_summaries = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            summaries.extend(batch_summaries)

        return summaries

    def cleanup(self):
        """Free up GPU memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
        print(f"✓ {self.config.name} cleaned up")


class EvaluationMetrics:
    """Comprehensive evaluation metrics for summaries"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True
        )

    def calculate_rouge(self, reference: str, summary: str) -> Dict:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, summary)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeLsum_f': scores['rougeLsum'].fmeasure,
        }

    def calculate_bertscore(self, reference: str, summary: str) -> Dict:
        """Calculate BERTScore (semantic similarity)"""
        P, R, F1 = bert_score(
            [summary],
            [reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False
        )
        return {
            'bertscore_precision': P.item(),
            'bertscore_recall': R.item(),
            'bertscore_f1': F1.item(),
        }

    def calculate_aviation_metrics(self, reference: str, summary: str) -> Dict:
        """Custom metrics for aviation debriefs"""

        # Key aviation terms to check coverage
        aviation_terms = [
            'takeoff', 'take off', 'landing', 'stall', 'climb',
            'circuit', 'approach', 'departure', 'taxi',
            'speed', 'altitude', 'height', 'checks'
        ]

        ref_lower = reference.lower()
        sum_lower = summary.lower()

        # Calculate term coverage
        terms_in_ref = [term for term in aviation_terms if term in ref_lower]
        terms_in_summary = [term for term in terms_in_ref if term in sum_lower]
        term_coverage = (
            len(terms_in_summary) / len(terms_in_ref) if terms_in_ref else 0
        )

        # Compression ratio
        compression_ratio = len(summary) / len(reference)

        return {
            'aviation_term_coverage': term_coverage,
            'compression_ratio': compression_ratio,
            'summary_length': len(summary),
            'summary_word_count': len(summary.split()),
        }

    def evaluate(self, reference: str, summary: str) -> Dict:
        """Complete evaluation"""
        metrics = {}
        metrics.update(self.calculate_rouge(reference, summary))
        metrics.update(self.calculate_bertscore(reference, summary))
        metrics.update(self.calculate_aviation_metrics(reference, summary))
        return metrics


def quick_summarize(
    debrief_text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 300,
    min_length: int = 100
) -> str:
    """
    Quick one-liner for production use.

    Args:
        debrief_text: The flight debrief to summarize
        model_name: Model to use (default: BART-Large-CNN)
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens

    Returns:
        Summary string

    Example:
        >>> summary = quick_summarize("Good sortie today...")
        >>> print(summary)
    """
    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device
    )

    result = summarizer(
        debrief_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
        length_penalty=2.0
    )

    return result[0]['summary_text']


# Utility functions for aviation-specific analysis
def extract_flight_phases(text: str) -> Dict[str, bool]:
    """Check which flight phases are mentioned"""
    text_lower = text.lower()
    phases = {
        'startup_taxi': any(term in text_lower for term in ['startup', 'taxi']),
        'takeoff': any(term in text_lower for term in ['takeoff', 'take off', 'departure']),
        'climb': 'climb' in text_lower,
        'maneuvers': any(term in text_lower for term in ['stall', 'loop', 'maneuver', 'manoeuvre']),
        'approach': 'approach' in text_lower,
        'circuit': 'circuit' in text_lower,
        'landing': any(term in text_lower for term in ['landing', 'flare']),
    }
    return phases


def extract_action_items(text: str) -> List[str]:
    """Extract potential action items from debrief"""
    action_keywords = [
        'watch', 'remember', 'careful', 'keep',
        'make sure', 'be mindful', 'don\'t forget'
    ]

    sentences = text.replace('!', '.').replace('?', '.').split('.')
    action_items = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and any(keyword in sentence.lower() for keyword in action_keywords):
            action_items.append(sentence)

    return action_items


def extract_safety_items(text: str) -> List[str]:
    """Extract safety-critical information"""
    safety_keywords = [
        'safety', 'critical', 'dangerous', 'risk',
        'oxygen', 'altitude', 'stall', 'buffet',
        'gear', 'flaps', 'warning', 'caution'
    ]

    sentences = text.replace('!', '.').replace('?', '.').split('.')
    safety_items = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and any(keyword in sentence.lower() for keyword in safety_keywords):
            safety_items.append(sentence)

    return safety_items


if __name__ == "__main__":
    # Example usage
    print("Flying Debrief Summarizer - Example Usage\n" + "="*60)

    sample_debrief = """Good sortie today. A few points to work on but overall
    a good performance. Take off and departure was all nice. On climb out just
    watch your speed and make sure you are always scanning it. The clean stall
    is fairly simple. Circuit work is coming along nicely. Remember to flare
    on landings."""

    print("\nOriginal Debrief:")
    print(sample_debrief)

    print("\n\nGenerating summary...")
    summary = quick_summarize(sample_debrief)

    print("\nSummary:")
    print(summary)

    print("\n\nFlight Phases:")
    phases = extract_flight_phases(sample_debrief)
    for phase, present in phases.items():
        print(f"  {phase}: {'✓' if present else '✗'}")

    print("\n\nAction Items:")
    actions = extract_action_items(sample_debrief)
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action}")
