"""
Example Usage Script for Flying Debrief Summarizer
===================================================

This script demonstrates how to use the debrief summarizer in your own code.
"""

from debrief_summarizer import (
    DebriefSummarizer,
    ModelConfig,
    GenerationParams,
    EvaluationMetrics,
    quick_summarize,
    extract_flight_phases,
    extract_action_items
)


def example_1_quick_usage():
    """Example 1: Quick one-line summarization"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick Usage")
    print("="*80)

    debrief = """Good sortie today. A few points to work on but overall a good
    performance. Take off and departure was all nice. On climb out just watch
    your speed and make sure you are always scanning it. The clean stall is
    fairly simple. Circuit work is coming along nicely. Remember to flare on
    landings."""

    # One-line summarization
    summary = quick_summarize(debrief)

    print("\nOriginal:", debrief)
    print("\nSummary:", summary)


def example_2_custom_model():
    """Example 2: Using custom model configuration"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Model Configuration")
    print("="*80)

    debrief = """Good sortie today. A few points to work on but overall a good
    performance. Take off and departure was all nice. Circuit work is coming
    along nicely."""

    # Use BART model
    config = ModelConfig.bart_large_cnn()
    summarizer = DebriefSummarizer(config)

    # Generate with balanced parameters
    params = GenerationParams.balanced()
    summary = summarizer.summarize(debrief, params)

    print("\nModel:", config.name)
    print("Summary:", summary)

    # Cleanup
    summarizer.cleanup()


def example_3_parameter_comparison():
    """Example 3: Compare different parameter configurations"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Parameter Comparison")
    print("="*80)

    debrief = """Good sortie today. A few points to work on but overall a good
    performance. Take off and departure was all nice. On climb out just watch
    your speed. Circuit work is coming along nicely. Remember to flare on
    landings."""

    config = ModelConfig.bart_large_cnn()
    summarizer = DebriefSummarizer(config)

    param_configs = [
        ("Balanced", GenerationParams.balanced()),
        ("Concise", GenerationParams.concise()),
        ("Detailed", GenerationParams.detailed()),
    ]

    for name, params in param_configs:
        summary = summarizer.summarize(debrief, params)
        print(f"\n{name} ({len(summary)} chars):")
        print(f"  {summary}")

    summarizer.cleanup()


def example_4_evaluation():
    """Example 4: Evaluate summary quality"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Evaluation Metrics")
    print("="*80)

    original = """Good sortie today. A few points to work on but overall a good
    performance. Take off and departure was all nice. On climb out just watch
    your speed. Circuit work is coming along nicely."""

    summary = quick_summarize(original)

    # Evaluate
    evaluator = EvaluationMetrics()
    metrics = evaluator.evaluate(original, summary)

    print("\nOriginal:", original)
    print("\nSummary:", summary)
    print("\nMetrics:")
    print(f"  ROUGE-1 F-score: {metrics['rouge1_f']:.3f}")
    print(f"  ROUGE-L F-score: {metrics['rougeL_f']:.3f}")
    print(f"  BERTScore F1: {metrics['bertscore_f1']:.3f}")
    print(f"  Aviation Coverage: {metrics['aviation_term_coverage']:.1%}")
    print(f"  Compression: {metrics['compression_ratio']:.1%}")


def example_5_aviation_analysis():
    """Example 5: Aviation-specific analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Aviation Analysis")
    print("="*80)

    debrief = """Good sortie today. A few points to work on but overall a good
    performance. Let's start with startup, taxi and take off. All ok there, just
    watch your speed on taxing. Take off and departure was all nice. On climb out
    just watch your speed and make sure you are always scanning it. Remember to
    flare on landings. Circuit work is coming along nicely."""

    summary = quick_summarize(debrief)

    # Extract flight phases
    original_phases = extract_flight_phases(debrief)
    summary_phases = extract_flight_phases(summary)

    print("\nFlight Phases Coverage:")
    for phase, in_original in original_phases.items():
        in_summary = summary_phases[phase]
        status = "✓" if in_original and in_summary else "✗" if in_original else "-"
        print(f"  {phase:20s}: {status}")

    # Extract action items
    action_items = extract_action_items(debrief)
    print(f"\nAction Items Found: {len(action_items)}")
    for i, action in enumerate(action_items, 1):
        print(f"  {i}. {action}")


def example_6_batch_processing():
    """Example 6: Batch processing multiple debriefs"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Processing")
    print("="*80)

    debriefs = [
        "Good sortie today. Take off was excellent. Watch your speed in the climb.",
        "Overall satisfactory performance. Landing needs work. Remember to flare.",
        "Strong performance. Stall recovery was well executed. Keep up the good work.",
    ]

    config = ModelConfig.bart_large_cnn()
    summarizer = DebriefSummarizer(config)

    # Batch summarize
    summaries = summarizer.batch_summarize(debriefs, batch_size=4)

    print(f"\nProcessed {len(debriefs)} debriefs:")
    for i, (original, summary) in enumerate(zip(debriefs, summaries), 1):
        print(f"\n{i}. Original: {original}")
        print(f"   Summary: {summary}")

    summarizer.cleanup()


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("FLYING DEBRIEF SUMMARIZER - USAGE EXAMPLES")
    print("="*80)

    examples = [
        example_1_quick_usage,
        example_2_custom_model,
        example_3_parameter_comparison,
        example_4_evaluation,
        example_5_aviation_analysis,
        example_6_batch_processing,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
