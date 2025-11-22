#!/usr/bin/env python3
"""
Evaluation script to parse LLM judge responses and compare against human annotations.
Computes Exact Match (EM) accuracy and RMSE metrics.

Usage:
    python eval.py --model <model_name> --dataset <dataset_name>
    python eval.py --model Claude-3-haiku --dataset bc5cdr_test
    python eval.py --model all --dataset all  # Evaluate all models and datasets
"""

import argparse
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def extract_json_from_markdown(text):
    """
    Extract JSON from markdown code blocks.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Extracted JSON string or original text
    """
    # Try to extract from ```json ... ```
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to extract from ``` ... ```
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text


def extract_json_object(text):
    """
    Extract JSON object from text that may contain additional content.

    Args:
        text: Text that may contain a JSON object

    Returns:
        Extracted JSON string or original text
    """
    # Find JSON object pattern { ... }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        # Return the first match that looks like it has our keys
        for match in matches:
            if 'predicted' in match.lower() or 'relations' in match.lower():
                return match
        # If no match has our keywords, return the first one
        return matches[0]

    return text


def repair_json(text):
    """
    Attempt to repair common JSON formatting issues.

    Args:
        text: Potentially malformed JSON string

    Returns:
        Repaired JSON string
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove trailing commas before closing braces
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Fix missing commas between key-value pairs
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)

    # Fix unquoted keys (simple case)
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

    # Remove any text before the first { or [
    first_brace = text.find('{')
    first_bracket = text.find('[')
    if first_brace != -1:
        if first_bracket == -1 or first_brace < first_bracket:
            text = text[first_brace:]
        elif first_bracket != -1:
            text = text[first_bracket:]

    # Remove any text after the last } or ]
    last_brace = text.rfind('}')
    last_bracket = text.rfind(']')
    if last_brace != -1:
        if last_bracket == -1 or last_brace > last_bracket:
            text = text[:last_brace+1]
        elif last_bracket != -1:
            text = text[:last_bracket+1]

    return text


def extract_with_regex(text):
    """
    Extract numeric values directly using regex patterns.

    Args:
        text: Text containing the response

    Returns:
        Tuple of (correct_predicted, total_predicted) or (None, None)
    """
    # Pattern 1: Look for "correctly_predicted_relations": VALUE
    correct_patterns = [
        r'"correctly_predicted_relations"\s*:\s*(\d+)',
        r'"correctly_predicted"\s*:\s*(\d+)',
        r'correctly_predicted_relations["\s:]+(\d+)',
        r'correct[ly]*\s*predicted[^:]*:\s*(\d+)',
    ]

    total_patterns = [
        r'"total_predicted_relations"\s*:\s*(\d+)',
        r'"total_predicted"\s*:\s*(\d+)',
        r'total_predicted_relations["\s:]+(\d+)',
        r'total\s*predicted[^:]*:\s*(\d+)',
    ]

    correct_value = None
    total_value = None

    # Try to find correct_predicted
    for pattern in correct_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            correct_value = int(match.group(1))
            break

    # Try to find total_predicted
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            total_value = int(match.group(1))
            break

    return correct_value, total_value


def extract_values_from_json(data):
    """
    Extract correct and total values from JSON data (dict or list).
    Handles cases where values are numbers or arrays (counts the array length).

    Args:
        data: Parsed JSON data

    Returns:
        Tuple of (correct, total) or (None, None)
    """
    # Handle dictionary
    if isinstance(data, dict):
        correct = data.get('correctly_predicted_relations', data.get('correct_predicted', None))
        total = data.get('total_predicted_relations', data.get('total_predicted', None))

        # If the values are lists/arrays, count their length
        if isinstance(correct, list):
            correct = len(correct)
        if isinstance(total, list):
            total = len(total)

        return correct, total

    # Handle list - check if first element is a dict
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        correct = data[0].get('correctly_predicted_relations', data[0].get('correct_predicted', None))
        total = data[0].get('total_predicted_relations', data[0].get('total_predicted', None))

        # If the values are lists/arrays, count their length
        if isinstance(correct, list):
            correct = len(correct)
        if isinstance(total, list):
            total = len(total)

        return correct, total

    return None, None


def parse_json_response(response):
    """
    Parse LLM response to extract correctly_predicted_relations and total_predicted_relations.
    Uses multiple strategies to handle various formats - identical logic to parse_llm_responses.py

    Args:
        response: String containing JSON response

    Returns:
        Tuple (correctly_predicted, total_predicted) - returns (0, 0) if parsing fails
    """
    if pd.isna(response) or not response:
        return 0, 0

    response_text = str(response)

    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response_text)
        correct, total = extract_values_from_json(data)

        if correct is not None and total is not None:
            return int(correct), int(total)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # Strategy 2: Extract from markdown code blocks
    extracted = extract_json_from_markdown(response_text)
    try:
        data = json.loads(extracted)
        correct, total = extract_values_from_json(data)

        if correct is not None and total is not None:
            return int(correct), int(total)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # Strategy 3: Extract JSON object from text
    json_obj = extract_json_object(response_text)
    try:
        data = json.loads(json_obj)
        correct, total = extract_values_from_json(data)

        if correct is not None and total is not None:
            return int(correct), int(total)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # Strategy 4: Repair JSON and try parsing
    repaired = repair_json(response_text)
    try:
        data = json.loads(repaired)
        correct, total = extract_values_from_json(data)

        if correct is not None and total is not None:
            return int(correct), int(total)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass

    # Strategy 5: Regex extraction
    correct, total = extract_with_regex(response_text)
    if correct is not None and total is not None:
        return correct, total

    # Strategy 6: All strategies failed - return (0, 0)
    return 0, 0


def compute_rmse(predictions, actuals):
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values

    Returns:
        RMSE value
    """
    predictions = np.array(predictions, dtype=float)
    actuals = np.array(actuals, dtype=float)

    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))

    if not np.any(valid_mask):
        return np.nan

    mse = np.mean((predictions[valid_mask] - actuals[valid_mask]) ** 2)
    return np.sqrt(mse)


def compute_avg_rmse(correct_pred, correct_actual, total_pred, total_actual):
    """
    Compute average RMSE for both correct_predicted and total_predicted.
    Formula: sqrt( (1/2) * mean((correct_pred - correct_actual)^2 + (total_pred - total_actual)^2) )

    Args:
        correct_pred: Array of predicted correct values
        correct_actual: Array of actual correct values
        total_pred: Array of predicted total values
        total_actual: Array of actual total values

    Returns:
        Average RMSE value
    """
    # Convert to numpy arrays and handle NaN values
    correct_pred = np.array(correct_pred, dtype=float)
    correct_actual = np.array(correct_actual, dtype=float)
    total_pred = np.array(total_pred, dtype=float)
    total_actual = np.array(total_actual, dtype=float)

    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(correct_pred) | np.isnan(correct_actual) |
                   np.isnan(total_pred) | np.isnan(total_actual))

    if not np.any(valid_mask):
        return np.nan

    # Use only valid values
    correct_diff_sq = (correct_pred[valid_mask] - correct_actual[valid_mask]) ** 2
    total_diff_sq = (total_pred[valid_mask] - total_actual[valid_mask]) ** 2
    combined_mse = np.mean(correct_diff_sq + total_diff_sq) / 2
    return np.sqrt(combined_mse)


def evaluate_model_dataset(llm_file, human_file):
    """
    Evaluate a single model on a single dataset.

    Args:
        llm_file: Path to LLM judge file with responses
        human_file: Path to human annotation file

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Read files
        llm_df = pd.read_csv(llm_file)
        human_df = pd.read_csv(human_file)

        # Check required columns
        if 'response' not in llm_df.columns:
            print(f"Error: 'response' column not found in {llm_file}")
            return None

        required_human_cols = ['correct_predicted', 'total_predicted']
        if not all(col in human_df.columns for col in required_human_cols):
            print(f"Error: Required columns not found in {human_file}")
            return None

        # Ensure same number of rows
        if len(llm_df) != len(human_df):
            print(f"Warning: Row count mismatch - LLM: {len(llm_df)}, Human: {len(human_df)}")
            min_len = min(len(llm_df), len(human_df))
            llm_df = llm_df.iloc[:min_len]
            human_df = human_df.iloc[:min_len]

        # Parse responses and extract values (returns (0, 0) if parsing fails)
        llm_correct = []
        llm_total = []
        parsing_successes = 0
        parsing_failures = 0

        for idx, row in llm_df.iterrows():
            # Try to parse - returns actual values or (0, 0) if fails
            correct, total = parse_json_response(row['response'])
            llm_correct.append(correct)
            llm_total.append(total)

            # Check if parsing was successful (non-zero values or explicit check)
            # We need to verify against the original response to know if it was a real parse
            try:
                json.loads(str(row['response']))
                parsing_successes += 1
            except:
                try:
                    json_pattern = r'\{[^{}]*"correctly_predicted_relations"[^{}]*"total_predicted_relations"[^{}]*\}'
                    matches = re.findall(json_pattern, str(row['response']), re.IGNORECASE | re.DOTALL)
                    if matches and json.loads(matches[0]):
                        parsing_successes += 1
                    else:
                        parsing_failures += 1
                except:
                    parsing_failures += 1

        llm_correct = np.array(llm_correct, dtype=float)
        llm_total = np.array(llm_total, dtype=float)
        human_correct = human_df['correct_predicted'].values.astype(float)
        human_total = human_df['total_predicted'].values.astype(float)

        # Count exact matches (now including parsing failures which are (0, 0))
        correct_matches = np.sum(llm_correct == human_correct)
        total_matches = np.sum(llm_total == human_total)
        em_matches = np.sum((llm_correct == human_correct) & (llm_total == human_total))

        # Calculate accuracy (over total rows, including parsing failures as (0, 0))
        total_rows = len(llm_df)
        parsing_success_rate = (parsing_successes / total_rows * 100) if total_rows > 0 else 0

        correct_accuracy = (correct_matches / total_rows * 100) if total_rows > 0 else 0
        total_accuracy = (total_matches / total_rows * 100) if total_rows > 0 else 0
        em = (em_matches / total_rows * 100) if total_rows > 0 else 0

        # Calculate RMSE (including parsing failures as (0, 0))
        correct_rmse = compute_rmse(llm_correct, human_correct)
        total_rmse = compute_rmse(llm_total, human_total)
        avg_rmse = compute_avg_rmse(llm_correct, human_correct, llm_total, human_total)

        return {
            'total_rows': total_rows,
            'parsing_successes': parsing_successes,
            'parsing_failures': parsing_failures,
            'parsing_success_rate': parsing_success_rate,
            'correct_matches': correct_matches,
            'total_matches': total_matches,
            'em_matches': em_matches,
            'correct_accuracy': correct_accuracy,
            'total_accuracy': total_accuracy,
            'EM': em,
            'correct_rmse': correct_rmse,
            'total_rmse': total_rmse,
            'avg_rmse': avg_rmse,
        }

    except Exception as e:
        print(f"Error evaluating files: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_available_models(llm_base):
    """Get list of available models in the structured folder."""
    structured_path = Path(llm_base) / "structured"
    if not structured_path.exists():
        return []
    return [d.name for d in structured_path.iterdir() if d.is_dir()]


def get_available_datasets(llm_base, model):
    """Get list of available datasets for a given model."""
    model_path = Path(llm_base) / "structured" / model
    if not model_path.exists():
        return []
    return [f.stem for f in model_path.glob("*.csv")]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge performance against human annotations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name (e.g., Claude-3-haiku) or 'all' for all models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (e.g., bc5cdr_test) or 'all' for all datasets"
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="structured",
        choices=["structured", "unstructured"],
        help="Approach type (default: structured)"
    )

    args = parser.parse_args()

    # Base paths
    base_path = Path(__file__).parent
    llm_base = base_path / "llm_judge_data"
    human_base = base_path / "human_reference_data"

    print("=" * 100)
    print("LLM JUDGE EVALUATION - Exact Match (EM) and RMSE Metrics")
    print("=" * 100)

    # Determine which models to evaluate
    if args.model == "all":
        models = get_available_models(llm_base)
    else:
        models = [args.model]

    if not models:
        print(f"Error: No models found in {llm_base / args.approach}")
        return

    print(f"\nApproach: {args.approach}")
    print(f"Models to evaluate: {', '.join(models)}")

    # Store all results
    all_results = []
    results_by_model = defaultdict(list)

    # Evaluate each model
    for model in sorted(models):
        llm_model_path = llm_base / args.approach / model

        if not llm_model_path.exists():
            print(f"\nWarning: Model path does not exist: {llm_model_path}")
            continue

        # Determine which datasets to evaluate
        if args.dataset == "all":
            datasets = get_available_datasets(llm_base, model)
        else:
            datasets = [args.dataset]

        for dataset in sorted(datasets):
            llm_file = llm_model_path / f"{dataset}.csv"
            human_file = human_base / args.approach / f"{dataset}.csv"

            if not llm_file.exists():
                print(f"\nWarning: LLM file not found: {llm_file}")
                continue

            if not human_file.exists():
                print(f"\nWarning: Human annotation file not found: {human_file}")
                continue

            print(f"\nEvaluating: {model} on {dataset}")
            result = evaluate_model_dataset(llm_file, human_file)

            if result:
                result['model'] = model
                result['dataset'] = dataset
                result['approach'] = args.approach
                all_results.append(result)
                results_by_model[model].append(result)

    if not all_results:
        print("\nNo results to display.")
        return

    # Display results
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    for model in sorted(results_by_model.keys()):
        results = results_by_model[model]

        print(f"\n{model}")
        print("-" * 100)
        print(f"{'Dataset':<20} {'Rows':<8} {'Parsed':<8} {'EM':<12} {'Correct Acc':<12} "
              f"{'Total Acc':<12} {'Avg RMSE':<12}")
        print("-" * 100)

        for r in sorted(results, key=lambda x: x['dataset']):
            parsed_display = f"{r['parsing_successes']}/{r['total_rows']}"
            print(f"{r['dataset']:<20} {r['total_rows']:<8} {parsed_display:<8} "
                  f"{r['EM']:>6.2f}%{'':<5} {r['correct_accuracy']:>6.2f}%{'':<5} "
                  f"{r['total_accuracy']:>6.2f}%{'':<5} {r['avg_rmse']:>8.4f}")

        # Model summary
        if len(results) > 1:
            total_rows = sum(r['total_rows'] for r in results)
            total_em_matches = sum(r['em_matches'] for r in results)
            total_correct_matches = sum(r['correct_matches'] for r in results)
            total_total_matches = sum(r['total_matches'] for r in results)

            avg_em = (total_em_matches / total_rows * 100) if total_rows > 0 else 0
            avg_correct_acc = (total_correct_matches / total_rows * 100) if total_rows > 0 else 0
            avg_total_acc = (total_total_matches / total_rows * 100) if total_rows > 0 else 0

            # Compute overall RMSE for this model
            all_correct_pred = []
            all_correct_actual = []
            all_total_pred = []
            all_total_actual = []

            for r in results:
                llm_file = llm_base / args.approach / model / f"{r['dataset']}.csv"
                human_file = human_base / args.approach / f"{r['dataset']}.csv"

                llm_df = pd.read_csv(llm_file)
                human_df = pd.read_csv(human_file)

                for idx, row in llm_df.iterrows():
                    correct, total = parse_json_response(row['response'])
                    if correct is not None and total is not None:
                        all_correct_pred.append(correct)
                        all_total_pred.append(total)
                        all_correct_actual.append(human_df['correct_predicted'].iloc[idx])
                        all_total_actual.append(human_df['total_predicted'].iloc[idx])

            overall_avg_rmse = compute_avg_rmse(all_correct_pred, all_correct_actual,
                                                all_total_pred, all_total_actual)

            print("-" * 100)
            print(f"{'AVERAGE':<20} {total_rows:<8} {'':<8} "
                  f"{avg_em:>6.2f}%{'':<5} {avg_correct_acc:>6.2f}%{'':<5} "
                  f"{avg_total_acc:>6.2f}%{'':<5} {overall_avg_rmse:>8.4f}")

    # Overall summary across all models
    if len(results_by_model) > 1:
        print("\n" + "=" * 100)
        print("OVERALL SUMMARY (All Models)")
        print("=" * 100)

        total_rows = sum(r['total_rows'] for r in all_results)
        total_parsing_successes = sum(r['parsing_successes'] for r in all_results)
        total_parsing_failures = sum(r['parsing_failures'] for r in all_results)
        total_em_matches = sum(r['em_matches'] for r in all_results)
        total_correct_matches = sum(r['correct_matches'] for r in all_results)
        total_total_matches = sum(r['total_matches'] for r in all_results)

        overall_parsing_rate = (total_parsing_successes / total_rows * 100) if total_rows > 0 else 0
        overall_em = (total_em_matches / total_rows * 100) if total_rows > 0 else 0
        overall_correct_acc = (total_correct_matches / total_rows * 100) if total_rows > 0 else 0
        overall_total_acc = (total_total_matches / total_rows * 100) if total_rows > 0 else 0

        print(f"\nTotal evaluations: {len(all_results)}")
        print(f"Total rows: {total_rows:,}")
        print(f"\nParsing Statistics:")
        print(f"  Successfully parsed: {total_parsing_successes:,} / {total_rows:,} ({overall_parsing_rate:.2f}%)")
        print(f"  Failed to parse: {total_parsing_failures:,} / {total_rows:,} ({100 - overall_parsing_rate:.2f}%)")
        print(f"\nExact Match (EM): {total_em_matches:,} / {total_rows:,} ({overall_em:.2f}%)")
        print(f"Correct Accuracy: {total_correct_matches:,} / {total_rows:,} ({overall_correct_acc:.2f}%)")
        print(f"Total Accuracy: {total_total_matches:,} / {total_rows:,} ({overall_total_acc:.2f}%)")

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = base_path / f"eval_results_{args.approach}.csv"
    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 100)
    print(f"✓ Detailed results saved to: {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()