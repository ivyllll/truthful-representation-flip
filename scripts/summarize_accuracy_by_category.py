import pandas as pd
import os
import re

def categorize_filename(filename):
    """
    Categorizes a CSV filename based on predefined rules.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        str: The category of the dataset.
    """
    if "common_claim_true_false" in filename:
        return "Common Claim"
    if "counterfact_true_false" in filename:
        return "Counterfactual"
    if "larger_than" in filename or "smaller_than" in filename:
        return "Number Comparison"

    # Check for combined categories first (e.g., negated conjunction)
    is_negated = "neg_" in filename
    is_conjunction = "_conj" in filename
    is_disjunction = "_disj" in filename

    if is_negated:
        if is_conjunction:
            return "Negated Conjunction"
        elif is_disjunction:
            return "Negated Disjunction"
        else:
            # Extract base name for negated affirmative, e.g., "Negated - Facts"
            base_name_match = re.match(r"neg_([a-zA-Z0-9_]+?)_[A-Za-z0-9-]+_results_with_accuracy\.csv", filename)
            if base_name_match:
                base_name = base_name_match.group(1).replace('_', ' ').title()
                return f"Negated - {base_name}"
            return "Negated (Other)"
    if is_conjunction:
        # Extract base name for conjunction, e.g., "Conjunction - Facts"
        base_name_match = re.match(r"([a-zA-Z0-9_]+?)_conj_[A-Za-z0-9-]+_results_with_accuracy\.csv", filename)
        if base_name_match:
            base_name = base_name_match.group(1).replace('_', ' ').title()
            return f"Conjunction - {base_name}"
        return "Conjunction (Other)"
    if is_disjunction:
        # Extract base name for disjunction, e.g., "Disjunction - Facts"
        base_name_match = re.match(r"([a-zA-Z0-9_]+?)_disj_[A-Za-z0-9-]+_results_with_accuracy\.csv", filename)
        if base_name_match:
            base_name = base_name_match.group(1).replace('_', ' ').title()
            return f"Disjunction - {base_name}"
        return "Disjunction (Other)"

    # Affirmative categories (base names)
    # This regex tries to capture the part before the model name and standard suffixes
    affirmative_match = re.match(r"^([a-zA-Z0-9_]+?)_[A-Za-z0-9-]+_results_with_accuracy\.csv$", filename)
    if affirmative_match:
        base_name = affirmative_match.group(1).replace('_', ' ').title()
        # Ensure it's not an already categorized type that somehow missed earlier checks
        known_simple_affirmatives = ["Element Symb", "Facts", "Inventors", "Sp En Trans", "Animal Class", "Cities"]
        if base_name in known_simple_affirmatives:
            return f"Affirmative - {base_name}"
        # If it's a simple affirmative but not in the explicit list, categorize generally
        return f"Affirmative - {base_name}"


    return "Affirmative (Unknown Base)" # Fallback for affirmative if no specific base name matched


def calculate_and_summarize_category_accuracy():
    """
    Reads result CSVs, categorizes them, and calculates accuracy summaries.
    """
    results_dir = "simple_test_outputs_with_accuracy"
    output_summary_dir = "category_accuracy_summary"
    os.makedirs(output_summary_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found. Please run the main script first.")
        return

    all_data_for_summary = []

    try:
        for filename in os.listdir(results_dir):
            if filename.endswith("_results_with_accuracy.csv"):
                file_path = os.path.join(results_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        print(f"Skipping empty file: {filename}")
                        continue
                    if not {"Prompt Type", "Accuracy"}.issubset(df.columns):
                        print(f"Skipping {filename}: Missing 'Prompt Type' or 'Accuracy' column.")
                        continue

                    category = categorize_filename(filename)
                    df["Category"] = category
                    # Accuracy column is 1 for correct, 0 for incorrect. Summing it gives correct predictions.
                    # Renaming for clarity in aggregation.
                    df.rename(columns={"Accuracy": "Correct_Predictions"}, inplace=True)
                    df["Total_Statements"] = 1

                    all_data_for_summary.append(df[["Category", "Prompt Type", "Correct_Predictions", "Total_Statements"]])
                except pd.errors.EmptyDataError:
                    print(f"Skipping {filename} due to EmptyDataError.")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: Could not access the results directory '{results_dir}'.")
        return
    except Exception as e:
        print(f"An error occurred while listing files in '{results_dir}': {e}")
        return

    if not all_data_for_summary:
        print("No valid data found in any result CSVs. Cannot generate category summary.")
        return

    combined_df = pd.concat(all_data_for_summary, ignore_index=True)

    if combined_df.empty:
        print("Combined DataFrame is empty. No summary to generate.")
        return

    # Group by Category and Prompt Type to get sum of correct predictions and total statements
    category_summary = combined_df.groupby(["Category", "Prompt Type"]).agg(
        Total_Correct_Predictions=('Correct_Predictions', 'sum'),
        Total_Statements_Overall=('Total_Statements', 'sum')
    ).reset_index()

    # Calculate Average Accuracy
    category_summary["Average Accuracy (%)"] = (
        (category_summary["Total_Correct_Predictions"] / category_summary["Total_Statements_Overall"]) * 100
    ).fillna(0).round(2) # fillna(0) for cases where Total_Statements_Overall might be 0 (though unlikely here)


    print("\n===== Accuracy Summary by Category and Prompt Type =====")
    # Option 1: Print raw summary
    # print(category_summary.to_string(index=False))

    # Option 2: Pivot for better readability
    try:
        pivot_summary = category_summary.pivot_table(
            index="Category",
            columns="Prompt Type",
            values="Average Accuracy (%)",
            aggfunc='first' # 'first' because values are already aggregated
        ).reset_index()
        pivot_summary.fillna('-', inplace=True) # Replace NaN with '-' for cleaner look
        print(pivot_summary.to_string(index=False))

        # Save the pivoted summary
        summary_filename_pivot = os.path.join(output_summary_dir, "categorized_accuracy_summary_pivot.csv")
        pivot_summary.to_csv(summary_filename_pivot, index=False)
        print(f"\nPivoted category accuracy summary saved to: {summary_filename_pivot}")

    except Exception as e:
        print(f"\nCould not create pivot table: {e}. Displaying/saving raw summary.")
        print(category_summary.to_string(index=False))
        summary_filename_raw = os.path.join(output_summary_dir, "categorized_accuracy_summary_raw.csv")
        category_summary.to_csv(summary_filename_raw, index=False)
        print(f"\nRaw category accuracy summary saved to: {summary_filename_raw}")

    # You might also want to see the counts
    try:
        pivot_counts = category_summary.pivot_table(
            index="Category",
            columns="Prompt Type",
            values=["Total_Correct_Predictions", "Total_Statements_Overall"],
            aggfunc='first'
        ).reset_index()
        pivot_counts.fillna(0, inplace=True) # Replace NaN with 0 for counts
        print("\n\n===== Detailed Counts by Category and Prompt Type =====")
        print(pivot_counts.to_string())
        counts_filename = os.path.join(output_summary_dir, "categorized_accuracy_counts_pivot.csv")
        pivot_counts.to_csv(counts_filename, index=False)
        print(f"\nPivoted counts saved to: {counts_filename}")

    except Exception as e:
        print(f"\nCould not create pivot table for counts: {e}.")


if __name__ == '__main__':
    # This block allows you to run this script directly.
    # It's good practice to ensure the output directories from the main script exist
    # or create dummy ones if you want to test this analysis script standalone.

    # For testing, you might want to ensure the dummy files from the main script are present
    # or create a few representative dummy output files here.
    # For example:
    # if not os.path.exists("simple_test_outputs_with_accuracy"):
    #     os.makedirs("simple_test_outputs_with_accuracy")
    #     # Create some dummy CSVs resembling the output of the main script
    #     pd.DataFrame({
    #         "Prompt Type": ["honest", "dishonest"], "Statement": ["s1","s2"],
    #         "Original Label (0/1)": [1,0], "Expected Output": ["True", "False"],
    #         "Generated Output": ["True", "True"], "Accuracy": [1,0]
    #     }).to_csv("simple_test_outputs_with_accuracy/facts_MyModel_results_with_accuracy.csv", index=False)
    #
    #     pd.DataFrame({
    #         "Prompt Type": ["honest"], "Statement": ["s3"],
    #         "Original Label (0/1)": [1], "Expected Output": ["True"],
    #         "Generated Output": ["False"], "Accuracy": [0]
    #     }).to_csv("simple_test_outputs_with_accuracy/neg_cities_MyModel_results_with_accuracy.csv", index=False)
    #
    #     pd.DataFrame({
    #         "Prompt Type": ["neutral"], "Statement": ["s4"],
    #         "Original Label (0/1)": [0], "Expected Output": ["False"],
    #         "Generated Output": ["False"], "Accuracy": [1]
    #     }).to_csv("simple_test_outputs_with_accuracy/facts_conj_MyModel_results_with_accuracy.csv", index=False)
    #
    #     pd.DataFrame({
    #         "Prompt Type": ["neutral"], "Statement": ["s5"],
    #         "Original Label (0/1)": [1], "Expected Output": ["True"],
    #         "Generated Output": ["True"], "Accuracy": [1]
    #     }).to_csv("simple_test_outputs_with_accuracy/common_claim_true_false_MyModel_results_with_accuracy.csv", index=False)


    print("Calculating category accuracies...")
    calculate_and_summarize_category_accuracy()
    print("\nDone with category accuracy calculation.")