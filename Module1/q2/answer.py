import pandas as pd

# -----------------------------
# Read Log File
# -----------------------------

log_file = "application.log"

records = []

with open(log_file, "r") as file:
    for line in file:
        parts = line.strip().split()

        # Check for valid log entry
        if len(parts) >= 5:

            log_level = parts[2]
            module = parts[3]

            if log_level in ["ERROR", "WARNING"]:
                records.append({
                    "Module": module,
                    "Log Level": log_level
                })

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame(records)

print("\nFiltered Log Entries")
print(df)

# -----------------------------
# Group by Module
# -----------------------------

report = (
    df.groupby(["Module", "Log Level"])
      .size()
      .reset_index(name="Count")
)

print("\nError Frequency Report")
print(report)

# -----------------------------
# Save Report
# -----------------------------

report.to_csv("error_report.csv", index=False)

print("\nCSV report generated successfully.")