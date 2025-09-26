This project gathers information about DPA decisions on the GDPR for comparative analysis and statistical analysis. We are at the very first step of prepping data.
Current structure:
/raw-data contains a CSV file with all decisions and their machine translations, pooled from MD files in subfolder. Each decision is given an ID.
/analyzed-decisions contains AI-processed decisions where each one has 68 fields extracted using AI. Prompt given to AI is an important file explaining the values, and is stored as .md in the folder.
Master file is CSV; each answer is delimited with newline.
Foe convenience, same responses (raw, uncleaned) are also included as JSON.
Therefore, master-analyzed-data-unclean.csv is the most vital file in the project.
We must assume additional rows may be added over time, so all our processing must be documented and reproducible.
/resources contains various resources and utilities - for instance, a list of ISIC abbreviations, codebooks, et.

/scripts is where we will save processing scripts, each with its own readme. NB: data-cleaning.md outlines our current idea of a plan and the scripts under /scripts have produced outputs in /outputs. THESE ARE JUST TESTS AND SUGGESTED APPROACHES, and must be scrutinized for efficiency and effectiveness, as well as best daaa science practices and academic rigor.