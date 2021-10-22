import pandas as pd

#######################
# Load and clean data
#######################
ssc_df = pd.read_csv("data/fixed.csv")

# select all columns which are not nootropics ratings
person_features_columns = ssc_df.columns[0:6]

side_effects_columns = ["SideEffectsofLSDMicrodosingverysmalldosewithouthallucinogeniceff",
                        "SideEffectsofAdderall",
                        "HowwereyouusingAdderall",
                        "PatternofAdderallUse",
                        "DoseofAdderallhighestdoseyouusedconsistentlyifyoutookmorethanonc",
                        "ResultsofAdderalluse",
                        "SideEffectsofPhenibut",
                        "PatternofPhenibutUse",
                        "Doseofphenibuthighestdoseyouusedconsistentlyifyoutookmorethanonc",
                        "ResultsofPhenibutuse"]

other_features = ssc_df.columns[52:70]

df_clean = ssc_df.drop(columns=list(person_features_columns) + list(side_effects_columns) + list(other_features))
# transform into a 3 columns (user_id, item_id, rating) matrix for Surprise
df_clean = df_clean.rename_axis('userID').reset_index()
df_clean = df_clean.melt(id_vars="userID", var_name="itemID", value_name="rating")
# remove rows when there is no rating
df_clean = df_clean[df_clean["rating"] != " "]
df_clean = df_clean.reset_index(drop=True)

df_clean.to_csv("dataset_clean.csv", index=False)