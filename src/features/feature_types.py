"""
Feature type classification for the Student Performance Analysis project.

Every predictor was coded by the original survey as an integer, but integer coding
does NOT imply ordinality. Treating a nominal variable (e.g. transport_mode) as ordinal
(i.e. as a plain number a model can compare with <, >) silently injects a false notion
of distance/order that the model will exploit — a common and under-reported error in
papers using this dataset. Each variable below is classified by inspecting whether its
*category labels* have a real, defensible order — not by assuming the survey's numbering
means anything.

ORDINAL: true increasing (or decreasing) scale -> encoded as-is (integer) or via
         OrdinalEncoder, safe to use in distance-based or linear models.
NOMINAL: unordered categories -> must be one-hot encoded, never treated as numeric.

Borderline cases are annotated with the reasoning.
"""

ORDINAL_FEATURES = {
    "age_group": "18-21 < 22-25 < above 26 — true age ordering",
    "scholarship_type": "None < 25% < 50% < 75% < Full — monotonic coverage",
    "salary_bracket": "monotonically increasing USD ranges",
    "mother_education": "primary < secondary < high school < university < MSc < PhD",
    "father_education": "same scale as mother_education",
    "sibling_count": "1 < 2 < 3 < 4 < 5+ — true count ordering",
    "weekly_study_hours": "none < <5h < 6-10h < 11-20h < >20h",
    "reading_nonscientific": "none < sometimes < often",
    "reading_scientific": "none < sometimes < often",
    "class_attendance": "always < sometimes < never — ordinal but DECREASING frequency; "
                         "direction must be flipped or interpreted carefully in coefficients/SHAP",
    "notetaking": "never < sometimes < always",
    "listening_in_class": "never < sometimes < always",
    "discussion_benefit": "never < sometimes < always",
    "gpa_last_semester": "<2.00 < 2.00-2.49 < 2.50-2.99 < 3.00-3.49 < >3.49",
    "gpa_expected_graduation": "same scale as gpa_last_semester",
}

NOMINAL_FEATURES = {
    "sex": "binary, no order",
    "highschool_type": "private / state / other — no inherent order",
    "additional_work": "binary yes/no",
    "artistic_sports_activity": "binary yes/no",
    "has_partner": "binary yes/no",
    "transport_mode": "bus / car / bicycle / other — no inherent order",
    "accommodation_type": "rental / dormitory / family / other — no inherent order",
    "parental_status": "married / divorced / deceased — categories, not a scale",
    "mother_occupation": "job categories — no inherent order",
    "father_occupation": "job categories — no inherent order",
    "seminar_attendance": "binary yes/no",
    "project_impact": "positive / negative / neutral — coded 1/2/3 in that order, which is "
                       "NOT a monotonic scale (negative is not 'between' positive and neutral); nominal",
    "midterm_prep_companion": "alone / with friends / not applicable — no inherent order",
    "midterm_prep_timing": "closest-to-exam / regularly-during-semester / never — the numeric "
                            "coding (1,2,3) does not correspond to a quality ranking "
                            "(option 2 is arguably best practice, not the middle option); nominal",
    "flipped_classroom_attitude": "not useful / useful / not applicable — 'not applicable' "
                                   "breaks a clean ordinal scale; nominal",
}

# course_id is handled separately (see build_features.py) since whether to include it
# at all is a primary experimental condition for this project, not a routine encoding choice.
COURSE_FEATURE = "course_id"

ALL_PREDICTORS = list(ORDINAL_FEATURES) + list(NOMINAL_FEATURES) + [COURSE_FEATURE]

if __name__ == "__main__":
    print(f"Ordinal features ({len(ORDINAL_FEATURES)}):", list(ORDINAL_FEATURES))
    print(f"Nominal features ({len(NOMINAL_FEATURES)}):", list(NOMINAL_FEATURES))
    print(f"Total predictors classified: {len(ALL_PREDICTORS)} (should be 31, incl. course_id)")
