"""Data loading utilities for the Student Performance Analysis project."""
import pandas as pd

RAW_PATH = "data/raw/student_performance_raw.csv"

COLUMN_NAMES = {
    "1": "age_group", "2": "sex", "3": "highschool_type", "4": "scholarship_type",
    "5": "additional_work", "6": "artistic_sports_activity", "7": "has_partner",
    "8": "salary_bracket", "9": "transport_mode", "10": "accommodation_type",
    "11": "mother_education", "12": "father_education", "13": "sibling_count",
    "14": "parental_status", "15": "mother_occupation", "16": "father_occupation",
    "17": "weekly_study_hours", "18": "reading_nonscientific", "19": "reading_scientific",
    "20": "seminar_attendance", "21": "project_impact", "22": "class_attendance",
    "23": "midterm_prep_companion", "24": "midterm_prep_timing", "25": "notetaking",
    "26": "listening_in_class", "27": "discussion_benefit", "28": "flipped_classroom_attitude",
    "29": "gpa_last_semester", "30": "gpa_expected_graduation", "COURSE ID": "course_id",
    "GRADE": "grade",
}

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset and apply readable column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"STUDENT ID": "student_id", **COLUMN_NAMES})
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df.columns.tolist())
