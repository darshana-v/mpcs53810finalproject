import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_students = 4000
num_colleges = 100
college_capacity = 20
applications_per_student = 10

# 1. Generate public scores
student_scores = np.random.uniform(0, 1, num_students)
college_scores = np.random.uniform(0, 1, num_colleges)

# 2. Generate private preferences and evaluations
epsilon = np.random.uniform(0, 1, (num_students, num_colleges))  # student private preferences
delta = np.random.uniform(0, 1, (num_colleges, num_students))    # college private evaluations

# 3. Compute student preferences: P_ij = S_j + epsilon_ij
P = college_scores + epsilon  # shape: (num_students, num_colleges)

# 4. Compute college evaluations: Q_ji = c_i + delta_ji
Q = student_scores.reshape(1, -1) + delta  # shape: (num_colleges, num_students)

# 5. Students apply to top-10 colleges based on P_ij
applications = np.argsort(-P, axis=1)[:, :applications_per_student]  # shape: (num_students, 10)

# 6. Colleges probabilistically accept applicants
acceptance_matrix = np.zeros((num_students, num_colleges), dtype=bool)

for student_id in range(num_students):
    for rank in range(applications_per_student):
        college_id = applications[student_id, rank]
        prob_accept = np.exp(student_scores[student_id] - college_scores[college_id])
        if np.random.rand() < prob_accept:
            acceptance_matrix[student_id, college_id] = True

# 7. For each college, randomly select up to 20 students from accepted pool
final_matches = np.full(num_students, -1)  # -1 indicates unmatched

for college_id in range(num_colleges):
    accepted_students = np.where(acceptance_matrix[:, college_id])[0]
    if len(accepted_students) > college_capacity:
        selected_students = np.random.choice(accepted_students, college_capacity, replace=False)
    else:
        selected_students = accepted_students
    for student_id in selected_students:
        if final_matches[student_id] == -1:
            final_matches[student_id] = college_id

# 8. Compute utilities
utilities = np.array([P[i, final_matches[i]] if final_matches[i] != -1 else 0 for i in range(num_students)])
matched = final_matches != -1
match_rate = np.mean(matched)
average_utility = np.mean(utilities[matched])

simulation_df = pd.DataFrame({
    'Student ID': np.arange(num_students),
    'Matched College': final_matches,
    'Utility': utilities,
    'Matched': matched
})

# Save to CSV
csv_path = "simulation_summary.csv"
simulation_df.to_csv(csv_path, index=False)
print(match_rate, average_utility)
