# ai_hw_checker

A modular, AI-powered grading system that mimics human checkers, provides personalized feedback, and lays the foundation for detecting plagiarism—including AI-generated content (future feature).


### 🔍 Features

- **AI-Based Evaluation:** Uses GPT to assess and generate human-like feedback.

- **Assignment Parsing:** Supports Hebrew/English Markdown parsing using AI.

- **Student Submissions Handling:** Automatically processes structured student folders.

- **CSV Export:** Saves feedback and scores per task for each student.



### ⚙️ Optimization: Caching & Parallelism

- **Caching:** Avoids re-evaluating identical submissions. In HW1, 21 cached evaluations saved API costs.

- **Parallel Execution:** Speeds up processing. HW1 (97 students) reduced from 13 minutes → 1 minute using 6 threads.