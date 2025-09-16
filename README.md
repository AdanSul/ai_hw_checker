# ai_hw_checker


AI Homework Checker is a **generic, extensible system** designed to automatically evaluate academic homework submissions.  
The project started with **code assignments (Python)** but was built in a way that can easily extend to other domains.

**Current Status:**

✅ Assignment parsing and task extraction.

✅ Student submission evaluation.

✅ Validation and result export (JSON/CSV).

⏳ AI plagiarism detection – coming soon!

---

## 🎬 Demo

**Here’s a quick demo video of the system in action:**

<!-- README.md -->
<video controls width="800" muted playsinline>
  <source src="demo/demo.mp4" type="video/mp4" />
    If it doesn’t load, download it here:
  <a href="demo/demo.mp4">Download video</a>
</video>

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/ai_hw_checker.git
cd ai_hw_checker
pip install -r requirements.txt
```


## 🚀 How to Run

You can run the system either from the command line or through the UI.

**🔹Command Line:**
```bash
python evaluate.py
```

**🔹UI Mode:**

Backend:
```bash
uvicorn backend.main:app --reload
```

Frontend:
```bash
cd frontend
python -m http.server 9000
```


## 🤝 Contributing

Pull requests and feature suggestions are welcome.
If you’d like to collaborate on expanding plagiarism detection or UI, feel free to open an issue.