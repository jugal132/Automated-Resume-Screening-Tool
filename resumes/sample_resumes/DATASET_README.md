# Synthetic Resume Dataset

## 1. Dataset Purpose
This synthetic resume dataset is created exclusively for testing, benchmarking, and developing the **Automated Resume Screening Tool** portfolio project. The target role benchmark for testing this dataset is **Junior Python Developer**.

The dataset provides a diverse distribution of candidate skill sets, experience levels, educational backgrounds, and technical focuses to rigorously evaluate resume parsing, keyword extraction, TF-IDF vectorization, and cosine similarity scoring algorithms.

---

## 2. Synthetic Disclaimer & Privacy
- **100% Synthetic Data:** All resumes contained within this directory are completely fictional and synthetically generated.
- **Zero Personally Identifiable Information (PII):** Resumes are identified solely by candidate identifiers (`CAND-001` through `CAND-008`). No real human names, email addresses, phone numbers, physical residential addresses, social security numbers, or personal web links are included.
- **Fictional Entities:** All educational institutions, corporate organizations, company names, and non-profit entities referenced in these resumes are entirely fictitious (e.g., NovaByte Solutions, Stellar Webworks, Tech Valley University).
- **No Sensitive Demographics:** No demographic data such as age, gender, race, ethnicity, religion, disability status, or nationality is present in these resumes.

---

## 3. Candidate Profiles Summary

| Candidate ID | Intended Profile Archetype | Target Role Fit | Key Highlighted Skills | Experience / Background |
| :--- | :--- | :--- | :--- | :--- |
| **CAND-001** | Strong Python Developer | Strong Match | Python, SQL, Pandas, NumPy, FastAPI, Flask, Git, GitHub, PostgreSQL, REST APIs | 1+ yr internship experience, microservice & data extraction projects |
| **CAND-002** | Python Backend Developer | Medium-High Match | Python, SQL, Flask, Git, REST APIs, SQLite, basic Pandas | Backend intern experience, API routing & database migration projects |
| **CAND-003** | Data / ML-Oriented Candidate | Partial / Domain Match | Python, NumPy, Pandas, Machine Learning, Statistics, Scikit-learn, Jupyter, Matplotlib | Data science intern experience, ML prediction pipelines, limited backend |
| **CAND-004** | Beginner Web Developer | Partial / Beginner Match | Python (basics), HTML, CSS, JavaScript, Git | Web apprentice experience, frontend UI focus, basic Python scripting |
| **CAND-005** | Java / C++ Developer | Adjacent / Transferable Match | Java, C++, Data Structures, Algorithms, OOP, basic Python | Software engineering intern, systems focus, strong CS fundamentals |
| **CAND-006** | Data Analyst | Partial / Analytics Match | SQL, Excel, Power BI, Python (basics), Pandas | Junior data analyst experience, business intelligence & data cleaning |
| **CAND-007** | Academic Python Candidate | Strong Technical / Academic | Python, SQL, Pandas, Git, OOP, Unit Testing | Summa Cum Laude CS grad, teaching assistant, open-source projects |
| **CAND-008** | Limited Technical Match | Low / Weak Match | Basic programming concepts, beginner Python, basic computer skills | Administrative assistant, career-switcher, introductory scripts |

---

## 4. Intended Testing Use Cases
This dataset is designed to test:
1. **Document & Text Parsing:** Ingesting plain-text resumes with varied formatting styles, indentation, and structure.
2. **Text Cleaning & Normalization:** Handling punctuation, capitalization, abbreviations, and bullet styles.
3. **Skill & Keyword Extraction:** Accurately identifying technical tokens both from explicit skill lists and contextual project/experience descriptions.
4. **TF-IDF & Cosine Similarity Ranking:** Validating that match scores realistically separate strong relevant candidates (e.g., CAND-001, CAND-002, CAND-007) from domain-adjacent candidates (CAND-003, CAND-006, CAND-005) and low-match candidates (CAND-004, CAND-008).

---

## 5. Responsible AI & Ethical Considerations

- **Screening Assistance, Not Automated Hiring:** This software tool is designed strictly as an assistive screening aid to help recruiters organize and navigate large applicant pools. It is **never** intended to operate as an autonomous hiring decision system.
- **Evidence of Mention vs. True Competence:** Similarity scores and keyword matches reflect solely the lexical and semantic evidence documented within the submitted resume text. The absence of a keyword or skill does **not** prove that a candidate lacks capability in that area.
- **Not Predictive of Job Performance:** Match scores must not be interpreted as probabilities of future job success, employee productivity, or candidate merit. Human review by qualified hiring managers remains mandatory for every application.
- **Mitigation of Algorithmic Bias:** By decoupling candidate evaluation from demographic signals and focusing transparently on predefined job description criteria, the system aims to promote fair, consistent evaluation.
