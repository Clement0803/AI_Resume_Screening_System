# 🚀 AI Resume Screening System Project

An intelligent resume optimization system powered by Claude AI that helps job seekers improve their resumes, check ATS compatibility, and generate personalized cover letters. **Now with structured job description input!**

### 🎯 Structured Job Description Input
- **Job Overview**: Paste or describe the role summary
- **Key Responsibilities**: List main duties and tasks
- **Requirements**: Required qualifications and skills
- **Preferred Qualifications**: Nice-to-have skills and experience

### 📊 Enhanced Features
- **4-Tab Interface**: Organized workflow (Input → ATS → Optimize → Cover Letter)
- **Position Details**: Optional company name and job title for personalization
- **Preview & Edit**: See formatted JD before analysis
- **Dual Input Mode**: Choose structured input or paste full JD
- **Editable Outputs**: Edit cover letters before downloading
- **Better UX**: Progress indicators, better messaging, and visual feedback

## 🎯 Core Features

### 1. 📊 ATS Score Analysis
- Comprehensive ATS compatibility score (0-100)
- Detailed keyword match analysis with percentages
- Skills gap identification (required vs. present)
- Experience alignment assessment
- Resume strengths highlighting
- Critical areas for improvement
- Formatting quality score (0-10)

### 2. ✨ Resume Optimization
- **Priority Actions**: Top 3-5 critical changes
- **Section-by-Section Guidance**:
  - Professional summary optimization
  - Experience bullet point rewrites
  - Skills section updates
  - Additional sections recommendations
- **Keyword Integration**: Natural incorporation strategies
- **Quantification Tips**: Adding metrics and numbers
- **ATS-Friendly Formatting**: Structure and layout tips

### 3. 📝 Cover Letter Generation
- Personalized 300-400 word cover letters
- Company-specific customization (when provided)
- Achievement highlighting from resume
- Natural, professional tone
- Editable before download
- Multiple format options

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Anthropic API key ([Get one here](https://console.anthropic.com))

### Setup Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd ai-resume-screening_system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the models** (first time only):
```bash
python train.py
```
*Note: You'll need a `data/Resume.csv` dataset. See "Model Training" section below.*

4. **Get your Anthropic API key:**
- Sign up at [https://console.anthropic.com](https://console.anthropic.com)
- Navigate to "API Keys" section
- Create a new key
- Copy and save it securely

## 🚀 Usage

### Quick Start (3 Steps)

1. **Start the app:**
```bash
streamlit run app.py
```

2. **Configure in sidebar:**
   - Enter your Anthropic API key
   - See features and tips

3. **Use the 4-tab workflow:**

#### Tab 1: 📝 Input
- Upload your resume (PDF)
- Add company name and position title (optional)
- Choose input method:
  - **Structured Input** (Recommended): Fill separate fields for overview, responsibilities, requirements, and preferred qualifications
  - **Full Job Description**: Paste complete JD as-is
- Preview formatted job description

#### Tab 2: 📊 ATS Analysis
- Click "Analyze ATS Score"
- View quick metrics (match score, category, status)
- Read detailed AI analysis with:
  - Overall ATS score
  - Keyword matches and gaps
  - Skills analysis
  - Experience alignment
  - Strengths and improvements
  - Formatting score
- Download analysis report

#### Tab 3: ✨ Resume Optimizer
- Click "Optimize My Resume"
- Get comprehensive suggestions:
  - Priority actions (must-do changes)
  - Section-by-section rewrites
  - Keyword integration strategies
  - Quantification examples
  - Formatting tips
- Download optimization guide
- Update resume and re-run analysis!

#### Tab 4: 💌 Cover Letter
- Click "Generate Cover Letter"
- Review AI-generated letter
- Edit directly in the text area
- Download as TXT file
- Regenerate if needed

## 📁 Project Structure

```
ai-resume-copilot/
├── app.py                  # Main application 
├── train.py                # training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── .env.example           # Environment variable template
├── data/
│   └── Resume.csv         # Training dataset
└── model/
    ├── tfidf_vectorizer.pkl    # TF-IDF model
    └── category_model.pkl      # Category classifier
```

## 🎨 Using Structured Input (Recommended)

The structured input method provides better analysis by clearly separating different aspects of the job description:

### Example: Data Reconciliation Programmer Role

**Job Overview:**
```
We are seeking a Data Reconciliation Programmer to support automated extraction 
and reconciliation of vendor data. This role focuses on building reliable Python 
scripts to collect, normalize, and validate data from multiple vendor back-office systems.
```

**Key Responsibilities:**
```
• Develop Python scripts to extract data from vendor back-office systems
• Automate data collection, parsing, and normalization for reconciliation purposes
• Implement validation checks to identify missing, duplicated, or mismatched data
• Support finance reporting by providing structured and accurate datasets
• Maintain documentation and version control for scripts
• Troubleshoot and enhance script stability and performance
• Collaborate closely with Finance Data Analyst on reconciliation frameworks
```

**Requirements:**
```
• Diploma or Bachelor's degree in IT, Computer Science, or related fields
• Strong academic foundation in Python programming
• Logical and structured approach to data handling
• Ability to follow defined frameworks and documentation
```

**Preferred Qualifications:**
```
• Experience with APIs, web scraping, or ETL concepts
• Familiarity with SQL or structured data formats (CSV, JSON)
• Interest in finance data and reconciliation logic
```

### Benefits of Structured Input:
✅ **Better keyword extraction** - AI can identify critical terms in each category
✅ **Focused analysis** - Separate evaluation of responsibilities vs. requirements
✅ **Clearer optimization** - Section-specific suggestions
✅ **More accurate matching** - Weighted scoring based on importance

## 🔑 API Key Setup

### Getting Your API Key:
1. Visit [Anthropic Console](https://console.anthropic.com)
2. Sign up or log in with your account
3. Go to "API Keys" in the dashboard
4. Click "Create Key"
5. Name your key (e.g., "Resume Copilot")
6. Copy the key immediately (you won't see it again!)

### Security Best Practices:
- ⚠️ **Never commit API keys to version control**
- Use the sidebar input (most convenient)
- Or use environment variables (more secure)
- Keep `.env` file in `.gitignore`
- Rotate keys periodically
- Use separate keys for development and production

## 📊 How It Works

### Technical Architecture

1. **Resume Processing Pipeline:**
   ```
   PDF Upload → Text Extraction → Text Cleaning → TF-IDF Vectorization → Similarity Score
   ```

2. **ATS Analysis:**
   ```
   Resume + Job Description → Claude AI Analysis → Detailed Report
   ```

3. **Optimization:**
   ```
   ATS Analysis → Claude AI Processing → Actionable Recommendations
   ```

4. **Cover Letter Generation:**
   ```
   Resume + JD + Details → Claude AI → Personalized Letter
   ```

### ML Models Used:
- **TF-IDF Vectorizer**: Converts text to numerical features (5000 features, unigrams + bigrams)
- **Logistic Regression**: Classifies resumes by job category (multinomial, L-BFGS solver)
- **Claude 4.5 Sonnet**: Provides intelligent analysis and generation

## 💡 Tips for Best Results

### Resume Tips:
✅ Use standard section headings (Education, Experience, Skills)
✅ Include keywords from job description naturally
✅ Use bullet points for achievements
✅ Quantify results (e.g., "Increased efficiency by 25%")
✅ Avoid tables, images, or complex formatting
✅ Keep to 1-2 pages
✅ Use a clean, professional font

### Job Description Input:
✅ Fill all sections for comprehensive analysis
✅ Be specific about requirements vs. preferred qualifications
✅ Include both technical and soft skills
✅ Mention key tools, technologies, and frameworks
✅ Add context about the role and team

### Using the Tool:
✅ Run analysis multiple times as you improve
✅ Focus on top 3-5 missing keywords first
✅ Add concrete examples for each skill
✅ Use action verbs (Led, Developed, Managed, etc.)
✅ Always review and personalize AI outputs

## 🔧 Model Training

### Dataset Format
Your `data/Resume.csv` should have:
- **Resume** column: Full resume text
- **Category** column: Job category label

Example categories:
- Data Science
- Web Developer
- Database Administrator
- Python Developer
- Java Developer
- etc.

### Training Process:
```bash
python train.py
```

This will:
1. Load and validate dataset
2. Clean resume text
3. Train TF-IDF vectorizer (5000 features)
4. Train Logistic Regression classifier
5. Evaluate model performance
6. Save models to `model/` directory

### Expected Performance:
- Accuracy: ~85-95% (depends on dataset quality)
- Training time: 1-5 minutes (depends on dataset size)

## 📈 Understanding Your Results

### ATS Score Breakdown:
| Score Range | Status | Action Needed |
|------------|--------|---------------|
| 90-100% | 🟢 Excellent | Minor tweaks only |
| 70-89% | 🟢 Good | Some optimization recommended |
| 50-69% | 🟡 Fair | Significant changes needed |
| Below 50% | 🔴 Poor | Major overhaul required |

### What to Focus On:
1. **Critical Keywords**: Terms from "Requirements" section
2. **Skills**: Both technical and soft skills
3. **Experience**: Relevant years and responsibilities
4. **Format**: ATS-friendly structure
5. **Quantification**: Numbers, percentages, metrics

## 🐛 Troubleshooting

### Common Issues:

**"Model files not found"**
```bash
python train.py
```

**"API Key Error"**
- Check for typos
- Ensure no extra spaces
- Verify key is active in Anthropic console
- Try regenerating the key

**"PDF Extraction Error"**
- Try different PDF export
- Ensure PDF isn't password-protected
- Check PDF contains selectable text (not scanned image)

**"Rate Limit Error"**
- Wait a few minutes
- Check your API usage in Anthropic console
- Consider upgrading your plan

**App won't start:**
```bash
streamlit --version  # Check installation
pip install --upgrade streamlit  # Update if needed
```

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Claude AI](https://www.anthropic.com/claude) (Anthropic)
- ML models using [Scikit-learn](https://scikit-learn.org/)
- PDF processing with [PyPDF2](https://pypdf2.readthedocs.io/)

## 📊 Changelog

### v2.0 (Latest)
- ✨ Added structured job description input
- 📝 4-tab interface for better workflow
- 🎯 Position details (company, title)
- 👁️ Preview and edit features
- ⚡ Improved UX with progress indicators
- 📥 Download options for all outputs
- 💡 Better tips and guidance


---

Made with ❤️ to help job seekers succeed in their career journey

**Star ⭐ this repo if it helped you land an interview!**
