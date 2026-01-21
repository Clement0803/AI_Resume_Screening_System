import streamlit as st
import pickle
import re
import string
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import os

# ------------------------
# Text Cleaning
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------------
# PDF Reader
# ------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ------------------------
# Format Job Description
# ------------------------
def format_job_description(overview, responsibilities, requirements, preferred):
    """Combine structured JD components into full description"""
    jd_parts = []
    
    # Overview is optional - only add if provided
    if overview.strip():
        jd_parts.append(f"**Job Overview:**\n{overview}")
    
    # Responsibilities and Requirements are primary - always include if provided
    if responsibilities.strip():
        jd_parts.append(f"**Key Responsibilities:**\n{responsibilities}")
    
    if requirements.strip():
        jd_parts.append(f"**Requirements:**\n{requirements}")
    
    # Preferred is optional
    if preferred.strip():
        jd_parts.append(f"**Preferred Qualifications:**\n{preferred}")
    
    # If nothing provided, return empty string
    if not jd_parts:
        return ""
    
    return "\n\n".join(jd_parts)

# ------------------------
# ATS Analysis with Claude
# ------------------------
def analyze_ats_score(resume_text, job_description, similarity_score):
    """Get detailed ATS analysis using Claude AI"""
    try:
        client = anthropic.Anthropic(api_key=st.session_state.get('api_key', ''))
        
        prompt = f"""You are an expert ATS (Applicant Tracking System) analyzer. Analyze this resume against the job description and provide a detailed ATS score breakdown.

**Job Description:**
{job_description}

**Resume:**
{resume_text}

**Initial Similarity Score:** {similarity_score:.2f}%

IMPORTANT: Focus primarily on matching Key Responsibilities and Requirements from the job description. These are the most critical sections.

Please provide a comprehensive analysis with the following sections:

1. **Overall ATS Score (0-100)**: Provide a weighted score considering:
   - Keyword matching from Responsibilities and Requirements (50%)
   - Skills alignment with Requirements (25%)
   - Experience relevance to Responsibilities (15%)
   - Formatting quality (10%)

2. **Key Responsibilities Match Analysis**:
   - Which responsibilities from the JD the candidate can fulfill based on their resume
   - Matching rate (e.g., "6 out of 7 responsibilities addressed")
   - Examples of how resume demonstrates relevant experience

3. **Requirements Match Analysis**:
   - ✅ Required qualifications present in resume
   - ❌ Required qualifications missing from resume
   - Education/degree match
   - Critical technical skills present/missing

4. **Skills Gap Analysis**:
   - Required skills found in resume
   - Required skills missing from resume
   - Recommended skills to add

5. **Resume Strengths**:
   - What the resume does exceptionally well
   - Standout qualifications that exceed requirements

6. **Critical Gaps** (Must Address):
   - Missing requirements that could disqualify the candidate
   - Missing responsibilities-related experience
   - Skills that MUST be added

7. **Formatting Score (0-10)**: 
   - Structure and readability
   - ATS-friendly elements
   - Professional presentation

Format your response clearly with these exact section headers."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        return f"Error analyzing ATS score: {str(e)}"

# ------------------------
# Resume Optimization
# ------------------------
def optimize_resume(resume_text, job_description, ats_analysis):
    """Generate optimized resume suggestions using Claude AI"""
    try:
        client = anthropic.Anthropic(api_key=st.session_state.get('api_key', ''))
        
        prompt = f"""You are an expert resume writer and career coach. Based on the ATS analysis, provide specific, actionable recommendations to optimize this resume.

**Job Description:**
{job_description}

**Current Resume:**
{resume_text}

**ATS Analysis:**
{ats_analysis}

Please provide detailed optimization suggestions organized as follows:

1. **Priority Actions** (Must-Do):
   - Top 3-5 critical changes needed immediately
   - Each with specific before/after examples

2. **Professional Summary Optimization**:
   - Rewrite the summary to include key missing keywords
   - Align with job requirements
   - Keep it 3-4 sentences

3. **Experience Section Enhancements**:
   - Rewrite 2-3 bullet points to include missing keywords naturally
   - Add quantifiable achievements (use metrics, numbers, percentages)
   - Use strong action verbs
   - Show specific examples

4. **Skills Section Updates**:
   - Technical skills to add/emphasize
   - Soft skills to highlight
   - Certifications or tools to mention
   - Format: Categorize by skill type

5. **Keyword Integration Strategy**:
   - Show exactly where and how to incorporate missing keywords
   - Provide natural phrasing examples
   - Avoid keyword stuffing

6. **Additional Sections to Consider**:
   - Projects section (if applicable)
   - Certifications section
   - Volunteer work (if relevant)

7. **Formatting Recommendations**:
   - ATS-friendly structure tips
   - Section ordering
   - Font and spacing guidelines

Make all suggestions concrete and actionable. Provide specific text examples wherever possible."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        return f"Error optimizing resume: {str(e)}"

# ------------------------
# Cover Letter Generation
# ------------------------
def generate_cover_letter(resume_text, job_description, company_name="", position_title=""):
    """Generate a tailored cover letter using Claude AI"""
    try:
        client = anthropic.Anthropic(api_key=st.session_state.get('api_key', ''))
        
        company_info = f"for {company_name}" if company_name else ""
        position_info = f"for the {position_title} position" if position_title else ""
        
        prompt = f"""You are an expert cover letter writer. Create a compelling, personalized cover letter {company_info} {position_info}.

**Job Description:**
{job_description}

**Candidate's Resume:**
{resume_text}

Please write a professional cover letter that:

**Structure:**
1. **Opening Paragraph** (Hook):
   - Express genuine enthusiasm for the role
   - Mention how you learned about the position (if company name provided)
   - Include a compelling statement about why you're an excellent fit

2. **Body Paragraph 1** (Key Achievement):
   - Highlight your most relevant achievement from the resume
   - Connect it directly to a key responsibility in the job description
   - Include specific metrics or results

3. **Body Paragraph 2** (Skills & Fit):
   - Demonstrate 2-3 key skills that match the requirements
   - Show understanding of the company/role (if company name provided)
   - Explain why this role aligns with your career goals

4. **Closing Paragraph** (Call to Action):
   - Reiterate enthusiasm
   - Express desire for an interview
   - Thank them for consideration
   - Professional sign-off

**Guidelines:**
- Length: 300-400 words (3-4 paragraphs)
- Tone: Professional yet personable
- Avoid clichés like "I am writing to apply"
- Show personality and genuine interest
- Make it specific to this role, not generic
- Include [Your Name] as placeholder for signature

Format it as a complete, ready-to-use cover letter."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"

# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        clf = pickle.load(open("model/category_model.pkl", "rb"))
        return tfidf, clf
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run `python train_improved.py` first.")
        return None, None

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Resume Copilot", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚀 AI Resume Copilot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent assistant for resume optimization, ATS scoring, and cover letter generation</p>', unsafe_allow_html=True)

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key for AI features")
    if api_key:
        st.session_state['api_key'] = api_key
        st.success("✓ API Key configured")
    else:
        st.warning("⚠️ Enter API key to use AI features")
    
    st.divider()
    
    st.markdown("### 🎯 Features")
    st.markdown("✅ **ATS Score Analysis**\nDetailed compatibility scoring")
    st.markdown("✅ **Resume Optimization**\nAI-powered suggestions")
    st.markdown("✅ **Cover Letter Generator**\nPersonalized cover letters")
    
    st.divider()
    
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. Upload your resume
    2. Fill in job details
    3. Get AI-powered insights
    4. Improve and apply!
    """)
    
    st.divider()
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use clean PDF resumes
    - Fill all JD sections
    - Review AI suggestions
    - Customize outputs
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📝 Input", "📊 ATS Analysis", "✨ Resume Optimizer", "💌 Cover Letter"])

# ------------------------
# TAB 1: INPUT
# ------------------------
with tab1:
    st.header("📝 Job Application Information")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Your Resume")
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF format)", 
            type=["pdf"],
            help="Upload your resume in PDF format for best results"
        )
        
        if uploaded_file:
            st.success(f"✓ Uploaded: {uploaded_file.name}")
            
            # Preview resume text
            if st.checkbox("Preview resume text"):
                with st.spinner("Extracting text..."):
                    resume_text = extract_text_from_pdf(uploaded_file)
                    st.session_state['resume_text'] = resume_text
                    st.text_area("Resume Preview", resume_text[:1000] + "...", height=200)
    
    with col2:
        st.subheader("🎯 Position Details")
        company_name = st.text_input(
            "Company Name (Optional)", 
            placeholder="e.g., Google, Microsoft"
        )
        position_title = st.text_input(
            "Position Title (Optional)", 
            placeholder="e.g., Data Reconciliation Programmer"
        )
        
        st.session_state['company_name'] = company_name
        st.session_state['position_title'] = position_title
    
    st.divider()
    
    st.subheader("📋 Job Description Details")
    st.info("💡 **Tip**: Focus on Key Responsibilities and Requirements for best results. Job Overview is optional.")
    
    # Option to choose input method
    input_method = st.radio(
        "Choose input method:",
        ["📝 Structured Input (Recommended)", "📄 Full Job Description"],
        horizontal=True
    )
    
    if input_method == "📝 Structured Input (Recommended)":
        # Primary sections (most important)
        st.markdown("### 🎯 Primary Sections (Required)")
        col1, col2 = st.columns(2)
        
        with col1:
            responsibilities = st.text_area(
                "Key Responsibilities *",
                placeholder="• Develop Python scripts...\n• Automate data collection...\n• Implement validation checks...",
                height=250,
                help="List the main responsibilities - this is critical for matching!"
            )
        
        with col2:
            requirements = st.text_area(
                "Requirements *",
                placeholder="• Bachelor's degree in IT...\n• Strong Python programming...\n• Logical approach to data...",
                height=250,
                help="List the required qualifications and skills - this is critical for matching!"
            )
        
        # Secondary sections (optional)
        st.markdown("### 📝 Additional Sections (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            overview = st.text_area(
                "Job Overview (Optional)",
                placeholder="Brief description of the role and what you'll be doing...",
                height=150,
                help="Optional: Paste the job overview or write a summary of the role"
            )
            
        with col2:
            preferred = st.text_area(
                "Preferred Qualifications (Optional)",
                placeholder="• Experience with APIs...\n• Familiarity with SQL...\n• Finance domain knowledge...",
                height=150,
                help="Optional: List preferred but not required qualifications"
            )
        
        # Combine into full job description
        if responsibilities.strip() or requirements.strip():
            job_description = format_job_description(overview, responsibilities, requirements, preferred)
            st.session_state['job_description'] = job_description
            st.session_state['jd_components'] = {
                'overview': overview,
                'responsibilities': responsibilities,
                'requirements': requirements,
                'preferred': preferred
            }
            
            # Show what's been filled
            filled_sections = []
            if overview.strip():
                filled_sections.append("Overview")
            if responsibilities.strip():
                filled_sections.append("Responsibilities")
            if requirements.strip():
                filled_sections.append("Requirements")
            if preferred.strip():
                filled_sections.append("Preferred")
            
            if filled_sections:
                st.success(f"✓ Sections filled: {', '.join(filled_sections)}")
        else:
            st.session_state['job_description'] = ""
            st.warning("⚠️ Please fill at least Key Responsibilities or Requirements")
    else:
        job_description = st.text_area(
            "Full Job Description",
            placeholder="Paste the complete job description here...",
            height=400,
            help="Paste the entire job description as-is"
        )
        st.session_state['job_description'] = job_description
        st.session_state['jd_components'] = None
    
    # Preview formatted JD
    if st.session_state.get('job_description', '').strip():
        with st.expander("👁️ Preview Formatted Job Description"):
            st.markdown(st.session_state['job_description'])
    
    # Validation
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if uploaded_file:
            st.success("✅ Resume uploaded")
        else:
            st.error("❌ Resume required")
    
    with col2:
        jd_filled = st.session_state.get('job_description', '').strip()
        if input_method == "📝 Structured Input (Recommended)":
            components = st.session_state.get('jd_components', {})
            has_required = (components.get('responsibilities', '').strip() or 
                          components.get('requirements', '').strip())
            if has_required:
                st.success("✅ Key sections provided")
            else:
                st.error("❌ Responsibilities or Requirements needed")
        else:
            if jd_filled:
                st.success("✅ Job description provided")
            else:
                st.error("❌ Job description required")
    
    with col3:
        if st.session_state.get('api_key', ''):
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ API key needed")

# ------------------------
# TAB 2: ATS Analysis
# ------------------------
with tab2:
    st.header("📊 ATS Score Analysis")
    st.write("Comprehensive analysis of how well your resume matches the job description through ATS systems")
    
    if st.button("🔍 Analyze ATS Score", type="primary", use_container_width=True):
        if not uploaded_file or not st.session_state.get('job_description', '').strip():
            st.error("⚠️ Please upload a resume and provide job description in the Input tab")
        elif not st.session_state.get('api_key'):
            st.error("🔑 Please enter your Anthropic API key in the sidebar")
        else:
            with st.spinner("🔄 Analyzing your resume... This may take 15-30 seconds"):
                # Load models
                tfidf, clf = load_models()
                
                if tfidf is None or clf is None:
                    st.stop()
                
                # Extract and process resume
                if 'resume_text' not in st.session_state:
                    resume_text = extract_text_from_pdf(uploaded_file)
                    st.session_state['resume_text'] = resume_text
                else:
                    resume_text = st.session_state['resume_text']
                
                clean_resume = clean_text(resume_text)
                clean_jd = clean_text(st.session_state['job_description'])
                
                # Calculate similarity
                resume_vector = tfidf.transform([clean_resume])
                jd_vector = tfidf.transform([clean_jd])
                similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
                predicted_category = clf.predict(resume_vector)[0]
                
                # Display basic metrics
                st.subheader("📈 Quick Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Initial Match Score", 
                        f"{similarity * 100:.1f}%",
                        help="TF-IDF similarity between resume and job description"
                    )
                
                with col2:
                    st.metric(
                        "Predicted Category", 
                        predicted_category,
                        help="ML model prediction of job category"
                    )
                
                with col3:
                    if similarity > 0.6:
                        status = "🟢 Strong Match"
                        delta = "Good"
                    elif similarity > 0.4:
                        status = "🟡 Fair Match"
                        delta = "Needs Work"
                    else:
                        status = "🔴 Weak Match"
                        delta = "Major Gaps"
                    st.metric("Match Status", status, delta)
                
                st.divider()
                
                # Get detailed ATS analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🤖 Getting AI-powered detailed analysis...")
                progress_bar.progress(50)
                
                ats_analysis = analyze_ats_score(
                    resume_text, 
                    st.session_state['job_description'], 
                    similarity * 100
                )
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.session_state['ats_analysis'] = ats_analysis
                
                # Display analysis
                st.subheader("🔍 Detailed ATS Analysis")
                st.markdown(ats_analysis)
                
                # Download button
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=f"ATS ANALYSIS REPORT\n{'='*50}\n\n{ats_analysis}",
                    file_name="ats_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ------------------------
# TAB 3: Resume Optimizer
# ------------------------
with tab3:
    st.header("✨ Resume Optimization")
    st.write("Get AI-powered, actionable suggestions to improve your resume based on the job description")
    
    if st.button("🚀 Optimize My Resume", type="primary", use_container_width=True):
        if not uploaded_file or not st.session_state.get('job_description', '').strip():
            st.error("⚠️ Please upload a resume and provide job description in the Input tab")
        elif not st.session_state.get('api_key'):
            st.error("🔑 Please enter your Anthropic API key in the sidebar")
        else:
            with st.spinner("🔄 Analyzing and optimizing your resume... This may take 30-45 seconds"):
                # Check if we already have ATS analysis
                if 'ats_analysis' not in st.session_state or 'resume_text' not in st.session_state:
                    # Need to run ATS analysis first
                    tfidf, clf = load_models()
                    
                    if tfidf is None or clf is None:
                        st.stop()
                    
                    resume_text = extract_text_from_pdf(uploaded_file)
                    clean_resume = clean_text(resume_text)
                    clean_jd = clean_text(st.session_state['job_description'])
                    
                    resume_vector = tfidf.transform([clean_resume])
                    jd_vector = tfidf.transform([clean_jd])
                    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
                    
                    st.info("📊 Running ATS analysis first...")
                    ats_analysis = analyze_ats_score(resume_text, st.session_state['job_description'], similarity * 100)
                    st.session_state['ats_analysis'] = ats_analysis
                    st.session_state['resume_text'] = resume_text
                
                # Get optimization suggestions
                st.info("💡 Generating optimization suggestions...")
                optimization = optimize_resume(
                    st.session_state['resume_text'],
                    st.session_state['job_description'],
                    st.session_state['ats_analysis']
                )
                
                st.session_state['optimization'] = optimization
                
                st.success("✅ Optimization complete!")
                st.divider()
                
                st.subheader("💡 Personalized Optimization Suggestions")
                st.markdown(optimization)
                
                # Download button
                st.download_button(
                    label="📥 Download Optimization Guide",
                    data=f"RESUME OPTIMIZATION GUIDE\n{'='*50}\n\n{optimization}",
                    file_name="resume_optimization_guide.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.info("💡 **Next Steps**: Review the suggestions above, update your resume, then re-run the ATS analysis to see your improved score!")

# ------------------------
# TAB 4: Cover Letter Generator
# ------------------------
with tab4:
    st.header("💌 Cover Letter Generator")
    st.write("Generate a personalized, professional cover letter based on your resume and the job description")
    
    # Show position details if available
    if st.session_state.get('company_name') or st.session_state.get('position_title'):
        st.info(f"**Position**: {st.session_state.get('position_title', 'Not specified')} at {st.session_state.get('company_name', 'Not specified')}")
    
    if st.button("✍️ Generate Cover Letter", type="primary", use_container_width=True):
        if not uploaded_file or not st.session_state.get('job_description', '').strip():
            st.error("⚠️ Please upload a resume and provide job description in the Input tab")
        elif not st.session_state.get('api_key'):
            st.error("🔑 Please enter your Anthropic API key in the sidebar")
        else:
            with st.spinner("🔄 Crafting your personalized cover letter... This may take 20-30 seconds"):
                if 'resume_text' not in st.session_state:
                    resume_text = extract_text_from_pdf(uploaded_file)
                    st.session_state['resume_text'] = resume_text
                
                cover_letter = generate_cover_letter(
                    st.session_state['resume_text'],
                    st.session_state['job_description'],
                    st.session_state.get('company_name', ''),
                    st.session_state.get('position_title', '')
                )
                
                st.session_state['cover_letter'] = cover_letter
                
                st.success("✅ Cover letter generated successfully!")
                st.divider()
                
                # Display cover letter
                st.subheader("📝 Your Personalized Cover Letter")
                
                # Editable text area
                edited_letter = st.text_area(
                    "Review and edit your cover letter:",
                    value=cover_letter,
                    height=500,
                    help="Feel free to edit and personalize further"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download as TXT",
                        data=edited_letter,
                        file_name=f"cover_letter_{st.session_state.get('company_name', 'job').replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                
                st.divider()
                
                st.success("💡 **Pro Tip**: Always review and personalize the cover letter before sending. Add specific details about why you're excited about this particular company and role!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🚀 AI Resume Copilot v2.0</strong> | Powered by Claude AI</p>
    <p style='font-size: 0.9em;'>Helping you land your dream job with AI-powered insights</p>
    <p style='font-size: 0.8em; margin-top: 10px;'>
        💡 <strong>Tips</strong>: Focus on Key Responsibilities & Requirements | Review all AI suggestions | Customize outputs to your voice
    </p>
</div>
""", unsafe_allow_html=True)