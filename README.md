Reads CV and job description in pdf format and outputs a json stating matching and missing skills, as well as a matching score of the candidate using RAG. The skills are grouped in required tech stack, optional tech stack, soft skills, domain knowledge and an overall score.

Provide a Groq API key by creating a file .env with the entry GROQ_API_TOKEN=[YOUR_KEY]

Example output:
{
    "tech_stack_required": {
        "matched_skills": [
            "Python",
            "PyTorch",
            "Containerization",
            "Model Deployment",
        ],
        "missing_skills": [
            "TensorFlow",
            "Cloud technologies",
        ],
        "match_percent": 66.7
    },
    "tech_stack_optional": {
        "matched_skills": [
            "VLM"
        ],
        "missing_skills": [
            "LLM",
            "NLP"
        ],
        "match_percent": 33.3
    },
    "soft_skills": {
        "matched_skills": [
            "Problem-solving",
            "Collaboration",
            "Communication"
        ],
        "missing_skills": [],
        "match_percent": 100
    },
    "domain_knowledge": {
        "matched_skills": [],
        "missing_skills": [
            "Energy markets",
            "Weather forecasting",
            "Production modelling"
        ],
        "match_percent": 0.0
    },
    "overall": {
        "match_score_percent": 53.3,
        "missing_required_skills": [
            "TensorFlow",
            "Cloud technologies",
            "AWS"
        ]
    }
}