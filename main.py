import os
import argparse
from dotenv import load_dotenv

from job_matcher import JobMatcher
from llm import LLM
from retriever import CVRetriever


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description='Match CV against job description')
    parser.add_argument('--cv', type=str, default='data/cv.pdf', help='Path to CV PDF file')
    parser.add_argument('--jd', type=str, default='data/job.pdf', help='Path to job description PDF file')
    parser.add_argument('--out', type=str, default='matched_skills.json', help='Path to output json')
    args = parser.parse_args()

    cv_path = args.cv
    jd_path = args.jd
    out_path = args.out

    matcher = JobMatcher(cv_path, jd_path)
    matcher.load_documents()

    cv_retriever = CVRetriever(matcher.cv_chunks)
    cv_retriever.build_retriever()

    llm = LLM(os.getenv("GROQ_API_TOKEN"))
    matcher.extract_skills_from_jd(llm=llm)
    matcher.check_skills_in_cv(llm=llm, retriever=cv_retriever.retriever)
    matcher.summarize_matching(out_path)