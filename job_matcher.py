import json
from typing import Dict, List, Any, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from llm import LLM
from retriever import CVRetriever


class JobMatcher:
    def __init__(self, cv_path: str, jd_path: str) -> None:
        self.cv_path: str = cv_path
        self.jd_path: str = jd_path
        self.cv_chunks: List[Document] = []
        self.jd_chunks: List[Document] = []
        self.required_skills: Dict[str, List[str]] = {}
        self.cv_skill_matches: Dict[str, List[Dict[str, Union[str, bool]]]] = {}

    def load_documents(self) -> None:
        # Load CV and Job Description
        self.cv_chunks = self._load_pdf(self.cv_path)
        self.jd_chunks = self._load_pdf(self.jd_path, False)

    def _load_pdf(self, path: str, chunked: bool = True) -> List[Document]:
        loader = PyPDFLoader(path)
        doc = loader.load()
        if chunked:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            return splitter.split_documents(doc)
        else:
            return doc

    def extract_skills_from_jd(self, llm: LLM) -> None:
        joined_text = " ".join([doc.page_content for doc in self.jd_chunks])

        system_prompt = (
            "You are a recruitment assistant tasked with extracting "
            "required skills from job descriptions. Categorize "
            "the skills into the following groups:\n\n"
            "1. tech_stack_required: Technical skills and tools "
            "that are clearly marked as mandatory or essential.\n"
            "2. tech_stack_optional: Technical skills and tools "
            "that are preferred or nice-to-have.\n"
            "3. soft_skills: Interpersonal and communication skills.\n"
            "4. domain_knowledge: Industry-specific knowledge or expertise.\n\n"
            "Rules:\n"
            "- Return only canonical, general-purpose skill names "
            "(e.g., 'Docker', 'Model Deployment', 'Communication')\n"
            "- Do not include entire ability statements like "
            "'Ability to integrate X...' or 'Strong experience with...'\n"
            "- Avoid phrases like 'Experience in...' or 'Ability to...'\n"
            "- Do not include responsibilities, certifications, or qualifications\n"
            "- Normalize synonyms into one name (e.g., 'building ML systems', "
            "'integrating models' -> 'Model Deployment')\n\n"
            "Provide the output in JSON format adhering to the following schema:\n\n"
            "{\n  \"tech_stack_required\": [list of strings],\n  "
            "\"tech_stack_optional\": [list of strings],\n  "
            "\"soft_skills\": [list of strings],\n  "
            "\"domain_knowledge\": [list of strings]\n}\n\n"
            "Only output the JSON. Do not add commentary, markdown, or explanations."
        )
        user_prompt = (
            "Job description: \n\n"
            f"{joined_text}"
        )

        try:
            response = llm.invoke(system_prompt, user_prompt, response_format={"type": "json_object"})
            res = json.loads(response)
        except Exception as e:
            print("Failed to read skills from job description!")
            raise e  #TODO Error handling
        self.required_skills = res

    def check_skills_in_cv(self, llm: LLM, retriever: CVRetriever) -> None:
        system_prompt = (
            "You are a recruitment assistant evaluating CVs. Your job is to determine whether the candidate "
            "has demonstrated a single given skill based on CV excerpts. "
            "If the skill is clearly present (directly or via related tools/terms), mark it as matched. "
            "Categorize the output into the following groups:\n\n"
            "1. skill: the skill being evaluated\n"
            "2. matched: true or false\n"
            "3. reason: a short explanation (max 1 sentence)\n\n"
            "Provide the output in JSON format adhering to the following schema:\n\n"
            "{\n  \"skill\": string,\n  "
            "\"matched\": boolean,\n  "
            "\"reason\": string\n}\n\n"
            "Only base your answer on the CV content provided. Do not infer anything not stated. "
            "Only output the json object with the given keys. Do not add any other keys."
        )

        for category, skills in self.required_skills.items():
            self.cv_skill_matches[category] = []

            for skill in skills:
                context_docs = retriever.get_relevant_documents(skill)
                context = "\n".join([doc.page_content for doc in context_docs])

                user_prompt = (
                    f"Skill to evaluate: '{skill}'\n\n"
                    f"CV excerpts:\n{context}"
                )

                try:
                    response = llm.invoke(system_prompt, user_prompt, response_format={"type": "json_object"})
                    match_data = json.loads(response)
                    self.cv_skill_matches[category].append(match_data)

                except Exception as e:
                    print(f"Error checking skill '{skill}':", e)
                    self.cv_skill_matches[category].append({
                        "skill": skill,
                        "matched": False,
                        "reason": "LLM call failed or returned invalid JSON"
                    })

    def summarize_matching(self, out_path: str) -> None:
        summary: Dict[str, Any] = {}
        total_score: float = 0
        total_possible: float = 0

        weights: Dict[str, float] = {
            "tech_stack_required": 5.0,
            "tech_stack_optional": 2.0,
            "soft_skills": 4.0,
            "domain_knowledge": 1.0
        }

        for category, matches in self.cv_skill_matches.items():
            matched = [s for s in matches if s["matched"]]
            missing = [s for s in matches if not s["matched"]]
            cat_weight = weights.get(category, 1.0)

            cat_score = len(matched) * cat_weight
            cat_total = len(matches) * cat_weight

            total_score += cat_score
            total_possible += cat_total

            summary[category] = {
                "matched_skills": [s["skill"] for s in matched],
                "missing_skills": [s["skill"] for s in missing],
                "match_percent": round(100 * len(matched) / len(matches), 1) if matches else 0.0
            }

        summary["overall"] = {
            "match_score_percent": round(100 * total_score / total_possible, 1) if total_possible else 0.0,
            "missing_required_skills": summary["tech_stack_required"]["missing_skills"],
        }

        with open(out_path, "w") as fp:
            json.dump(summary, fp, indent=4)
