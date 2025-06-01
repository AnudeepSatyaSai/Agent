import os
import logging
from datetime import datetime
import hnswlib
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate

# Load .env and configure Gemini
load_dotenv()
GEN_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEN_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("output/agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AgenticCodeGen")

# Memory Store using HNSWlib
class VectorStore:
    def __init__(self, dim=384, max_elements=10000):
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.data = {}
        self.dim = dim
        self.current_id = 0

    def add(self, text, metadata):
        vector = self._embed(text)
        self.index.add_items(vector, [self.current_id])
        self.data[self.current_id] = {"text": text, "metadata": metadata}
        self.current_id += 1

    def search(self, query, k=3):
        try:
            vector = self._embed(query)
            labels, _ = self.index.knn_query(vector, k=k)
            return [self.data[i] for i in labels[0]]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _embed(self, text):
        hash_val = hash(text) % (2 ** 32)
        vec = np.random.RandomState(hash_val).rand(self.dim).astype('float32')
        return vec.reshape(1, -1)

# Agents
class ArchitectureAgent:
    def design(self, description, language, complexity):
        return f"# Architecture Design\n# System for: {description}\n# Language: {language}, Complexity: {complexity}\n"

class CodeGeneratorAgent:
    def generate(self, description, language, complexity, context):
        prompt = PromptTemplate(
            input_variables=["description", "language", "complexity", "context"],
            template="""
You are an expert software engineer. Write complete {language} code for:

Description: {description}
Complexity: {complexity}
Context:
{context}

Write only code, no explanation.
"""
        )
        return model.generate_content(prompt.format(
            description=description,
            language=language,
            complexity=complexity,
            context=context
        )).text

class ReviewerAgent:
    def review(self, code):
        return model.generate_content(f"Suggest improvements for this code:\n{code}").text

class DocumentationAgent:
    def document(self, code):
        return model.generate_content(f"Add inline comments and documentation for this code:\n{code}").text

class TestingAgent:
    def test(self, description, language):
        return model.generate_content(f"Generate unit tests for a {language} code that implements: {description}").text

# Codegen Controller
class CodeGenAgent:
    def __init__(self):
        self.memory = VectorStore()
        self.architect = ArchitectureAgent()
        self.generator = CodeGeneratorAgent()
        self.reviewer = ReviewerAgent()
        self.documentation = DocumentationAgent()
        self.tester = TestingAgent()

        os.makedirs("output/python", exist_ok=True)
        os.makedirs("output/html", exist_ok=True)
        os.makedirs("output/other", exist_ok=True)

    def generate(self, description, language, complexity="medium", context=""):
        logger.info(f"🔍 Generating code for: {description} in {language}")

        past_context = "\n".join([s["text"] for s in self.memory.search(description)])
        arch = self.architect.design(description, language, complexity)
        raw_code = self.generator.generate(description, language, complexity, context + "\n" + past_context + "\n" + arch)
        documented = self.documentation.document(raw_code)
        reviewed = self.reviewer.review(raw_code)
        tests = self.tester.test(description, language)

        final_code = f"{arch}\n\n# Code:\n{documented}\n\n# Suggestions:\n{reviewed}\n\n# Tests:\n{tests}"

        folder = "output/other"
        ext = ".txt"
        if language.lower() == "python":
            folder = "output/python"
            ext = ".py"
        elif language.lower() in ["html", "html5"]:
            folder = "output/html"
            ext = ".html"

        filename = f"{description.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(final_code)

        self.memory.add(final_code, {"description": description, "lang": language})
        print(f"✅ Code generated at: {path}")

# Main loop
def main():
    agent = CodeGenAgent()
    while True:
        desc = input("🧠 What should I build? ").strip()
        lang = input("💻 Language (e.g., python, html): ").strip()
        comp = input("⚙️  Complexity (simple/medium/complex): ").strip() or "medium"
        ctx = input("📄 Additional context (optional): ").strip()

        agent.generate(desc, lang, comp, ctx)
        if input("🔁 Generate more? (y/n): ").lower() != "y":
            break

if __name__ == "__main__":
    main()
