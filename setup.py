from setuptools import setup, find_packages

setup(
    name="agentic-rag-engine",
    version="0.1.0",
    description="Multi-tenant Agentic-RAG engine on GCP Vertex AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "google-cloud-aiplatform",
        "pydantic",
        "pydantic-settings",
    ],
    python_requires=">=3.10",
)