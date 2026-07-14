import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from settings import (
    GROQ_API_KEY,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=LLM_MODEL,
    temperature=0
)


def ask_question(question):

    if question.strip() == "":
        return "Please enter a question."

    if not os.path.exists(VECTOR_DB_PATH):
        return "Please upload a PDF first."

    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    docs = db.similarity_search(
        question,
        k=TOP_K
    )

    if len(docs) == 0:
        return "No relevant information found."

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    sources = list(
        set(
            doc.metadata.get("source", "Unknown")
            for doc in docs
        )
    )

    prompt = f"""
You are an expert AI assistant.

Use ONLY the information from the provided context.

Give a detailed answer with proper explanations.

If the topic contains definitions, advantages, types, applications, examples or important points, explain all of them.

If the answer is not available in the context, reply:
"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Detailed Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    answer += "\n\n**Sources:**\n"

    for source in sources:
        answer += f"- {source}\n"

    return answer


if __name__ == "__main__":

    while True:

        question = input("Ask: ")

        if question.lower() == "exit":
            break

        print()
        print(ask_question(question))