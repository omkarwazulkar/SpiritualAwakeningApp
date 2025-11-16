import os
import httpx
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Set API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

os.environ["OPENAI_API_KEY"] = api_key

def explainSelectedVerses(selectedDocs):
    insecure_http_client = httpx.Client(verify=False)

    raw_openai = OpenAI(
        api_key=api_key,
        http_client=insecure_http_client
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=200,
        openai_api_key=api_key,
        client=raw_openai.chat.completions
    )

    all_explanations = []

    for key, doc in selectedDocs.items():
        verseNo = doc.metadata.get("verse_no", "Unknown")
        sanskrit = doc.metadata.get("sanskrit_text", "")
        content = doc.page_content.strip()

        prompt = ChatPromptTemplate.from_template(
            """In the verse (shlok) {verse_no}, it is explained:
            "{page_content}"

            Provide a clear and insightful explanation in English within 200 tokens.

            The verse that explains this is:
            {sanskrit_text}
            """
        )

        response = llm.invoke(prompt.format(
            verse_no=verseNo,
            page_content=content,
            sanskrit_text=sanskrit
        ))

        explanation = response.content.strip()

        all_explanations.append({
            "verse_no": verseNo,
            "sanskrit_text": sanskrit,
            "translation": content,
            "explanation": explanation
        })
        print()
        print("✅ Explanation Generated.")
        print(all_explanations)
    return all_explanations