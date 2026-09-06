import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load environment variables from .env
load_dotenv()


# Check required API keys
required_keys = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
]

missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print("Error: Missing required environment variables:")

    for key in missing_keys:
        print(f"  - {key}")

    print("\nPlease check your .env file.")
    sys.exit(1)


# Create the OpenAI model through LangChain
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
)


print("=" * 60)
print("LangChain + OpenAI + LangSmith Chat")
print("=" * 60)
print("Type your question and press Enter.")
print("Type 'exit' or 'quit' to stop the program.")
print("=" * 60)


# Continuous question-answer loop
while True:

    prompt = input("\nYou: ").strip()

    # Exit condition
    if prompt.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Don't send empty questions
    if not prompt:
        print("Please enter a question.")
        continue

    try:
        # Send the question through LangChain
        response = model.invoke(prompt)

        # Print the model response
        print("\nAI:", response.content)

    except Exception as e:
        print("\nError:", e)
