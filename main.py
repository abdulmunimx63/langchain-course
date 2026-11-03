import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def main():
    print("Hello from langchain-course!")
    print(api_key)


if __name__ == "__main__":
    main()