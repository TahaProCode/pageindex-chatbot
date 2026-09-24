import json
import time
import urllib.request

payload = {
    "model": "qwen3:4b",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are navigating a document's outline to find sections relevant to a "
                "question. You are shown a numbered list of section titles and short "
                "summaries. Reply with ONLY the numbers of sections that might help answer "
                "the question, separated by commas (example: 1,3). If none of them seem "
                "relevant, reply with exactly: NONE. Do not add any explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Question: What is inside this document?\n\n"
                "Sections:\n"
                "1. Chapter 1: Introduction - Introduces the document's topic.\n"
                "2. Chapter 2: Methodology - Describes methods used.\n\n"
                "Which section numbers might contain the answer?"
            ),
        },
    ],
    "stream": False,
    "think": False,
    "options": {"temperature": 0},
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

start = time.time()
with urllib.request.urlopen(req, timeout=180) as resp:
    body = resp.read().decode("utf-8")
elapsed = time.time() - start

print(f"Took {elapsed:.1f} seconds\n")
print("Raw response:")
print(json.dumps(json.loads(body), indent=2))