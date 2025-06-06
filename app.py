from flask import Flask, request, jsonify
from rag.chain_setup import get_rag_chain

app = Flask(__name__)

# Initialize your RAG chain once when the app starts
rag_chain = get_rag_chain()

@app.route("/")
def home():
    return "Smart Calendar Assistant is running."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = rag_chain.run(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)