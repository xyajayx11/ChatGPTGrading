from flask import Flask, request, render_template
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

def grade_essay(essay):
    doc = nlp(essay)
    pfo_score = grade_pfo(doc)
    ee_score = grade_ee(doc)
    conventions_score = grade_conventions(essay)
    return {"PFO": pfo_score, "EE": ee_score, "Conventions": conventions_score}

def grade_pfo(doc):
    thesis_keywords = ["thesis", "claim", "argument", "main idea", "position"]
    transition_keywords = ["first", "second", "finally", "in conclusion", "next", "therefore"]
    intro_keywords = ["introduction", "overview"]
    conclusion_keywords = ["conclusion", "final thoughts", "summary"]
    thesis_found = any(keyword in doc.text.lower() for keyword in thesis_keywords)
    transitions_found = sum(1 for token in doc if token.text.lower() in transition_keywords) > 3
    intro_found = any(keyword in doc.text.lower() for keyword in intro_keywords)
    conclusion_found = any(keyword in doc.text.lower() for keyword in conclusion_keywords)
    if thesis_found and transitions_found and intro_found and conclusion_found:
        return 4
    elif thesis_found and transitions_found:
        return 3
    elif thesis_found or intro_found or conclusion_found:
        return 2
    else:
        return 1

def grade_ee(doc):
    evidence_keywords = ["evidence", "example", "fact", "detail", "study", "source", "data", "research"]
    academic_vocab = ["significant", "crucial", "impact", "perspective", "interpretation", "theory", "framework"]
    evidence_found = sum(1 for token in doc if token.text.lower() in evidence_keywords) > 2
    vocab_used = sum(1 for token in doc if token.text.lower() in academic_vocab) > 3
    if evidence_found and vocab_used:
        return 4
    elif evidence_found or vocab_used:
        return 3
    elif any(token.text.lower() in evidence_keywords for token in doc):
        return 2
    else:
        return 1

def grade_conventions(essay):
    doc = nlp(essay)
    spelling_errors = sum(1 for token in doc if token.is_alpha and not nlp.vocab.has_vector(token.text.lower()))
    grammar_errors = sum(1 for token in doc if token.pos_ in ['VERB', 'ADJ', 'NOUN'] and len(token.text) > 15)
    if spelling_errors == 0 and grammar_errors == 0:
        return 2
    elif spelling_errors <= 2 and grammar_errors <= 2:
        return 1
    else:
        return 0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        essay = request.form["essay"]
        grades = grade_essay(essay)
        return render_template("index.html", grades=grades, essay=essay)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
