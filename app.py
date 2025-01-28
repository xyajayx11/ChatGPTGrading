from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

# Predefined rubrics for grading
RUBRICS = {
    "organization": {
        4: "Highly effective organization and logical progression of ideas.",
        3: "Effective organization with logical progression and minor flaws.",
        2: "Some organization but with inconsistent logic and progression.",
        1: "Minimal organization with limited progression.",
        0: "No clear organization."
    },
    "elaboration": {
        4: "Fully developed ideas with strong evidence and explanation.",
        3: "Well-developed ideas with some evidence and explanation.",
        2: "Some evidence or explanation, but inconsistently developed.",
        1: "Limited evidence or explanation; weakly developed ideas.",
        0: "No clear evidence or explanation."
    },
    "conventions": {
        2: "Few to no errors in grammar, spelling, or punctuation.",
        1: "Some errors in grammar, spelling, or punctuation, but they do not impede understanding.",
        0: "Many errors that impede understanding."
    }
}

# Home Page
@main.route('/')
def home():
    return render_template("index.html")

# Grading Route
@main.route('/grade', methods=["POST"])
def grade():
    essay = request.form.get("essay")
    scores = {}

    # Analyze organization
    if len(essay.split()) > 200:  # Example logic
        scores["organization"] = 4
    elif len(essay.split()) > 150:
        scores["organization"] = 3
    elif len(essay.split()) > 100:
        scores["organization"] = 2
    elif len(essay.split()) > 50:
        scores["organization"] = 1
    else:
        scores["organization"] = 0

    # Analyze elaboration (word count, depth, quotes)
    if 'for example' in essay or 'this shows' in essay:
        scores["elaboration"] = 4 if len(essay.split()) > 200 else 3
    elif 'because' in essay:
        scores["elaboration"] = 2
    else:
        scores["elaboration"] = 1 if len(essay.split()) > 50 else 0

    # Analyze conventions (spelling or grammar errors as an example)
    errors = sum([1 for word in essay.split() if word.lower() not in ["the", "and", "is", "a", "to", "in"]])  # Simple error simulation
    scores["conventions"] = 2 if errors < 5 else 1 if errors < 10 else 0

    # Return results
    results = {category: RUBRICS[category][score] for category, score in scores.items()}
    total_score = sum(scores.values())
    return render_template("result.html", scores=scores, results=results, total_score=total_score)
