import os
import feedparser
import pandas as pd
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_mysqldb import MySQL
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fakenews123")

# ================= DATABASE =================
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "fake_news_db"

mysql = MySQL(app)

# ================= ML MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

fake_df = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X, y)

# ================= RSS NEWS =================
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.dawn.com/feed"
]

def get_news():
    news_list = []
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "source": feed.feed.get("title", "News")
            })
    return news_list

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def home():
    news = get_news()

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if news_text:
            words = news_text.split()
            valid_word_count = sum(1 for w in words if len(w) >= 3)

            if len(words) < 5 or valid_word_count < 4:
                return render_template("home.html",
                    invalid=True,
                    invalid_msg="Please enter a valid news article or headline (at least 5 meaningful words).",
                    news=news,
                    news_text=news_text,
                    result=None
                )

            vec = vectorizer.transform([news_text])
            prediction = model.predict(vec)[0]
            decision = model.decision_function(vec)[0]
            confidence = round(min(abs(float(decision)) * 20, 100), 1)

            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO feedback (news_text, prediction) VALUES (%s, %s)",
                (news_text[:500], prediction)
            )
            mysql.connection.commit()
            news_id = cur.lastrowid
            cur.close()

            return redirect(url_for("result_page", news_id=news_id))

    return render_template("home.html", result=None, news=news, news_text="", invalid=False, invalid_msg="")

# ================= RESULT PAGE =================
@app.route("/result/<int:news_id>")
def result_page(news_id):
    news = get_news()

    cur = mysql.connection.cursor()
    cur.execute("SELECT news_text, prediction FROM feedback WHERE id=%s", (news_id,))
    row = cur.fetchone()
    cur.close()

    if not row:
        return redirect(url_for("home"))

    news_text, prediction = row
    vec = vectorizer.transform([news_text])
    decision = model.decision_function(vec)[0]
    confidence = round(min(abs(float(decision)) * 20, 100), 1)

    result = {
        "label": prediction,
        "confidence": confidence,
        "cls": "real" if prediction == "REAL" else "fake",
        "id": news_id
    }

    return render_template("home.html", result=result, news=news, news_text=news_text, invalid=False, invalid_msg="")

# ================= LIKE =================
@app.route("/like", methods=["POST"])
def like():
    data = request.json
    user_name = data.get("user_name", "").strip()
    news_id = data.get("news_id")

    if not user_name or not news_id:
        return jsonify({"error": "Invalid input"}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO likes (news_id, user_name) VALUES (%s, %s)",
            (news_id, user_name)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"message": "Liked"})
    except Exception:
        return jsonify({"error": "Already liked or DB error"}), 409

# ================= COMMENT =================
@app.route("/comment", methods=["POST"])
def comment():
    data = request.json
    user_name = data.get("user_name", "").strip()
    comment_text = data.get("comment_text", "").strip()
    news_id = data.get("news_id")

    if not user_name or not comment_text or not news_id:
        return jsonify({"error": "Invalid input"}), 400

    if len(comment_text) > 500:
        return jsonify({"error": "Too long"}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO comments (news_id, user_name, comment_text) VALUES (%s,%s,%s)",
            (news_id, user_name, comment_text)
        )
        mysql.connection.commit()
        cur.close()
        return jsonify({"message": "Comment added"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= GET COMMENTS =================
@app.route("/get_comments/<int:news_id>")
def get_comments(news_id):
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT user_name, comment_text, commented_at FROM comments WHERE news_id=%s ORDER BY id DESC",
        (news_id,)
    )
    rows = cur.fetchall()
    cur.close()

    return jsonify([
        {
            "user_name": r[0],
            "comment_text": r[1],
            "commented_at": str(r[2])
        }
        for r in rows
    ])

# ================= GET LIKES =================
@app.route("/get_likes/<int:news_id>")
def get_likes(news_id):
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM likes WHERE news_id=%s",
        (news_id,)
    )
    count = cur.fetchone()[0]
    cur.close()

    return jsonify({"likes": count})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)