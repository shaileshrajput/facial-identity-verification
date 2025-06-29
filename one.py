import os, cv2, sqlite3, numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 0.  Choose the best execution provider ----------
def best_providers():
    avail = ort.get_available_providers()
    preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return [p for p in preferred if p in avail]

# ---------- 1.  Init model ----------
app = FaceAnalysis(name='buffalo_l', providers=best_providers())
app.prepare(ctx_id=0)
print("Providers in use:", best_providers())

# ---------- 2.  Init / connect SQLite ----------
conn = sqlite3.connect('face_db.sqlite')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS users (
                  id        INTEGER PRIMARY KEY AUTOINCREMENT,
                  name      TEXT UNIQUE,
                  embedding BLOB
              )''')
conn.commit()

# ---------- 3.  Detect & register ----------
def register_face(image_path: str, name: str):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return f"No face in {image_path}"
    emb = faces[0].embedding.astype(np.float32).tobytes()
    try:
        cur.execute('INSERT INTO users (name, embedding) VALUES (?,?)', (name, emb))
        conn.commit()
        return f"✅ Registered {name}"
    except sqlite3.IntegrityError:
        return f"User {name} already exists"

# ---------- 4.  Verify (1:1) ----------
def verify_face(image_path: str, claimed_name: str, thresh=0.6):
    cur.execute('SELECT embedding FROM users WHERE name=?', (claimed_name,))
    row = cur.fetchone()
    if not row:
        return "User not found"
    known_emb = np.frombuffer(row[0], dtype=np.float32).reshape(1, -1)

    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return "No face detected"
    test_emb = faces[0].embedding.astype(np.float32).reshape(1, -1)

    sim = cosine_similarity(known_emb, test_emb)[0, 0]
    return sim >= thresh, float(sim)

# ---------- 5.  Identify (1:N) ----------
def identify_face(image_path: str, thresh=0.6):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return "No face detected"
    test_emb = faces[0].embedding.astype(np.float32).reshape(1, -1)

    cur.execute('SELECT name, embedding FROM users')
    matches = []
    for name, blob in cur.fetchall():
        known_emb = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
        sim = cosine_similarity(known_emb, test_emb)[0, 0]
        if sim >= thresh:
            matches.append((name, sim))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0] if matches else ("Unknown", None)

# ---------- 6.  FindSimilar ----------
def find_similar(image_path: str, top_n=3):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return "No face detected"
    test_emb = faces[0].embedding.astype(np.float32).reshape(1, -1)

    cur.execute('SELECT name, embedding FROM users')
    sims = []
    for name, blob in cur.fetchall():
        known_emb = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
        sim = cosine_similarity(known_emb, test_emb)[0, 0]
        sims.append((name, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

# ---------- 7.  Example sequence ----------
if __name__ == "__main__":
    print(register_face("amit.jpeg", "Amit"))
    print(register_face("lata.jpeg",   "Lata"))

    ok, score = verify_face("amit_test.jpeg", "Amit")
    print("Verification:", ok, "Score:", score)

    who, score = identify_face("rajesh.jpeg")
    print("Identified:", who, "Score:", score)

    print("Top‑3 similar:", find_similar("rajesh.jpeg"))
