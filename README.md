
Prerequisites:

pip install opencv-python insightface onnxruntime numpy sqlite3
python -m pip install onnxruntime

How to run:
python one.py

Features:- 

Detect – app.get(img) gives bounding boxes & landmarks.

Verify – verify_face() returns True/False + score for a claimed identity.

Identify – identify_face() scans the SQL table and returns the best match.

FindSimilar – find_similar() returns the N most similar faces with scores.

SQL storage – embeddings are persisted as BLOBs; names act as primary keys.
