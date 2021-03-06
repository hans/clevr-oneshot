"""
Convert CLEVR question programs to a nice LoT format, amenable to semantic
parsing.
"""

import json
from pathlib import Path

from clevros.clevr import functionalize_program


# split = "val"
# data_file = Path(__file__).parents[0] / ".." / "data" / "CLEVR_v1.0" / "questions" / ("CLEVR_%s_questions.json" % split)
# with data_file.open() as f:
#   data = json.load(f)

# qs = data['questions']

# import pickle
# with open("val.pkl", "wb") as f:
#   pickle.dump(qs, f, pickle.HIGHEST_PROTOCOL)
import pickle
with open("val.pkl", "rb") as f:
  qs = pickle.load(f)

for i, question in enumerate(qs):
  print(question["question"])
  print(question["program"])
  print(functionalize_program(question["program"], merge_filters=False))
  print(functionalize_program(question["program"]))
  print()

  if i == 5:
    sys.exit()
