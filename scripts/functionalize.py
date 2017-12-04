"""
Convert CLEVR question programs to a nice LoT format, amenable to semantic
parsing.
"""

import json
from pathlib import Path

from clevros.clevr import functionalize_program


split = "val"
data_file = Path(__file__).parents[0] / ".." / "data" / "CLEVR_v1.0" / "questions" / ("CLEVR_%s_questions.json" % split)
with data_file.open() as f:
  data = json.load(f)

qs = data['questions']

for question in qs:
  print(question["question"])
  print(functionalize_program(question["program"]))
  print()
  sys.exit()
