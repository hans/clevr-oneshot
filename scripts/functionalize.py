"""
Convert CLEVR question programs to a nice LoT format, amenable to semantic
parsing.
"""

import json
from pathlib import Path

split = "train"
data_file = Path(__file__).parents[0] / ".." / "data" / "CLEVR_v1.0" / "questions" / ("CLEVR_%s_questions.json" % split)
with data_file.open() as f:
    data = json.load(f)

qs = data['questions']

def functionalize_program(program):
    def inner(p):
        if p['function'] == 'scene':
            return 'scene'
        ret = '%s(%s' % (p['function'],
                         ','.join(inner(program[x]) for x in p['inputs']))
        if p['value_inputs']:
            ret += ',' + ','.join(map(repr, p['value_inputs']))
        ret += ')'
        return ret
    return inner(program[-1])

for question in qs:
    print(functionalize_program(question["program"]))
