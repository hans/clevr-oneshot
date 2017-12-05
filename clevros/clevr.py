"""
Contains CLEVR metadata and minor utilities for working with CLEVR.
"""

from nltk.sem.logic import Expression


# CLEVR constants
# https://github.com/facebookresearch/clevr-dataset-gen/blob/master/question_generation/metadata.json
ENUMS = {
  "shape": ["cube", "sphere", "cylinder"],
  "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
  "relation": ["left", "right", "behind", "front"],
  "size": ["small", "large"],
  "material": ["rubber", "metal"],
}

# Maps object properties back to enumeration types.
# Assumes no overlap in enum values (true for now).
VAL_TO_ENUM = {val: enum for enum, vals in ENUMS.items()
               for val in vals}


ENUM_LF_TEMPLATES = {
  "shape": [r"\a.filter_shape(scene,a)"],
  "color": [r"\a.\x.filter_color(x,a)"],
  "size": [r"\a.\x.filter_size(x,a)"],
  "material": [r"\a.\x.filter_material(x,a)"],
}
ENUM_LF_TEMPLATES = {enum: [Expression.fromstring(expr)
                            for expr in exprs]
                     for enum, exprs in ENUM_LF_TEMPLATES.items()}


def scene_candidate_referents(scene):
  candidates = set()

  # for now, just enumerate object properties
  for obj in scene['objects']:
    for k, v in obj.items():
      if isinstance(v, str):
        enum = VAL_TO_ENUM[v]
        templates = ENUM_LF_TEMPLATES[enum]
        v_expr = Expression.fromstring(v)

        for template in templates:
          candidates.add(template.applyto(v_expr).simplify())

  return candidates


def functionalize_program(program, merge_filters=True):
  """
  Convert a CLEVR question program into a sexpr format,
  amenable to semantic parsing.
  """

  if merge_filters:
    # Traverse the program structure and merge nested filter operations.
    def merge_inner(idx):
      node = program[idx]
      fn = node["function"]
      if not fn.startswith("filter"):
        node["inputs"] = [merge_inner(input) for input in node["inputs"]]
        return idx

      assert len(node["inputs"]) == 1
      assert len(node["value_inputs"]) == 1

      filter_type = fn[fn.index("_") + 1:]

      # reduced form: (green obj)
      reduced_form = {
        "function": node["value_inputs"].replace("'", ""),
        "inputs": [0],
        "value_inputs": [],
      }
      program.append(reduced_form)
      reduced_form_idx = len(program) - 1

      child_idx = node["inputs"][0]
      child_idx = merge_inner(child_idx)
      if program[child_idx]["function"] == "filter":
        # Child is already merged. Add a reduced form of this node to the child
        # and return.
        program[child_idx]["inputs"].append(reduced_form_idx)
        return child_idx
      else:
        # Child is not a merged filter. Create a new function call.
        filter_call = {
          "function": "filter",
          "inputs": [
            child_idx,
            reduced_form_idx
          ],
          "value_inputs": [],
        }

        program.append(filter_call)
        return len(program) - 1

    merge_inner(len(program) - 1)

  def inner(p):
    if p['function'] == 'scene':
      return 'scene'
    ret = '(%s %s' % (p['function'],
                      ' '.join(inner(program[x]) for x in p['inputs']))
    if p['value_inputs']:
      ret += ' ' + ' '.join(map(repr, p['value_inputs']))
    ret += ')'
    return ret
  return inner(program[-1])


if __name__ == '__main__':
  question = "Are there any other things that are the same shape as the big metallic object?"
  program = None
  print(functionalize_program(program, merge_filters=False))
  print(functionalize_program(program, merge_filters=True))
