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
