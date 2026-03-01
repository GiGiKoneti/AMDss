import os
import sys

# Ensure linguist_core is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from linguist_core.extractor import KnowledgeExtractor
import logging

logging.basicConfig(level=logging.INFO)

extractor = KnowledgeExtractor(use_mock=False)

test_input = "Newton's Second Law states that Force equals mass multiplied by acceleration. This principle underlies rocket propulsion, where thrust force acts on the rocket's mass to produce acceleration."

triplets = extractor.extract_triplets(test_input)

print("--- STEP 1 DIAGNOSTIC RESULTS ---")
for t in triplets:
    print(f"({t.subject}) --[{t.predicate}]--> ({getattr(t, 'object_', getattr(t, 'object', 'None'))})")
print("---------------------------------")
