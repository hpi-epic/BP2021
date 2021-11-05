import os
import os.path
import re

pattern = "^args\.env-best_-?[0-9]+\.[0-9]+_marketplace\.dat$"
for root, dirs, files in os.walk(os.getcwd()):
    for file in filter(lambda x: re.match(pattern, x), files):
        os.remove(os.path.join(root, file))
