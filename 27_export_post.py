"""Copy the blog post and the figures it uses into the website's post folder.

Reads post/index.qmd, finds every figure it embeds (iframe data-fig="NAME"), and copies
output/story/NAME.html to <post folder>/figures/NAME.html, plus index.qmd itself.
Usage: python 27_export_post.py [post folder]
  default post folder: ./post  (only the figures are copied there)
  e.g.  python 27_export_post.py ~/path/to/aadhavr.github.io/posts/who-shows-up
"""
import re
import shutil
import sys
from pathlib import Path
from config import OUT

SRC = Path(__file__).parent / "post" / "index.qmd"
dest = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else SRC.parent
names = re.findall(r'<iframe[^>]*data-fig="([^"]+)"', SRC.read_text(encoding="utf-8"))
(dest / "figures").mkdir(parents=True, exist_ok=True)
missing = []
for n in dict.fromkeys(names):
    f = OUT / "story" / f"{n}.html"
    if f.exists():
        shutil.copy2(f, dest / "figures" / f.name)
    else:
        missing.append(n)
if dest.resolve() != SRC.parent.resolve():
    shutil.copy2(SRC, dest / "index.qmd")
print(f"[27] {len(set(names)) - len(missing)} figures copied to {dest / 'figures'}")
if missing:
    print(f"[27] MISSING (run 26_story_figures.py or the script that feeds them): {missing}")
