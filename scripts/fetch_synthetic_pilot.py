"""Download public synthetic data; invert upstream labels; group entire issue families.
This is an integration pilot, not human-text evaluation. Original generation batch IDs
are unavailable; issue grouping is a conservative proxy, not verified provenance.
"""
import json
from pathlib import Path
import requests

out=Path("data")
out.mkdir(exist_ok=True)
repo="JyotiNayak/political_ideologies"
records=[]
for split,total in (("train",2560),("test",320),("validation",320)):
    for offset in range(0,total,100):
        r=requests.get("https://datasets-server.huggingface.co/rows",params={"dataset":repo,"config":"default","split":split,"offset":offset,"length":min(100,total-offset)},timeout=60)
        r.raise_for_status()
        for item in r.json()["rows"]:
            x=item["row"]
            if x["label"] not in (0,1):
                raise ValueError("Unexpected upstream label")
            records.append({"text":x["statement"],"label":1-x["label"],"source":repo,"label_kind":"synthetic","group_ids":[f"{repo}:issue:{x['issue_type']}"],"group_quality":"topic_proxy_not_generation_batch","upstream_split":split,"upstream_id":x["__index_level_0__"]})
    print(split,len(records),flush=True)
(out/"synthetic.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
print("Saved",len(records),"synthetic records; not a human-verified final evaluation.")
