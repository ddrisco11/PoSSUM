"""Download available source tables and normalize candidate records.
This is ingestion, not automatic admission to a viewpoint training corpus.
"""
import collections,hashlib,json,re
from pathlib import Path
import requests
import pyarrow.parquet as pq

SOURCES=['Jacobvs/PoliticalTweets','mlburnham/dem_rep_party_platform_topics','cajcodes/political-bias']

def clean(text):
    return ' '.join(re.sub(r'https?://\S+|(?<!\w)@\w+',' ',text).split())

def main():
    root=Path('data/available');root.mkdir(parents=True,exist_ok=True)
    report={}
    for repo in SOURCES:
        slug=repo.replace('/','_');dest=root/slug;dest.mkdir(exist_ok=True)
        r=requests.get('https://datasets-server.huggingface.co/parquet',params={'dataset':repo},timeout=60);r.raise_for_status()
        files=r.json()['parquet_files'];records=[];counts=collections.Counter();hashes={}
        for i,item in enumerate(files):
            path=dest/f"{item['split']}-{i}.parquet"
            if not path.exists():
                response=requests.get(item['url'],timeout=180);response.raise_for_status();path.write_bytes(response.content)
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            for row in pq.read_table(path).to_pylist():
                counts['raw']+=1
                if repo.startswith('Jacobvs'):
                    party=row['party']
                    if party not in ('Democrat','Republican'):
                        counts['unknown_party']+=1;continue
                    if not row['username']:
                        counts['missing_author']+=1;continue
                    if str(row['text']).lstrip().startswith('RT '):
                        counts['retweet']+=1;continue
                    text=clean(row['text']);label=int(party=='Republican');kind='weak'
                    groups=['twitter:author:'+row['username'].casefold()]
                    quality='author';upstream_id=str(row['id'])
                elif repo.startswith('mlburnham'):
                    party=row['dataset']
                    if party not in ('Dem. Party Platforms','Rep. Party Platforms'):
                        raise ValueError(f'Unknown platform source {party}')
                    text=clean(row['premise']);label=int(party=='Rep. Party Platforms');kind='weak'
                    groups=['platform:entire_source'];quality='source_only_document_ids_missing';upstream_id=None
                else:
                    original=row['label']
                    if original not in range(5):raise ValueError('Unknown five-way label')
                    if original==2:
                        counts['neutral']+=1;continue
                    text=clean(row['text']);label=int(original<2);kind='synthetic'
                    groups=['cajcodes:entire_generation_source'];quality='source_only_generation_ids_missing';upstream_id=None
                if not 20<=len(text.split())<=150:
                    counts['length_excluded']+=1;continue
                records.append(dict(text=text,label=label,source=repo,label_kind=kind,group_ids=groups,group_quality=quality,upstream_id=upstream_id,upstream_split=item['split'],admission='candidate_pending_politicalness_and_ambiguity_filter' if kind=='weak' else 'candidate'))
        unique={};conflicts=set()
        for row in records:
            key=hashlib.sha256(' '.join(re.findall(r'\w+',row['text'].casefold())).encode()).hexdigest()
            if key in unique:
                counts['exact_duplicates']+=1
                if unique[key]['label']!=row['label']:conflicts.add(key)
                unique[key]['group_ids']=sorted(set(unique[key]['group_ids']+row['group_ids']))
            else:unique[key]=row
        records=[row for key,row in unique.items() if key not in conflicts]
        counts['conflicting_texts']=len(conflicts)
        counts['candidates']=len(records)
        counts['left']=sum(r['label']==0 for r in records);counts['right']=sum(r['label']==1 for r in records)
        (dest/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        report[repo]={'counts':dict(counts),'unique_groups':len({g for r in records for g in r['group_ids']}),'download_sha256':hashes,'status':'downloaded and normalized; cross-source near deduplication, split assignment and weak-label admission remain separate steps'}
        print(repo,report[repo],flush=True)
    (root/'ingestion-report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
