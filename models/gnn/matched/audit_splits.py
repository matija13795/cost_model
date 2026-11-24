"""Report exact program/annotation overlap without altering supplied membership."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from .batches import program_index
from ..io_utils import atomic_json,sha256


def compare(train,valid):
    by_annotation=defaultdict(list)
    for name,row in valid.items():by_annotation[row['annotation_sha256']].append(name)
    same_ids=sorted(set(train)&set(valid))
    same_annotations={name:by_annotation[row['annotation_sha256']] for name,row in train.items()
                      if row['annotation_sha256'] in by_annotation}
    return dict(program_id_overlap=len(same_ids),program_ids=same_ids,
                train_programs_with_exact_validation_annotation=len(same_annotations),
                annotation_matches=same_annotations,
                no_exact_overlap=not same_ids and not same_annotations)


def audit(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Preserve existing split audit; choose another output')
    indices={s:program_index(root/s,verify=True) for s in ('1m','5m','10m','validation')}
    comparisons={s:compare(indices[s],indices['validation']) for s in ('1m','5m','10m')}
    result=dict(phase='split_audit',updated_unix=time.time(),comparisons=comparisons,
                program_manifest_sha256={s:sha256(root/s/'manifest.json') for s in indices},
                no_train_validation_overlap=all(x['no_exact_overlap'] for x in comparisons.values()),
                scope='Exact program IDs and canonical annotation JSON hashes; not a semantic clone detector. No membership changed.')
    atomic_json(result,out)
    print(json.dumps({s:{k:v for k,v in r.items() if k not in ('program_ids','annotation_matches')} for s,r in comparisons.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.root,a.out)
