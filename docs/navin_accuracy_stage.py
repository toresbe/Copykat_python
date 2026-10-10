"""Stage one matrix group independently of the inference dispatcher."""
import argparse,json,traceback
from navin_accuracy_data import ROOT,save,stage_group
p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--name',required=True)
a=p.parse_args();ids=a.cases.split(',')
cases=[c for c in json.loads((ROOT/'dataset_manifest.json').read_text())['cases'] if c['id'] in ids]
try:
    assert len(cases)==len(ids)
    stage_group(cases);result=dict(status='ok',cases=ids)
except Exception:result=dict(status='input_error',cases=ids,error=traceback.format_exc())
save(ROOT/'results'/(a.name+'.json'),result)
