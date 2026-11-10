"""Export reviewable tables from uniform-horizon reruns and raw logs."""
import csv
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT
OUT=ROOT/'analysis'
WORK=OUT


def stat(vals):return {'n':len(vals),'mean':st.mean(vals),'sd':st.pstdev(vals)}


def csv_write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


main=[];mechanisms={};translations={};fusion={}
for v in ['su2','so3']:
    for frac in ['0.002','0.01']:
        logs=[];causals=[]
        for s in range(5):
            run=f'data/su2/{v}_N2048_f{frac}_s{s}'
            rs=[json.loads(x) for x in (SRC/run/'log.jsonl').read_text().splitlines()]
            logs.append(next(x for x in reversed(rs) if x['step']==100000))
            path=WORK/'selected_checkpoints'/run/'step_100000'
            causals.append(json.loads((path/'causal_100000.json').read_text()))
        def agg(fn):return stat([fn(x) for x in logs])
        row={'variant':v,'fraction':float(frac),'step':100000,'seeds':5,'causal_evaluation_n':8192}
        key='fve_0.5' if v=='su2' else 'fve_1.0'
        for label,field in [('selected_FVE',key),('noisy_T2','acc_T2'),('center_cosine','pm_cos_mean')]:
            ss=agg(lambda x:x[field]);row[label+'_mean']=ss['mean'];row[label+'_sd']=ss['sd']
        cf='r2_upto_j0.5' if v=='su2' else 'r2_upto_j1.0'
        ss=stat([c['character_fit'][cf] for c in causals]);row['character_R2_mean']=ss['mean'];row['character_R2_sd']=ss['sd']
        for label,iv in [('exact_positive','none'),('remove_spin_half','fablate_spin_half'),('remove_spin_one','fablate_spin_one')]:
            ss=stat([c['interventions'][iv]['task_acc'] for c in causals]);row[label+'_mean']=ss['mean'];row[label+'_sd']=ss['sd']
        main.append(row)
        if frac=='0.01':
            md=[];td=[];fd=[]
            for s in range(5):
                path=WORK/'selected_checkpoints'/f'data/su2/{v}_N2048_f{frac}_s{s}'/'step_100000'
                md.append(json.loads((path/'mechanism_100000.json').read_text()))
                td.append(json.loads((path/'twosided_100000.json').read_text()))
                fd.append(json.loads((path/'fusion_100000.json').read_text()))
            mechanisms[v]={'action':{k:stat([d['action'][k] for d in md]) for k in md[0]['action']}}
            if v=='su2':mechanisms[v]['product']={k:stat([d['product'][k] for d in md]) for k in md[0]['product']}
            translations[v]={name:{k:stat([d['systems'][name][k] for d in td]) for k in td[0]['systems'][name]} for name in td[0]['systems']}
            fusion[v]={k:stat([d[k] for d in fd]) for k in ['diag_sector_energy_frac','contraction_frac_of_diag','contraction_dim','diag_sector_dim']}
csv_write('verified_mainline_100k.csv',main)

definitions=[('SU(2)','su2','su2','0.002',['fve_0.5'],None),
 ('SO(3)','su2','so3','0.002',['fve_1.0'],None),
 ('SU(3)','su3','su3','0.01',['fve_10'],None),
 ('PSU(3)','su3','psu3','0.01',['fve_11'],None),
 ('Spin(4)','spin4','spin4','0.01',['fve_0.5_0','fve_0_0.5'],None),
 ('SO(4)','spin4','so4','0.01',['fve_0.5_0.5'],None),
 ('SO(3) x SU(2)','spin4','leftq','0.01',['fve_1_0','fve_0_0.5'],'fve_1_0.5'),
 ('PSO(4)','spin4','pso4','0.01',['fve_1_0','fve_0_1'],'fve_1_1'),
 ('Spin(5)','spin5','spin5','0.01',['fve_4'],None),
 ('SO(5) averaged','spin5','so5','0.01',['fve_5'],'fve_10'),
 ('SO(5) intrinsic','spin5','so5rf','0.03',['fve_5'],'fve_10')]
groups=[]
for label,family,v,frac,keys,alt in definitions:
    rs=[];configs=[]
    for s in range(5):
        name=f'{v}_f{frac}_s{s}' if v=='so5rf' else f'{v}_N2048_f{frac}_s{s}'
        path=SRC/'data'/family/name
        logs=[json.loads(x) for x in (path/'log.jsonl').read_text().splitlines()]
        rs.append(next(x for x in reversed(logs) if x['step']==100000))
        configs.append(json.loads((path/'config.json').read_text()))
    fve=stat([sum(r[k] for k in keys) for r in rs]);acc=stat([r['acc_T2'] for r in rs])
    row={'group':label,'family':family,'variant':v,'fraction':float(frac),'step':100000,'seeds':5,
         'FVE_mean':fve['mean'],'FVE_sd':fve['sd'],'T2_mean':acc['mean'],'T2_sd':acc['sd'],
         'alternative_FVE_mean':None if alt is None else st.mean(r[alt] for r in rs),
         'alternative_FVE_sd':None if alt is None else st.pstdev(r[alt] for r in rs),
         'reference_count_m':configs[0]['m'],'observed_width':2*configs[0]['m'] if family=='su3' else configs[0]['m'],
         'alpha':configs[0]['alpha'],'sigma':configs[0]['sigma'],'exclusion_radius':configs[0]['excl_radius'],
         'batch':configs[0]['batch'],'n_eval':configs[0]['n_eval']}
    groups.append(row)
csv_write('verified_group_conditions_100k.csv',groups)

# Keep all intervention summaries at the same checkpoint; never select a
# diagnostic by lexicographic filename order or mix a 200k pilot into 100k.
diagnostics={}; log_metrics={}; provenance=[]
for family,variants in [('su2',['su2','so3']),('su3',['su3','psu3']),
                        ('spin4',['spin4','so4','leftq','pso4'])]:
    for v in variants:
        ds=[];ls=[]
        for seed in range(5):
            run=Path('data')/family/f'{v}_N2048_f0.01_s{seed}'
            path=WORK/'selected_checkpoints'/run/'step_100000'/'causal_100000.json'
            d=json.loads(path.read_text());assert d['step']==100000
            ds.append(d)
            ls.append(next(r for r in reversed([json.loads(x) for x in
                (SRC/run/'log.jsonl').read_text().splitlines()]) if r['step']==100000))
            provenance.append({'variant':v,'seed':seed,'step':100000,
                'causal_path':str(path.relative_to(ROOT)),
                'log_path':str((SRC/run/'log.jsonl').relative_to(ROOT))})
        diagnostics[v]={'interventions':{name:{k:stat([d['interventions'][name][k] for d in ds])
            for k in fields} for name,fields in ds[0]['interventions'].items()
            if all(name in d['interventions'] for d in ds)}}
        matched=json.loads((OUT/'rank_matched_ablation.json').read_text())
        matched_rows=[r for r in matched['records'] if r['variant']==v]
        if matched_rows:
            assert len(matched_rows)==5
            diagnostics[v]['interventions']['ablate_random_selected_rank']={
                k:stat([r[k] for r in matched_rows]) for k in ['task_acc','center_disc']}
        if 'character_fit' in ds[0]:
            diagnostics[v]['character_fit']={k:stat([d['character_fit'][k] for d in ds])
                for k,value in ds[0]['character_fit'].items() if isinstance(value,(int,float))}
        log_metrics[v]={k:stat([r[k] for r in ls]) for k,value in ls[0].items()
            if isinstance(value,(int,float)) and k not in ('step','wall_s')}
(OUT/'verified_diagnostics_100k.json').write_text(json.dumps({
    'step':100000,'fraction':0.01,'diagnostics':diagnostics,'log_metrics':log_metrics,
    'sd_convention':'population','provenance':provenance},indent=2))

audit=json.loads((OUT/'handoff_checkpoint_audit.json').read_text());cost={}
for arm in ['frame','block']:
    rs=[r for r in audit['records'] if r['step']==60000 and r['run'].startswith(f'data/su2/so3_so3{arm}_f0.01_s')]
    cost[arm]={'total_norm':stat([r['norms']['total'] for r in rs]),
               'accuracy':stat([r['log_at_step']['acc_T2'] for r in rs]),
               'margin':stat([r['log_at_step']['margin_T2'] for r in rs]),
               'margin_per_norm':stat([r['log_at_step']['margin_T2']/r['norms']['total'] for r in rs])}
cost['block_norm_reduction_fraction']=1-cost['block']['total_norm']['mean']/cost['frame']['total_norm']['mean']
cost['margin_per_norm_gain_fraction']=cost['block']['margin_per_norm']['mean']/cost['frame']['margin_per_norm']['mean']-1
costcells=[]
for path in sorted((WORK/'summary').glob('p1b_reanalysis_*.json')):
    d=json.loads(path.read_text())['results'];s=d['paired_statistics']['margin_at_norm_budget']
    costcells.append({'cell':d['cell'],'mean_margin_advantage':s['mean'],'one_sided_p':s['p_one_sided'],
        'paired_differences':s['paired_diffs_block_minus_frame'],
        'all_final_accuracies_clear_floor':all(v['floor_ok'] for v in d['per_seed'].values())})
report={'mainline_100k':main,'groups_100k':groups,'mechanisms_100k':mechanisms,
        'translations_100k':translations,'fusion_100k':fusion,'cost_base_60k':cost,'cost_cells':costcells,
        'sd_convention':'population standard deviation across the five observed seeds, matching the original figure script',
        'source':'raw training logs for FVE/T2 and uniform100k reruns for causal/mechanism/translation/fusion; checkpoint weights independently validated'}
(OUT/'verified_summary.json').write_text(json.dumps(report,indent=2))
print('Wrote verified_summary.json and canonical CSV tables')
print('Cost norm reduction',cost['block_norm_reduction_fraction'],'margin/norm gain',cost['margin_per_norm_gain_fraction'])
