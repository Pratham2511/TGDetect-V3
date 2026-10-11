"""Run pinned upstream preprocessing against a restored full CADETS database.

Database settings come from ORTHRUS_DB_HOST/PORT/USER/PASSWORD/NAME.
Only runtime configuration is overridden; upstream source files are unmodified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def prepare(output):
    root=Path(__file__).resolve().parents[1]/'orthrus_official'
    sys.path.insert(0,str(root/'src'))
    import config
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    config.ROOT_ARTIFACT_DIR=str(out/'upstream')+'/'
    # The pinned submodule contains a darpa/ directory inside its checkout.
    config.ROOT_GROUND_TRUTH_DIR=str(root/'Ground_Truth'/'darpa')
    config.DATABASE_DEFAULT_CONFIG.update(host=os.getenv('ORTHRUS_DB_HOST','127.0.0.1'),
        port=os.getenv('ORTHRUS_DB_PORT','5432'),user=os.getenv('ORTHRUS_DB_USER','postgres'),
        password=os.getenv('ORTHRUS_DB_PASSWORD',''))
    name=os.getenv('ORTHRUS_DB_NAME','cadets_e3')
    config.DATASET_DEFAULT_CONFIG['CADETS_E3'].update(database=name,database_all_file=name)
    args=config.get_runtime_required_args(args=['CADETS_E3','--cpu',
        '--detection.gnn_training.num_epochs=20',
        '--detection.gnn_training.encoder.graph_attention.dropout=0.25',
        '--detection.evaluation.node_evaluation.kmeans_top_K=30'])
    cfg=config.get_yml_cfg(args)
    for filename in cfg.dataset.ground_truth_relative_path:
        if not (Path(cfg._ground_truth_dir)/filename).is_file():
            raise FileNotFoundError(Path(cfg._ground_truth_dir)/filename)
    from provnet_utils import init_database_connection
    from .protocol import require_coverage
    cur,connection=init_database_connection(cfg)
    cur.execute("SELECT to_char(to_timestamp(timestamp_rec/1e9) AT TIME ZONE 'America/New_York', 'YYYY-MM-DD'), count(*) FROM event_table GROUP BY 1 ORDER BY 1")
    raw_counts=dict(cur.fetchall())
    days=[int(v[-2:]) for v in raw_counts if v.startswith('2018-04-')];require_coverage(days)
    # The pinned upstream release uses positional rows. Fail explicitly if a
    # newer public dump has changed that schema rather than misreading columns.
    expected={
        'subject_node_table':['node_uuid','hash_id','path','cmd','index_id'],
        'file_node_table':['node_uuid','hash_id','path','index_id'],
        'netflow_node_table':['node_uuid','hash_id','src_addr','src_port','dst_addr','dst_port','index_id'],
        'event_table':['src_node','src_index_id','operation','dst_node','dst_index_id','event_uuid','timestamp_rec','_id']}
    schemas={}
    for table, columns in expected.items():
        cur.execute('SELECT * FROM '+table+' LIMIT 0')
        schemas[table]=[v.name for v in cur.description]
        if schemas[table] != columns:
            raise ValueError(f'Upstream database schema mismatch for {table}: {schemas[table]}; expected {columns}')
    from preprocessing import build_orthrus_graphs
    from featurization import build_feature_word2vec,embed_edges_feature_word2vec
    import wandb
    wandb.init(mode='disabled')
    for name,stage in [('graphs',build_orthrus_graphs),('word2vec',build_feature_word2vec),('features',embed_edges_feature_word2vec)]:
        marker=out/(name+'.complete')
        if not marker.exists():
            stage.main(cfg);marker.write_text('complete\n')
    attrs={}
    for table,kind in [('subject_node_table','subject'),('file_node_table','file'),('netflow_node_table','netflow')]:
        cur.execute('SELECT * FROM '+table)
        for row in cur.fetchall():
            if kind=='subject': item=dict(type=kind,path=row[2],cmd_line=row[3])
            elif kind=='file': item=dict(type=kind,path=row[2])
            else:
                item=dict(type=kind,remote_ip=row[4])
                if cfg.featurization.embed_nodes.feature_word2vec.use_port:item['remote_port']=row[5]
            attrs[str(row[-1])]=item
    attrpath=out/'node_attributes.json';attrpath.write_text(json.dumps(attrs))
    from labelling import get_ground_truth,get_GP_of_each_attack
    truth,_,_=get_ground_truth(cfg)
    gtpath=out/'ground_truth_nodes.json';gtpath.write_text(json.dumps(sorted(truth)))
    attackpath=out/'attack_nodes.json'
    attackpath.write_text(json.dumps({str(cfg.dataset.ground_truth_relative_path[i]):sorted(nodes)
                                     for i,nodes in get_GP_of_each_attack(cfg).items()},indent=2))
    folder=Path(cfg.featurization.embed_edges._edge_embeds_dir)
    paths={split:[str(p) for p in sorted((folder/split).iterdir()) if p.is_file()] for split in ('train','val','test')}
    import torch
    from .run import sha256
    stats={}; digests={str(p):sha256(p) for p in [attrpath,gtpath,attackpath]}
    graph_paths={}
    for split, items in paths.items():
        nodes=set(); edges=0
        graph_paths[split]=[]
        for path in items:
            digests[path]=sha256(path)
            graph=torch.load(path,map_location='cpu',weights_only=False)
            nodes.update(graph.src.tolist());nodes.update(graph.dst.tolist());edges+=len(graph.src)
            filename=Path(path).name.removesuffix('.TemporalData.simple')
            original=Path(cfg.preprocessing.build_graphs._graphs_dir)/('graph_'+str(int(filename[8:10])))/filename
            if not original.is_file(): raise FileNotFoundError(original)
            graph_paths[split].append(str(original));digests[str(original)]=sha256(original)
        stats[split]=dict(nodes=len(nodes),edges=edges,windows=len(items),ground_truth_nodes=len(nodes & truth))
    manifest=dict(dataset='CADETS_E3',days=days,official_preprocessing=True,files=paths,
        n_nodes=max(map(int,attrs))+1,node_attributes=str(attrpath),ground_truth_nodes=str(gtpath),
        upstream_commit='d26613684b83d12e5d1f4bc51283b7de96124776',
        features='upstream Word2Vec128 + node type3 per endpoint + relation10',
        attribute_contract=['type','path','cmd_line','remote_ip'],
        word2vec_scope='all database node attributes, as in pinned release; transductive',
        split_timezone='America/New_York',raw_counts_by_date=raw_counts,table_schemas=schemas,
        split_statistics=stats,file_sha256=digests,graph_files=graph_paths,attack_nodes=str(attackpath),
        paper_test_nodes=268153,paper_ground_truth_nodes=68)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Config may contain a user-supplied DB password: save only noncredential sections.
    (out/'model_config.yml').write_text(cfg.detection.dump())
    connection.close();return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    prepare(p.parse_args().output)
