'''
ClusterCommit replication on JIT-Defects4J (Apache Commons-style projects).

Paper: Shehab, Hamou-Lhadj, Alawneh.
       "ClusterCommit: A Just-in-Time Defect Prediction Approach Using
        Clusters of Projects", SANER 2022.

Methodology reproduced:
  1. Project Clustering
     - Parse each project's Maven pom.xml to extract (groupId:artifactId) deps.
     - Build a bipartite community graph G=(V,E): V = projects + libraries,
       E = {(project, library) | project uses library}.
     - Apply the Label Propagation (LP) algorithm (Raghavan et al.) to find
       communities. Projects that share many libraries end up in the same
       community. Implemented via networkx.algorithms.community.
     - ant-ivy has no pom.xml in ./mavens; it is placed in its own singleton
       cluster (documented deviation forced by missing input data).
  2. Feature Extraction
     - Kamei's 14 JIT expert features. The paper drops SEXP because it is
       >70% correlated with EXP/REXP. We do the same -> 13 features.
  3. Classifier
     - Random Forest (sklearn RandomForestClassifier, class_weight='balanced'
       because the dataset is imbalanced ~8.5% buggy).
  4. Validation
     - The paper uses time-based validation (6-month train, gap=min avg fix
       time, test=gap length). Our datasets are already pre-split into
       Train/Valid/Test sets, so we train on TrainDatasets.xlsx and test on
       TestDatasets.xlsx (same protocol as PCC.py for comparability). This is
       a documented deviation forced by the available data.
     - For each cluster: train ONE model on the combined commits of all
       projects in the cluster; test each project individually against that
       model (per-project metrics) and also compute cluster-overall metrics.
  5. Metrics
     - F1, Gmean, MCC  (paper's MCC + gmean for class-imbalance view)
     - Effort-aware (effort = la + ld = LOC):
         Recall@20%Effort, Effort@20%Recall, Popt
       (EffortAwareMetrics reused verbatim from PCC.py.)

Outputs (written to ./cc_output/):
  - predictions.csv            : per-commit true/pred labels, prob, cluster
  - per_project_metrics.csv    : per-project F1/Gmean/MCC/effort-aware metrics
  - cluster_summary.txt        : clustering result + overall/cluster metrics
  - metrics printed to console
'''

import os
import re
import math
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from lxml import etree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, auc as sk_auc,
                             matthews_corrcoef)


# ============================ configuration ============================
TRAIN_PATH = 'TrainDatasets.xlsx'
TEST_PATH = 'TestDatasets.xlsx'
MAVEN_DIR = 'mavens'
OUTPUT_DIR = 'cc_output'

# Kamei's 14 features minus SEXP (paper drops SEXP, >70% correlated w/ EXP/REXP)
FEATURES = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev", "lt",
            "nuc", "age", "exp", "rexp", "fix"]   # 13 features
LABEL_COL = "is_buggy_commit"
PROJECT_COL = "project"

RANDOM_STATE = 42
RF_N_ESTIMATORS = 100


# ============================ Maven dependency parsing ============================
def _strip_ns(tree):
    '''Remove XML namespaces so xpath can use plain local tag names.'''
    root = tree.getroot()
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith('{'):
            el.tag = etree.QName(el).localname
    # also clear nsmap so newly created/queried tags stay clean
    for el in root.iter():
        if el.nsmap:
            el.nsmap.clear() if hasattr(el.nsmap, 'clear') else None
    return tree


def parse_pom_dependencies(pom_path):
    '''Return set of "groupId:artifactId" strings used by this pom.

    Captures <dependencies>/<dependency> (incl. <dependencyManagement>) and
    any nested <dependency> blocks. The version is ignored on purpose: the
    paper links projects to *libraries*, not library versions.
    '''
    deps = set()
    try:
        tree = etree.parse(pom_path)
        tree = _strip_ns(tree)
    except Exception as e:
        print(f'  [WARN] failed to parse {pom_path}: {e}')
        return deps
    for dep in tree.xpath('//dependency'):
        gid = (dep.findtext('groupId') or '').strip()
        aid = (dep.findtext('artifactId') or '').strip()
        if gid and aid:
            deps.add(f'{gid}:{aid}')
    return deps


def discover_project_poms(maven_dir):
    '''Map project_name -> pom_path for every *-pom.xml in maven_dir.

    The dataset uses project names like "commons-math", "parquet-mr", "gora";
    the pom files are named "commons-math-pom.xml", etc. We strip the "-pom"
    suffix to recover the dataset project name.
    '''
    mapping = {}
    for fn in sorted(os.listdir(maven_dir)):
        m = re.match(r'^(.+)-pom\.xml$', fn)
        if not m:
            continue
        project_name = m.group(1)
        mapping[project_name] = os.path.join(maven_dir, fn)
    return mapping


# ============================ clustering ============================
def build_dependency_graph(project_deps):
    '''Build a bipartite community graph.

    Nodes: project names + "lib:<groupId:artifactId>" library ids.
    Edges: undirected edge (project, library) if the project depends on it.

    Undirected edges are used because the LP implementation in networkx
    operates on undirected graphs (matches the paper's community-graph idea:
    two projects that share many libraries will be pulled into one community).
    '''
    G = nx.Graph()
    for proj, deps in project_deps.items():
        G.add_node(proj, type='project')
        for d in deps:
            lib_node = f'lib:{d}'
            G.add_node(lib_node, type='library')
            G.add_edge(proj, lib_node)
    return G


def run_label_propagation(G):
    '''Run LP and return list of communities (each a set of node names).

    Uses networkx's asynchronous LPA. Communities are returned deterministically
    given the same graph (we seed by sorting the node iteration through the
    graph's node order, which is insertion-stable).
    '''
    communities = list(nx.algorithms.community.label_propagation_communities(G))
    return communities


def project_clusters_from_communities(communities, all_projects, projects_without_pom):
    '''Map every project to a cluster id.

    - Projects that appear in some LP community -> assigned to that community.
    - Projects with no pom.xml (in projects_without_pom) -> singleton cluster,
      one cluster per such project (cannot share libraries w/ anyone).
    Returns: list of (cluster_id, [project names]).
    '''
    clusters = []
    seen = set()
    for comm in communities:
        # keep only project nodes
        proj_in_comm = sorted(n for n in comm if not n.startswith('lib:'))
        if not proj_in_comm:
            continue
        clusters.append(proj_in_comm)
        seen.update(proj_in_comm)

    # any project not seen + projects_without_pom -> own singleton cluster
    for p in sorted(all_projects):
        if p not in seen:
            clusters.append([p])
    return clusters


# ============================ model ============================
def make_rf():
    '''Random Forest classifier. class_weight=balanced because ~8.5% buggy.'''
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_model_safe(X, y, name=''):
    '''Train RF; return None if training data has only one class.'''
    if len(np.unique(y)) < 2:
        print(f'  [skip] {name}: only one class in training data '
              f'(labels={np.unique(y)}), no model built.')
        return None
    m = make_rf()
    m.fit(X, y)
    return m


def predict_proba_buggy(model, X):
    '''Return P(label==buggy) for each row.'''
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


# ============================ effort-aware metrics ============================
# Verbatim from PCC.py (same EffortAwareMetrics) so results are directly
# comparable to the PCC baseline. Adds MCC on top of PCC's F1/Gmean/effort set.
class EffortAwareMetrics:
    def get_recall_at_k_percent_effort(self, percent_effort, result_df_arg, n_real_buggy):
        cum_LOC_k_percent = (percent_effort / 100.0) * result_df_arg.iloc[-1]['cum_LOC']
        buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
        buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1.0]
        if n_real_buggy == 0:
            return 0.0
        return len(buggy_commit) / float(n_real_buggy)

    def eval(self, df_in: pd.DataFrame) -> dict:
        result_df = df_in.copy()
        pred = result_df['defective_commit_pred']
        y_test = result_df['label']

        _, _, f1, _ = precision_recall_fscore_support(
            y_test, pred, average='binary', zero_division=0)
        try:
            AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])
        except Exception:
            AUC = float('nan')

        result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']
        result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']

        result_df = result_df.sort_values(by='defect_density', ascending=False)
        actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
        actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

        result_df['cum_LOC'] = result_df['LOC'].cumsum()
        actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
        actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

        n_real_buggy = int((result_df['label'] == 1.0).sum())

        cum_LOC_20 = 0.2 * result_df.iloc[-1]['cum_LOC']
        buggy_line_20 = result_df[result_df['cum_LOC'] <= cum_LOC_20]
        buggy_commit_20 = buggy_line_20[buggy_line_20['label'] == 1.0]
        recall_at_20_effort = (len(buggy_commit_20) / float(n_real_buggy)) if n_real_buggy > 0 else 0.0

        if n_real_buggy > 0:
            real_buggy_commits = result_df[result_df['label'] == 1.0]
            buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * n_real_buggy))
            effort_at_20_recall = int(buggy_20_percent.iloc[-1]['cum_LOC']) / float(result_df.iloc[-1]['cum_LOC'])
        else:
            effort_at_20_recall = 0.0

        percent_effort_list, pred_recall_list, actual_recall_list, worst_recall_list = [], [], [], []
        for percent_effort in np.arange(10, 101, 10):
            percent_effort_list.append(percent_effort / 100.0)
            pred_recall_list.append(self.get_recall_at_k_percent_effort(percent_effort, result_df, n_real_buggy))
            actual_recall_list.append(self.get_recall_at_k_percent_effort(percent_effort, actual_result_df, n_real_buggy))
            worst_recall_list.append(self.get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df, n_real_buggy))
        auc_actual = sk_auc(percent_effort_list, actual_recall_list)
        auc_pred = sk_auc(percent_effort_list, pred_recall_list)
        auc_worst = sk_auc(percent_effort_list, worst_recall_list)
        denom = auc_actual - auc_worst
        p_opt = (1 - (auc_actual - auc_pred) / denom) if abs(denom) > 1e-12 else float('nan')

        rec_1 = recall_score(y_test, pred, average='binary', zero_division=0)
        rec_0 = recall_score(y_test, pred, pos_label=0, average='binary', zero_division=0)
        gmean = math.sqrt(max(rec_0, 0.0) * max(rec_1, 0.0))

        # MCC (paper's metric)
        try:
            mcc = matthews_corrcoef(y_test, pred)
        except Exception:
            mcc = float('nan')

        return {
            'f1': round(f1, 4),
            'auc': round(AUC, 4),
            'gmean': round(gmean, 4),
            'mcc': round(mcc, 4),
            'recall_at_20_percent_effort': round(recall_at_20_effort, 4),
            'effort_at_20_percent_LOC_recall': round(effort_at_20_recall, 4),
            'p_opt': round(p_opt, 4),
        }


def build_result_df(y_true, y_pred, y_prob, loc_arr):
    return pd.DataFrame({
        'label': y_true.astype(float),
        'defective_commit_pred': y_pred.astype(float),
        'defective_commit_prob': y_prob.astype(float),
        'LOC': loc_arr.astype(float),
    })


# ============================ main pipeline ============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('=== Loading data ===')
    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)
    print(f'  train: {len(train_df)} rows | test: {len(test_df)} rows')
    print(f'  train label dist: {dict(Counter(train_df[LABEL_COL]))}')
    print(f'  test  label dist: {dict(Counter(test_df[LABEL_COL]))}')

    all_projects = sorted(train_df[PROJECT_COL].unique())
    print(f'  projects in dataset: {len(all_projects)} -> {all_projects}')

    # ---------- 1. parse Maven pom.xml ----------
    print('\n=== 1. Parsing Maven pom.xml ===')
    pom_map = discover_project_poms(MAVEN_DIR)
    print(f'  found {len(pom_map)} pom files: {sorted(pom_map)}')
    projects_without_pom = sorted(set(all_projects) - set(pom_map.keys()))
    print(f'  projects WITHOUT pom.xml (own singleton cluster): {projects_without_pom}')

    project_deps = {}
    for proj, path in pom_map.items():
        deps = parse_pom_dependencies(path)
        project_deps[proj] = deps
        print(f'  {proj:<22} {len(deps):>4} distinct libraries')

    # ---------- 2. cluster via LP ----------
    print('\n=== 2. Building dependency graph + Label Propagation ===')
    G = build_dependency_graph(project_deps)
    n_proj_nodes = sum(1 for n in G.nodes if G.nodes[n].get('type') == 'project')
    n_lib_nodes = sum(1 for n in G.nodes if G.nodes[n].get('type') == 'library')
    print(f'  graph: {G.number_of_nodes()} nodes ({n_proj_nodes} projects, '
          f'{n_lib_nodes} libraries), {G.number_of_edges()} edges')
    communities = run_label_propagation(G)
    clusters = project_clusters_from_communities(communities, all_projects, projects_without_pom)
    print(f'  LP returned {len(communities)} communities -> {len(clusters)} project clusters:')
    for i, c in enumerate(clusters):
        tag = '(singleton, no pom)' if len(c) == 1 and c[0] in projects_without_pom else ''
        print(f'    cluster {i}: {c} {tag}')

    # save cluster mapping
    proj2cluster = {}
    for i, c in enumerate(clusters):
        for p in c:
            proj2cluster[p] = i
    with open(os.path.join(OUTPUT_DIR, 'clusters.json'), 'w', encoding='utf-8') as f:
        json.dump({'clusters': clusters, 'project_to_cluster': proj2cluster}, f, indent=2)
    print(f'  saved cluster mapping -> {os.path.join(OUTPUT_DIR, "clusters.json")}')

    # ---------- 3. train RF per cluster, test per project ----------
    print('\n=== 3. Training RF per cluster + predicting ===')
    # assemble feature matrices (no BoW/AST: ClusterCommit uses only Kamei 13)
    def feat_matrix(df):
        X = df[FEATURES].copy()
        X['fix'] = X['fix'].astype(int)   # bool -> int
        return X.values.astype(np.float64)

    X_train_all = feat_matrix(train_df)
    y_train_all = train_df[LABEL_COL].astype(int).values
    X_test_all = feat_matrix(test_df)
    y_test_all = test_df[LABEL_COL].astype(int).values

    la_test = test_df['la'].astype(float).values
    ld_test = test_df['ld'].astype(float).values
    loc_test = la_test + ld_test

    cluster_models = {}
    for ci, cluster_projects in enumerate(clusters):
        train_mask = train_df[PROJECT_COL].isin(cluster_projects).values
        Xc = X_train_all[train_mask]
        yc = y_train_all[train_mask]
        print(f'  cluster {ci} {cluster_projects}: train rows={len(Xc)} buggy={int(yc.sum())}')
        cluster_models[ci] = train_model_safe(Xc, yc, name=f'cluster_{ci}')

    # predict: every test row -> its project's cluster model
    n = len(test_df)
    pred_label = np.zeros(n, dtype=int)
    pred_prob = np.zeros(n, dtype=float)
    cluster_used = np.full(n, -1, dtype=int)
    for i in range(n):
        proj = test_df[PROJECT_COL].iloc[i]
        ci = proj2cluster[proj]
        cluster_used[i] = ci
        model = cluster_models.get(ci)
        if model is not None:
            p = predict_proba_buggy(model, X_test_all[i:i+1])[0]
            pred_prob[i] = p
            pred_label[i] = int(p >= 0.5)
        else:
            pred_prob[i] = 0.0
            pred_label[i] = 0

    # ---------- 4. save predictions ----------
    pred_df = pd.DataFrame({
        'commit_hash': test_df['commit_hash'].astype(str).values,
        'project': test_df[PROJECT_COL].astype(str).values,
        'cluster': cluster_used,
        'true_label': y_test_all,
        'pred_label': pred_label,
        'pred_prob_buggy': pred_prob,
        'la': la_test,
        'ld': ld_test,
        'LOC': loc_test,
    })
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f'\n  saved predictions -> {pred_path}')

    # ---------- 5. metrics ----------
    def per_project_metrics(project_name):
        mask = (test_df[PROJECT_COL].values == project_name)
        y_t = y_test_all[mask]
        y_p = pred_label[mask]
        y_pr = pred_prob[mask]
        loc = loc_test[mask]
        if len(y_t) == 0:
            return None
        ea = EffortAwareMetrics().eval(build_result_df(y_t, y_p, y_pr, loc))
        try:
            mcc = matthews_corrcoef(y_t, y_p)
        except Exception:
            mcc = float('nan')
        return {
            'project': project_name,
            'cluster': proj2cluster[project_name],
            'n_test': int(mask.sum()),
            'n_buggy': int(y_t.sum()),
            'precision': round(precision_score(y_t, y_p, zero_division=0), 4),
            'recall': round(recall_score(y_t, y_p, zero_division=0), 4),
            'f1': ea['f1'],
            'gmean': ea['gmean'],
            'mcc': round(mcc, 4),
            'recall_at_20_percent_effort': ea['recall_at_20_percent_effort'],
            'effort_at_20_percent_LOC_recall': ea['effort_at_20_percent_LOC_recall'],
            'p_opt': ea['p_opt'],
            'auc': ea['auc'],
        }

    print('\n=== 4. Metrics ===')
    # overall
    ea_overall = EffortAwareMetrics().eval(
        build_result_df(y_test_all, pred_label, pred_prob, loc_test))
    try:
        mcc_overall = matthews_corrcoef(y_test_all, pred_label)
    except Exception:
        mcc_overall = float('nan')
    overall_metrics = {
        'name': 'OVERALL',
        'n': n,
        'f1': ea_overall['f1'],
        'gmean': ea_overall['gmean'],
        'mcc': round(mcc_overall, 4),
        'recall_at_20_percent_effort': ea_overall['recall_at_20_percent_effort'],
        'effort_at_20_percent_LOC_recall': ea_overall['effort_at_20_percent_LOC_recall'],
        'p_opt': ea_overall['p_opt'],
        'auc': ea_overall['auc'],
    }

    # per-cluster
    cluster_metrics = []
    for ci, cluster_projects in enumerate(clusters):
        mask = np.isin(cluster_used, [ci])
        if mask.sum() == 0:
            continue
        y_t = y_test_all[mask]; y_p = pred_label[mask]
        y_pr = pred_prob[mask]; loc = loc_test[mask]
        ea = EffortAwareMetrics().eval(build_result_df(y_t, y_p, y_pr, loc))
        try:
            mcc = matthews_corrcoef(y_t, y_p)
        except Exception:
            mcc = float('nan')
        cluster_metrics.append({
            'name': f'CLUSTER-{ci}',
            'projects': cluster_projects,
            'n': int(mask.sum()),
            'f1': ea['f1'],
            'gmean': ea['gmean'],
            'mcc': round(mcc, 4),
            'recall_at_20_percent_effort': ea['recall_at_20_percent_effort'],
            'effort_at_20_percent_LOC_recall': ea['effort_at_20_percent_LOC_recall'],
            'p_opt': ea['p_opt'],
            'auc': ea['auc'],
        })

    # per-project
    per_proj_rows = []
    for p in all_projects:
        r = per_project_metrics(p)
        if r is not None:
            per_proj_rows.append(r)
    per_proj_df = pd.DataFrame(per_proj_rows)
    per_proj_path = os.path.join(OUTPUT_DIR, 'per_project_metrics.csv')
    per_proj_df.to_csv(per_proj_path, index=False)
    print(f'  saved per-project metrics -> {per_proj_path}')

    # ---------- summary ----------
    summary = []
    summary.append('===== ClusterCommit Replication Summary =====\n')
    summary.append(f'Train size: {len(train_df)} | Test size: {len(test_df)}')
    summary.append(f'Projects: {len(all_projects)} | Features (Kamei - SEXP): {len(FEATURES)}')
    summary.append(f'Classifier: RandomForest (n_estimators={RF_N_ESTIMATORS}, class_weight=balanced)')
    summary.append('')
    summary.append('--- Clustering (Label Propagation on Maven deps) ---')
    summary.append(f'Projects with pom.xml: {len(pom_map)} | '
                   f'without pom (singleton): {len(projects_without_pom)} -> {projects_without_pom}')
    summary.append(f'Graph: {n_proj_nodes} project nodes, {n_lib_nodes} library nodes, '
                   f'{G.number_of_edges()} edges')
    summary.append(f'#clusters: {len(clusters)}')
    for i, c in enumerate(clusters):
        tag = '  (singleton, no pom.xml -> own cluster)' if len(c) == 1 and c[0] in projects_without_pom else ''
        summary.append(f'  cluster {i}: {c}{tag}')
    summary.append('')
    summary.append('--- Overall metrics (all test commits) ---')
    summary.append(f'  [OVERALL] n={overall_metrics["n"]} '
                   f'F1={overall_metrics["f1"]} Gmean={overall_metrics["gmean"]} '
                   f'MCC={overall_metrics["mcc"]} '
                   f'Recall@20%Effort={overall_metrics["recall_at_20_percent_effort"]} '
                   f'Effort@20%Recall={overall_metrics["effort_at_20_percent_LOC_recall"]} '
                   f'Popt={overall_metrics["p_opt"]} (AUC={overall_metrics["auc"]})')
    summary.append('')
    summary.append('--- Per-cluster metrics ---')
    for m in cluster_metrics:
        summary.append(f'  [{m["name"]}] n={m["n"]} projects={m["projects"]}')
        summary.append(f'      F1={m["f1"]} Gmean={m["gmean"]} MCC={m["mcc"]} '
                       f'Recall@20%Effort={m["recall_at_20_percent_effort"]} '
                       f'Effort@20%Recall={m["effort_at_20_percent_LOC_recall"]} '
                       f'Popt={m["p_opt"]} (AUC={m["auc"]})')
    summary.append('')
    summary.append('--- Per-project metrics (ClusterCommit tests each project individually) ---')
    for r in per_proj_rows:
        summary.append(
            f'  {r["project"]:<22} cluster={r["cluster"]} n={r["n_test"]:<5} '
            f'buggy={r["n_buggy"]:<4} P={r["precision"]} R={r["recall"]} '
            f'F1={r["f1"]} Gmean={r["gmean"]} MCC={r["mcc"]} '
            f'R@20%E={r["recall_at_20_percent_effort"]} '
            f'E@20%R={r["effort_at_20_percent_LOC_recall"]} '
            f'Popt={r["p_opt"]} (AUC={r["auc"]})')
    # averages
    avg_f1 = round(per_proj_df['f1'].mean(), 4)
    avg_gmean = round(per_proj_df['gmean'].mean(), 4)
    avg_mcc = round(per_proj_df['mcc'].mean(), 4)
    avg_r20 = round(per_proj_df['recall_at_20_percent_effort'].mean(), 4)
    avg_e20 = round(per_proj_df['effort_at_20_percent_LOC_recall'].mean(), 4)
    avg_popt = round(per_proj_df['p_opt'].mean(), 4)
    summary.append('')
    summary.append(f'  [PROJECT-AVG] F1={avg_f1} Gmean={avg_gmean} MCC={avg_mcc} '
                   f'Recall@20%Effort={avg_r20} Effort@20%Recall={avg_e20} Popt={avg_popt}')

    summary_text = '\n'.join(summary)
    print('\n' + summary_text)
    sum_path = os.path.join(OUTPUT_DIR, 'cluster_summary.txt')
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f'\n  saved cluster summary -> {sum_path}')


if __name__ == '__main__':
    main()
