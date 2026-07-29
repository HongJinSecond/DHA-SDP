'''
PCC (Personalized Defect Prediction) replication on JIT-Defects4J.

Paper: Jiang, Tan, Kim. "Personalized Defect Prediction", ASE 2013.

Deviation from the original paper (forced by the dataset):
  The original PCC builds AST characteristic vectors from the FULL source file
  before/after a change (via Deckard). JIT-Defects4J only keeps the added/removed
  *lines* of each commit, so we approximate the characteristic vector directly
  on the diff:
    - char(added_lines)  and char(removed_lines)  via tree-sitter (Java)
    - diff = char(added) - char(removed)   ~  paper's "difference" feature set
    - char(added)                          ~  weak proxy for paper's "after" set
  The full-file "after" vector cannot be reconstructed and is substituted by
  char(added) as a documented approximation.

Feature groups:
  1) AST characteristic vectors  -> [diff, added]   (tree-sitter, Java grammar)
  2) Bag-of-words                -> TF on added lines + TF on removed lines
                                    (Snowball stemmer, like the paper's Weka BoW)
  3) Metadata                    -> Kamei's 14 expert features (StandardScaler)

Model:
  ADTree is approximated by AdaBoost + decision stumps
  (sklearn AdaBoostClassifier, base=DecisionTreeClassifier(max_depth=1)),
  which is the standard ADTree analogue and exposes predict_proba.

Personalization (per user spec):
  - Authors with > PERSONAL_THRESHOLD (100) training samples -> a personal model.
  - All remaining training data (small authors) -> one Global model.
  - At test time: route to the author's personal model if one exists,
    otherwise to the Global model.

Outputs (written to ./pcc_output/):
  - predictions.csv            : true/pred labels, pred prob, routing info
  - personal_models_summary.csv: per-author personal-model stats
  - routing_summary.txt        : human-readable routing log + metrics
  - metrics printed to console
'''

import os
import re
import math
import ast as _ast
import numpy as np
import pandas as pd
from collections import Counter

from tree_sitter import Language, Parser
from tree_sitter_java import language as java_language
from snowballstemmer import stemmer as _snowball

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, auc as sk_auc)


# ============================ configuration ============================
TRAIN_PATH = 'TrainDatasets.xlsx'
TEST_PATH = 'TestDatasets.xlsx'
OUTPUT_DIR = 'pcc_output'

EXPERT_FEATURES = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev", "lt",
                   "nuc", "age", "exp", "rexp", "sexp", "fix"]
LABEL_COL = "is_buggy_commit"
AUTHOR_COL = "author_name"
ADDED_COL = "added lines"
REMOVED_COL = "removed lines"

PERSONAL_THRESHOLD = 100      # an author needs > this many train samples to get a personal model
RANDOM_STATE = 42
N_BOOSTING_ITER = 50          # ADTree's single hyperparameter = #boosting iterations
BOW_MIN_DF = 5                # ignore tokens appearing in < min_df train docs
BOW_MAX_FEAT = 1000           # cap vocabulary per side (added / removed)


# ============================ AST characteristic vectors ============================
# tree-sitter-java uses snake_case node type names (not CamelCase).
NODE_TYPES = [
    # control-flow statements
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_statement", "try_statement", "catch_clause", "throw_statement",
    "return_statement", "break_statement", "continue_statement",
    "assert_statement", "synchronized_statement",
    # declarations
    "method_declaration", "constructor_declaration", "class_declaration",
    "interface_declaration", "enum_declaration", "field_declaration",
    "local_variable_declaration",
    # expressions
    "method_invocation", "object_creation_expression", "array_access",
    "array_creation_expression", "binary_expression", "update_expression",
    "assignment_expression", "ternary_expression", "cast_expression",
    "instanceof_expression", "lambda_expression",
    # other
    "annotation",
]
_NODE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
_N_AST = len(NODE_TYPES)

# The dataset stores code already whitespace-tokenized with operators split
# (e.g. "==", "<=", "&&"). tree-sitter cannot parse those, so we re-merge them.
_OP_FIXES = [
    ("> > >", ">>>"), ("< <", "<<"), ("> >", ">>"),
    ("= =", "=="), ("! =", "!="), ("< =", "<="), ("> =", ">="),
    ("+ =", "+="), ("- =", "-="), ("* =", "*="), ("/ =", "/="),
    ("% =", "%="), ("& =", "&="), ("| =", "|="), ("^ =", "^="),
    ("& &", "&&"), ("| |", "||"), ("+ +", "++"), ("- -", "--"),
    ("- >", "->"),
]

_JAVA_LANG = Language(java_language())
_ts_parser = Parser(_JAVA_LANG)


def _normalize_code(text: str) -> str:
    '''Re-merge split operators so tree-sitter can parse the tokenized diff.'''
    for a, b in _OP_FIXES:
        text = text.replace(a, b)
    return text


def char_vector(code: str) -> np.ndarray:
    '''Count AST node types in a code fragment -> characteristic vector.

    tree-sitter's error recovery handles incomplete diff fragments; even when
    some nodes become ERROR nodes, the recognizable statement/expression nodes
    (if_statement, method_invocation, ...) are still counted, which is exactly
    the syntactic-structure signal the paper wants.
    '''
    vec = np.zeros(_N_AST, dtype=np.float64)
    if not code or not code.strip():
        return vec
    tree = _ts_parser.parse(bytes(code, "utf8"))
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        i = _NODE_IDX.get(n.type)
        if i is not None:
            vec[i] += 1.0
        stack.extend(n.children)
    return vec


def ast_features(added_lines, removed_lines) -> np.ndarray:
    '''Build AST feature = [diff, added-proxy] for one commit.

    diff  = char(added) - char(removed)   -> paper's "difference" feature set
    added = char(added)                   -> proxy for paper's "after" set
    '''
    add_code = _normalize_code("\n".join(added_lines))
    rem_code = _normalize_code("\n".join(removed_lines))
    a = char_vector(add_code)
    r = char_vector(rem_code)
    diff = a - r
    return np.concatenate([diff, a])   # length = 2 * _N_AST


# ============================ diff-line parsing ============================
# added/removed columns are stored as Python set-literal strings. Some rows
# contain un-escaped quotes that break ast.literal_eval, so we fall back to a
# regex that extracts the quoted string elements directly.
_QUOTED = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"")


def parse_lines(value) -> list:
    '''Parse a {set}/[list]-literal column value into a list of code strings.

    Lines are sorted for a canonical order. This matters because the column is
    stored as a Python set-literal: set iteration order is subject to Python's
    per-process string-hash randomization, which would otherwise change the
    line order of the joined code between runs and thus (via tree-sitter's
    error recovery) change AST node counts. Sorting makes runs reproducible.
    '''
    if not isinstance(value, str):
        return []
    try:
        obj = _ast.literal_eval(value)
        if isinstance(obj, (set, list, tuple)):
            return sorted(str(x) for x in obj)
        if isinstance(obj, str):
            return [obj]
        return []
    except Exception:
        pass
    # fallback: pull out every quoted string element
    out = []
    for m in _QUOTED.finditer(value):
        out.append(m.group(1) if m.group(1) is not None else m.group(2))
    return sorted(out)


# ============================ bag-of-words ============================
# Paper: Weka StringToWordVector + Snowball stemmer, term-frequency counts.
# We apply the same idea to added lines and removed lines separately, producing
# two TF vectors (added-BoW, removed-BoW) so the model sees what was inserted
# vs. deleted.
_ENGLISH_STEMMER = _snowball('english')
_WORD_RE = re.compile(r'[A-Za-z0-9_]+')


def _bow_tokenize(text: str) -> list:
    '''Lowercase, split on non-word chars, Snowball-stem, drop length-1 tokens.'''
    toks = []
    for t in _WORD_RE.findall(text.lower()):
        if len(t) < 2:
            continue
        toks.append(_ENGLISH_STEMMER.stemWord(t))
    return toks


def make_bow_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        tokenizer=_bow_tokenize,
        token_pattern=None,        # required when passing a custom tokenizer
        min_df=BOW_MIN_DF,
        max_features=BOW_MAX_FEAT,
        binary=False,              # TF counts (paper: "occurrence of each word")
    )


def lines_to_text(lines: list) -> str:
    return "\n".join(lines)


# ============================ feature assembly ============================
def build_feature_matrix(df, bow_add, bow_rem, scaler, fit_bow=False, fit_scaler=False):
    '''Compute the full feature matrix for a dataframe.

    Returns (X, y, authors, commit_hashes).
    bow_add/bow_rem/scaler are fit on this call when fit_* flags are True.
    '''
    n = len(df)
    # --- AST features ---
    ast_mat = np.empty((n, 2 * _N_AST), dtype=np.float64)
    added_lines_col = df[ADDED_COL].tolist()
    removed_lines_col = df[REMOVED_COL].tolist()
    for i in range(n):
        ast_mat[i] = ast_features(parse_lines(added_lines_col[i]),
                                  parse_lines(removed_lines_col[i]))
        if i % 3000 == 0 and i > 0:
            print(f'    AST progress: {i}/{n}')

    # --- BoW features ---
    add_texts = [lines_to_text(parse_lines(v)) for v in added_lines_col]
    rem_texts = [lines_to_text(parse_lines(v)) for v in removed_lines_col]
    if fit_bow:
        bow_add.fit(add_texts)
        bow_rem.fit(rem_texts)
    bow_add_mat = bow_add.transform(add_texts).toarray()
    bow_rem_mat = bow_rem.transform(rem_texts).toarray()

    # --- metadata (Kamei 14) ---
    meta = df[EXPERT_FEATURES].copy()
    meta['fix'] = meta['fix'].astype(int)          # bool -> int
    meta = meta.values.astype(np.float64)
    if fit_scaler:
        scaler.fit(meta)
    meta_mat = scaler.transform(meta)

    X = np.hstack([ast_mat, bow_add_mat, bow_rem_mat, meta_mat])
    y = df[LABEL_COL].astype(int).values
    authors = df[AUTHOR_COL].astype(str).values
    commits = df['commit_hash'].astype(str).values
    return X, y, authors, commits


# ============================ model ============================
def make_adtree():
    '''ADTree approximation: AdaBoost over decision stumps.

    ADTree (Freund & Mason) is boosting with stump-like tests; sklearn's
    AdaBoostClassifier with max_depth=1 trees is the standard analogue and
    provides predict_proba (used as the confidence/probability output).
    n_estimators corresponds to ADTree's number of boosting iterations.
    '''
    return AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        n_estimators=N_BOOSTING_ITER,
        random_state=RANDOM_STATE,
    )


def train_model_safe(X, y, name=''):
    '''Train an ADTree; return None if training data lacks both classes.'''
    if len(np.unique(y)) < 2:
        print(f'  [skip] {name}: only one class present in training data '
              f'(labels={np.unique(y)}), no personal model built.')
        return None
    m = make_adtree()
    m.fit(X, y)
    return m


def predict_proba_buggy(model, X) -> np.ndarray:
    '''Return P(label==buggy) for each row, mapping via the model's classes_.'''
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        j = classes.index(1)
        return proba[:, j]
    # buggy class never seen in training -> probability 0
    return np.zeros(len(X))


# ============================ effort-aware metrics ============================
# Effort = cumulative LOC (la + ld) of commits, ranked by predicted defect
# density (prob / LOC). This mirrors the reference PerformanceMeasure impl:
#   - Recall@20%Effort : fraction of buggy commits found within the top 20% LOC
#   - Effort@20%Recall : LOC fraction needed to recover the top 20% buggy commits
#   - P_opt            : optimality of the predicted ranking vs optimal/worst
class EffortAwareMetrics:
    def get_recall_at_k_percent_effort(self, percent_effort, result_df_arg, n_real_buggy):
        cum_LOC_k_percent = (percent_effort / 100.0) * result_df_arg.iloc[-1]['cum_LOC']
        buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
        buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1.0]
        if n_real_buggy == 0:
            return 0.0
        return len(buggy_commit) / float(n_real_buggy)

    def eval(self, df_in: pd.DataFrame) -> dict:
        '''df_in must have columns: label, defective_commit_pred, defective_commit_prob, LOC.'''
        result_df = df_in.copy()
        pred = result_df['defective_commit_pred']
        y_test = result_df['label']

        _, _, f1, _ = precision_recall_fscore_support(
            y_test, pred, average='binary', zero_division=0)
        try:
            AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])
        except Exception:
            AUC = float('nan')

        # defect density = predicted prob / LOC ; actual density = label / LOC
        result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']
        result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']

        # three rankings: predicted (by model density), optimal (by actual density desc),
        # worst (by actual density asc). cum_LOC computed within each ordering.
        result_df = result_df.sort_values(by='defect_density', ascending=False)
        actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
        actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

        result_df['cum_LOC'] = result_df['LOC'].cumsum()
        actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
        actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

        n_real_buggy = int((result_df['label'] == 1.0).sum())

        # Recall@20%Effort
        cum_LOC_20 = 0.2 * result_df.iloc[-1]['cum_LOC']
        buggy_line_20 = result_df[result_df['cum_LOC'] <= cum_LOC_20]
        buggy_commit_20 = buggy_line_20[buggy_line_20['label'] == 1.0]
        recall_at_20_effort = (len(buggy_commit_20) / float(n_real_buggy)) if n_real_buggy > 0 else 0.0

        # Effort@20%Recall: LOC fraction to recover the top-20% (highest density) buggy commits
        if n_real_buggy > 0:
            real_buggy_commits = result_df[result_df['label'] == 1.0]
            buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * n_real_buggy))
            effort_at_20_recall = int(buggy_20_percent.iloc[-1]['cum_LOC']) / float(result_df.iloc[-1]['cum_LOC'])
        else:
            effort_at_20_recall = 0.0

        # P_opt over 10%..100% effort
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

        # gmean = sqrt(recall_0 * recall_1)
        rec_1 = recall_score(y_test, pred, average='binary', zero_division=0)
        rec_0 = recall_score(y_test, pred, pos_label=0, average='binary', zero_division=0)
        gmean = math.sqrt(max(rec_0, 0.0) * max(rec_1, 0.0))

        return {
            'f1': round(f1, 4),
            'auc': round(AUC, 4),
            'gmean': round(gmean, 4),
            'recall_at_20_percent_effort': round(recall_at_20_effort, 4),
            'effort_at_20_percent_LOC_recall': round(effort_at_20_recall, 4),
            'p_opt': round(p_opt, 4),
        }


def build_result_df(y_true, y_pred, y_prob, loc_arr) -> pd.DataFrame:
    '''Assemble the dataframe expected by EffortAwareMetrics.eval.'''
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

    # ---- feature extraction (fit BoW + scaler on train only) ----
    print('=== Extracting features (fit on train) ===')
    bow_add = make_bow_vectorizer()
    bow_rem = make_bow_vectorizer()
    scaler = StandardScaler()
    X_train, y_train, authors_train, commits_train = build_feature_matrix(
        train_df, bow_add, bow_rem, scaler, fit_bow=True, fit_scaler=True)
    print(f'  train feature matrix: {X_train.shape}  '
          f'(AST={2*_N_AST}, BoW_add={len(bow_add.get_feature_names_out())}, '
          f'BoW_rem={len(bow_rem.get_feature_names_out())}, meta={len(EXPERT_FEATURES)})')

    print('=== Extracting features (test) ===')
    X_test, y_test, authors_test, commits_test = build_feature_matrix(
        test_df, bow_add, bow_rem, scaler, fit_bow=False, fit_scaler=False)
    print(f'  test feature matrix: {X_test.shape}')

    # ---- author -> row indices on training set ----
    train_author_counts = pd.Series(authors_train).value_counts()
    personal_candidates = train_author_counts[train_author_counts > PERSONAL_THRESHOLD]
    print(f'=== Personalization ===')
    print(f'  authors in train: {len(train_author_counts)} | '
          f'with > {PERSONAL_THRESHOLD} samples: {len(personal_candidates)}')

    # map author -> train indices (personal authors)
    train_idx_by_author = {}
    for idx, a in enumerate(authors_train):
        if a in personal_candidates.index:
            train_idx_by_author.setdefault(a, []).append(idx)

    # ---- train personal models ----
    personal_models = {}          # author -> fitted model
    personal_train_info = []      # rows for summary csv
    for author, idxs in train_idx_by_author.items():
        Xa = X_train[idxs]
        ya = y_train[idxs]
        n_buggy = int(ya.sum())
        model = train_model_safe(Xa, ya, name=f'author="{author}"')
        info = {
            'author': author,
            'n_train': len(idxs),
            'n_train_buggy': n_buggy,
            'has_personal_model': model is not None,
        }
        if model is None:
            info['skip_reason'] = 'single_class_in_training'
        else:
            info['skip_reason'] = ''
            personal_models[author] = model
        personal_train_info.append(info)
    print(f'  personal models actually built: {len(personal_models)} / '
          f'{len(personal_candidates)}')

    # ---- train Global model on the *remaining* (small-author) data ----
    personal_author_set = set(personal_candidates.index)
    global_mask = np.array([a not in personal_author_set for a in authors_train])
    global_idxs = np.where(global_mask)[0]
    X_global = X_train[global_idxs]
    y_global = y_train[global_idxs]
    print(f'  global model training samples (small authors): {len(global_idxs)} '
          f'(buggy={int(y_global.sum())})')
    global_model = train_model_safe(X_global, y_global, name='GLOBAL')
    if global_model is None:
        # extremely unlikely; fall back to predicting all-clean
        print('  [WARN] Global model could not be trained (single class). '
              'Will predict majority class.')

    # ---- route test predictions ----
    print('=== Predicting on test set ===')
    n = len(y_test)
    pred_label = np.zeros(n, dtype=int)
    pred_prob = np.zeros(n, dtype=float)
    used_model_type = []        # 'personal' / 'global'
    model_used_name = []        # author name or 'GLOBAL'

    test_author_counts = pd.Series(authors_test).value_counts()
    n_routed_personal = 0
    n_routed_global = 0
    test_authors_personal = set()    # test authors that had a personal model

    for i in range(n):
        a = authors_test[i]
        if a in personal_models:
            model = personal_models[a]
            used_model_type.append('personal')
            model_used_name.append(a)
            test_authors_personal.add(a)
            n_routed_personal += 1
        else:
            model = global_model
            used_model_type.append('global')
            model_used_name.append('GLOBAL')
            n_routed_global += 1
        if model is not None:
            p = predict_proba_buggy(model, X_test[i:i+1])[0]
            pred_prob[i] = p
            pred_label[i] = int(p >= 0.5)
        else:
            pred_prob[i] = 0.0
            pred_label[i] = 0

    print(f'  routed to personal models: {n_routed_personal}')
    print(f'  routed to global model   : {n_routed_global}')

    # ---- save predictions ----
    # LOC = la + ld (Kamei effort measure); attached by row position (predictions
    # are emitted in test_df order).
    la_test = test_df['la'].astype(float).values
    ld_test = test_df['ld'].astype(float).values
    loc_test = la_test + ld_test
    pred_df = pd.DataFrame({
        'commit_hash': commits_test,
        'author_name': authors_test,
        'true_label': y_test,
        'pred_label': pred_label,
        'pred_prob_buggy': pred_prob,
        'used_model_type': used_model_type,
        'model_used': model_used_name,
        'la': la_test,
        'ld': ld_test,
        'LOC': loc_test,
    })
    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f'  saved predictions -> {pred_path}')

    # ---- personal models summary ----
    # add per-author test usage counts
    test_usage = Counter(model_used_name)
    for row in personal_train_info:
        row['n_test_routed_to_personal'] = test_usage.get(row['author'], 0)
    personal_summary = pd.DataFrame(personal_train_info)
    pers_path = os.path.join(OUTPUT_DIR, 'personal_models_summary.csv')
    personal_summary.to_csv(pers_path, index=False)
    print(f'  saved personal models summary -> {pers_path}')

    # ---- metrics ----
    def metrics(y_true, y_pred, y_prob, name):
        if len(np.unique(y_true)) < 2:
            auc = float('nan')
        else:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = float('nan')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            'name': name, 'n': len(y_true),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': auc, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }

    overall = metrics(y_test, pred_label, pred_prob, 'OVERALL')
    pers_mask = np.array([t == 'personal' for t in used_model_type])
    glob_mask = ~pers_mask
    pers_m = metrics(y_test[pers_mask], pred_label[pers_mask],
                     pred_prob[pers_mask], 'PERSONAL-subset') if pers_mask.any() else None
    glob_m = metrics(y_test[glob_mask], pred_label[glob_mask],
                     pred_prob[glob_mask], 'GLOBAL-subset') if glob_mask.any() else None

    # ---- effort-aware & extra metrics (effort = la + ld = LOC) ----
    ea = EffortAwareMetrics()
    ea_overall = ea.eval(build_result_df(y_test, pred_label, pred_prob, loc_test))
    ea_pers = (ea.eval(build_result_df(y_test[pers_mask], pred_label[pers_mask],
                                       pred_prob[pers_mask], loc_test[pers_mask]))
               if pers_mask.any() else None)
    ea_glob = (ea.eval(build_result_df(y_test[glob_mask], pred_label[glob_mask],
                                       pred_prob[glob_mask], loc_test[glob_mask]))
               if glob_mask.any() else None)

    # ---- routing summary ----
    summary_lines = []
    summary_lines.append('===== PCC Routing & Metrics Summary =====\n')
    summary_lines.append(f'Train size: {len(train_df)} | Test size: {len(test_df)}')
    summary_lines.append(f'Authors in train: {len(train_author_counts)} | '
                         f'authors with > {PERSONAL_THRESHOLD} train samples: '
                         f'{len(personal_candidates)}')
    summary_lines.append(f'Personal models built: {len(personal_models)} / '
                         f'{len(personal_candidates)} candidates')
    summary_lines.append(f'Global model train samples (small authors only): '
                         f'{len(global_idxs)} (buggy={int(y_global.sum())})')
    summary_lines.append('')
    summary_lines.append('--- Test routing ---')
    summary_lines.append(f'Total test samples: {n}')
    summary_lines.append(f'  routed to a personal model: {n_routed_personal}')
    summary_lines.append(f'  routed to the Global model : {n_routed_global}')
    summary_lines.append(f'Distinct test authors that used a personal model: '
                         f'{len(test_authors_personal)}')
    summary_lines.append('')
    summary_lines.append('--- Authors WITH a personal model ---')
    for r in sorted(personal_train_info, key=lambda x: -x['n_train']):
        flag = 'OK' if r['has_personal_model'] else f"SKIPPED({r['skip_reason']})"
        summary_lines.append(
            f'  {r["author"]:<28} train={r["n_train"]:<6} '
            f'buggy={r["n_train_buggy"]:<6} test_used={r["n_test_routed_to_personal"]:<5} {flag}')
    summary_lines.append('')
    summary_lines.append('--- Metrics (effort-free, threshold = 0.5) ---')
    for m in [overall, pers_m, glob_m]:
        if m is None:
            continue
        summary_lines.append(
            f'  [{m["name"]}] n={m["n"]} acc={m["accuracy"]:.4f} '
            f'P={m["precision"]:.4f} R={m["recall"]:.4f} F1={m["f1"]:.4f} '
            f'AUC={m["auc"]:.4f} TP={m["tp"]} FP={m["fp"]} FN={m["fn"]} TN={m["tn"]}')
    summary_lines.append('')
    summary_lines.append('--- Effort-aware & extra metrics (effort = la + ld = LOC) ---')
    for name, eam in [('OVERALL', ea_overall),
                      ('PERSONAL-subset', ea_pers),
                      ('GLOBAL-subset', ea_glob)]:
        if eam is None:
            continue
        summary_lines.append(
            f'  [{name}] gmean={eam["gmean"]} '
            f'Recall@20%Effort={eam["recall_at_20_percent_effort"]} '
            f'Effort@20%Recall={eam["effort_at_20_percent_LOC_recall"]} '
            f'Popt={eam["p_opt"]}  (F1={eam["f1"]} AUC={eam["auc"]})')

    summary_text = '\n'.join(summary_lines)
    print('\n' + summary_text)
    sum_path = os.path.join(OUTPUT_DIR, 'routing_summary.txt')
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f'\n  saved routing summary -> {sum_path}')


if __name__ == '__main__':
    main()
