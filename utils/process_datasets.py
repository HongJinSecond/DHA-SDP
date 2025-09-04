import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
from utils.util import parse_jit_args, build_model_tokenizer_config, set_seed
from sklearn.preprocessing import StandardScaler

# project_names=['ant-ivy', 'commons-math', 'opennlp', 'parquet-mr', 'commons-lang',
#        'commons-net', 'commons-collections', 'commons-beanutils',
#        'commons-codec', 'commons-compress', 'commons-configuration',
#        'commons-digester', 'commons-jcs', 'commons-io', 'commons-scxml',
#        'commons-validator', 'commons-vfs', 'giraph', 'commons-bcel',
#        'commons-dbcp', 'gora']


def normalize_df(df, feature_columns):
    df['fix'] = df['fix'].map({'True': True, 'False': False}).astype(int)
    df = df.astype({i: "float32" for i in feature_columns})
    return df[["commit_hash","author_name","project","commit_message"] + feature_columns]


def parse_data_file(args, mode):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)

    # store parsed data.
    examples = []

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
    
    # scaler
    scaler=StandardScaler()
    Base=normalize_df(pd.read_pickle(args.train_data_file[1]),manual_features_columns)
    
    # feature `fix` do not need normalization
    scaler.fit(Base[manual_features_columns[:-1]])
    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = scaler.transform(fedata[manual_features_columns[:-1]])
    fedata[manual_features_columns[:-1]] = manual_features

    # parse ccdata.
    commit_ids, labels, msgs, codes = ccdata
    for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
        manual_features = fedata[fedata["commit_hash"] == commit_id][manual_features_columns].to_numpy().squeeze()
        examples.append((commit_id, label, msg, code, manual_features))

    if mode == "train":
        random.seed(args.seed)
        random.shuffle(examples)

    return examples



# Add special tokens to tokenizer
def further_parse_for_CCT5(example, tokenizer, args):
    """
    Add special tokens to code lines and return tokenized text and other features.
    :param example: the example list read by function parse_data_file
    :param tokenizer:
    :param args:
    :param contains_developer:
    :return: tuple of several tensors samples (input_ids, attention_mask, manual_features, label)
    """
    commit_id, label, msg, code, manual_features = example
    label = int(label)
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '<add>'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '<del>'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = msg_tokens + ['<add>'] + added_tokens + ['<del>'] + removed_tokens
    input_tokens = input_tokens[:args.max_input_tokens - 2]
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return commit_id,torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label



# Add special tokens to tokenizer
def further_parse(example, tokenizer, args):
    """
    Add special tokens to code lines and return tokenized text and other features.
    :param example: the example list read by function parse_data_file
    :param tokenizer:
    :param args:
    :param contains_developer:
    :return: tuple of several tensors samples (input_ids, attention_mask, manual_features, label)
    """
    commit_id, label, msg, code, manual_features = example
    label = int(label)
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]

    added_codes = [' '.join(line.split()) for line in code['added_code']]
    codes = '[ADD]'.join([line for line in added_codes if len(line)])
    added_tokens.extend(tokenizer.tokenize(codes))

    removed_codes = [' '.join(line.split()) for line in code['removed_code']]
    codes = '[DEL]'.join([line for line in removed_codes if len(line)])
    removed_tokens.extend(tokenizer.tokenize(codes))

    input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
    input_tokens = input_tokens[:args.max_input_tokens - 2]
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)

    padding_length = args.max_input_tokens - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)

    assert len(input_ids) == args.max_input_tokens
    assert len(input_mask) == args.max_input_tokens

    return commit_id,torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(manual_features), label

# For Project Aware
def load_project_datas(tokenizer,args,mode):
    """
    Load the datasets only for project, and return a dict for target datasets.
    :param args:
    :param mode:
    :return: dict( project_name: torch.utils.data.Dataset )
    """
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
        # scaler
    scaler=StandardScaler()
    Base=normalize_df(pd.read_pickle(args.train_data_file[1]),manual_features_columns)
    scaler.fit(Base[manual_features_columns[:-1]])
    # feature `fix` do not need normalization

    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = scaler.transform(fedata[manual_features_columns[:-1]])
    fedata[manual_features_columns[:-1]] = manual_features

    # parse ccdata.
    _, labels, _, codes = ccdata
    # use dict to save information
    project_dict=dict()
    fedata["codes"]=codes
    fedata["new_labels"]=labels

    # get project names
    project_names = fedata["project"].unique().tolist()

    for project in project_names:
        examples = []
        # get new data
        project_data = fedata[fedata["project"]==project]
        project_labels = project_data["new_labels"].values.tolist()
        project_codes = project_data["codes"].values.tolist()
        project_msgs = project_data["commit_message"].values.tolist()
        project_ids = project_data["commit_hash"].values.tolist()
        project_manual_features = project_data[manual_features_columns].to_numpy().squeeze()

        for commit_id,label, msg, code,manual_features in zip(project_ids,project_labels, project_msgs, project_codes,project_manual_features):
            # manual_features = project_data[manual_features_columns].to_numpy().squeeze()
            examples.append((commit_id, label, msg, code, manual_features))
        if mode == "train":
            random.seed(args.seed)
            random.shuffle(examples)
        if args.pretrained_model=="cct5":
            project_dict[project]=CCT5Dataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=project)
        else:
            project_dict[project]=StyleDataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=project)
    
    return project_dict
    

def load_developer_datas(tokenizer,args,mode,min_size=100):
    codechange_file = ""
    feature_file = ""
    if mode == "train":
        codechange_file, feature_file = args.train_data_file
    elif mode == "eval":
        codechange_file, feature_file = args.eval_data_file
    elif mode == "test":
        codechange_file, feature_file = args.test_data_file

    ccdata = pd.read_pickle(codechange_file)
    fedata = pd.read_pickle(feature_file)

    # parse fedata.
    manual_features_columns = ["la", "ld", "nf", "ns", "nd", "entropy", "ndev",
                               "lt", "nuc", "age", "exp", "rexp", "sexp", "fix"]
        # scaler
    scaler=StandardScaler()
    Base=normalize_df(pd.read_pickle(args.train_data_file[1]),manual_features_columns)
    scaler.fit(Base[manual_features_columns[:-1]])
    # feature `fix` do not need normalization

    fedata = normalize_df(fedata, manual_features_columns)
    # standardize fedata along any features.
    manual_features = scaler.transform(fedata[manual_features_columns[:-1]])
    fedata[manual_features_columns[:-1]] = manual_features

    # only the author contains at least 100(min size) samples can be used to train lora
    author_counts = fedata["author_name"].value_counts()
    large_authors_list = author_counts[author_counts > min_size].index.tolist()
    # small_authors_list = author_counts[author_counts <= min_size].index.tolist()


    # parse ccdata.
    _, labels, _, codes = ccdata
    # use dict to save information
    developer_dict=dict()
    fedata["codes"]=codes
    fedata["new_labels"]=labels
    
    for developer in large_authors_list:
        examples = []
        # get new data
        developer_data = fedata[fedata["author_name"]==developer]
        developer_labels = developer_data["new_labels"].values.tolist()
        developer_codes = developer_data["codes"].values.tolist()
        developer_msgs = developer_data["commit_message"].values.tolist()
        developer_ids = developer_data["commit_hash"].values.tolist()
        developer_manual_features = developer_data[manual_features_columns].to_numpy().squeeze()

        for commit_id,label, msg, code,manual_features in zip(developer_ids,developer_labels, developer_msgs, developer_codes,developer_manual_features):
            # manual_features = developer_data[manual_features_columns].to_numpy().squeeze()
            examples.append((commit_id, label, msg, code, manual_features))
        if mode == "train":
            random.seed(args.seed)
            random.shuffle(examples)
        if args.pretrained_model=="cct5":
            developer_dict[developer]=CCT5Dataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=developer)
        else:        
            developer_dict[developer]=StyleDataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=developer)
    
    ############## other developer ###############

    if mode=="test":
    ############## only test need this data ###############
        developer="other"
        examples = []
        # get new data
        developer_data = fedata[~(fedata["author_name"].isin(large_authors_list))]
        developer_labels = developer_data["new_labels"].values.tolist()
        developer_codes = developer_data["codes"].values.tolist()
        developer_msgs = developer_data["commit_message"].values.tolist()
        developer_ids = developer_data["commit_hash"].values.tolist()
        developer_manual_features = developer_data[manual_features_columns].to_numpy().squeeze()

        for commit_id,label, msg, code,manual_features in zip(developer_ids,developer_labels, developer_msgs, developer_codes,developer_manual_features):
            # manual_features = developer_data[manual_features_columns].to_numpy().squeeze()
            examples.append((commit_id, label, msg, code, manual_features))
        if args.pretrained_model=="cct5":
            developer_dict[developer]=CCT5Dataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=developer)
        else:        
            developer_dict[developer]=StyleDataset(examples=examples,tokenizer=tokenizer,args=args,cluster_name=developer)
    return developer_dict

    # return examples

class JITFineDataset(Dataset):
    def __init__(self, tokenizer, args, mode):
        self.mid_examples = parse_data_file(args, mode)
        if args.pretrained_model=="cct5":
            self.examples = [further_parse_for_CCT5(item, tokenizer, args) for item in self.mid_examples]
        else:
            self.examples = [further_parse(item, tokenizer, args) for item in self.mid_examples]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class StyleDataset(Dataset):
    def __init__(self, examples, tokenizer, args, cluster_name):
        self.cluster_name = cluster_name
        self.mid_examples = examples
        self.examples = [further_parse(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    

class CCT5Dataset(Dataset):
    def __init__(self, examples, tokenizer, args, cluster_name=None):
        # print("--load CCT5 Datasets--")
        self.cluster_name = cluster_name
        self.mid_examples = examples
        self.examples = [further_parse_for_CCT5(item, tokenizer, args) for item in self.mid_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ClusterDataset(Dataset):
    def __init__(self, examples,cluster_name=None):
        self.cluster_name = cluster_name
        self.mid_examples = examples
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    

    def __getitem__(self, index):
        return self.examples[index]

if __name__ == "__main__":
    args = parse_jit_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.available_gpu)
    args.device = device

    set_seed(args)

    model, tokenizer, config = build_model_tokenizer_config(args)

    examples = parse_data_file(args, "test")
    print(examples[0])
    from torch.utils.data import DataLoader 
    # train_dataset = JITFineDataset(tokenizer, args, "train")
    eval_dataset = JITFineDataset(tokenizer, args, "eval")
    # test_dataset = JITFineDataset(tokenizer, args, "test")
    train_dataloader = DataLoader(eval_dataset, batch_size=2)
    print(eval_dataset.examples[0])
    print(eval_dataset.examples[1])
    # print(len(train_dataset))
    # print(len(eval_dataset))
    # print(len(test_dataset))


