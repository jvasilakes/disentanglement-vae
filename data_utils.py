import re
import json
from collections import defaultdict

# External packages
import torch
from sklearn.preprocessing import LabelEncoder


class LabeledTextDataset(torch.utils.data.Dataset):

    def __init__(self, docs, labels, word2idx, label_encoders):
        super(LabeledTextDataset, self).__init__()
        self.docs = docs
        assert isinstance(labels[0], dict)
        self.labels = labels
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")
        self.word2idx = word2idx
        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
        self.label_encoders = label_encoders
        self.Xs = [self.doc2tensor(doc) for doc in self.docs]
        self.Ys = [self.label2tensor(lab) for lab in self.labels]

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return len(self.Xs)

    @property
    def y_dims(self):
        dims = dict()
        for (label_name, encoder) in self.label_encoders.items():
            num_classes = len(encoder.classes_)
            if num_classes == 2:
                num_classes = 1
            dims[label_name] = num_classes
        return dims

    def doc2tensor(self, doc):
        idxs = []
        for tok in doc:
            try:
                idxs.append(self.word2idx[tok])
            except KeyError:
                idxs.append(self.word2idx["<UNK>"])
        return torch.LongTensor([[idxs]])

    def label2tensor(self, label_dict):
        tensorized = dict()
        for (label_name, label) in label_dict.items():
            encoder = self.label_encoders[label_name]
            # CrossEntropy requires LongTensors
            # BCELoss requires FloatTensors
            if len(encoder.classes_) > 2:
                tensor_fn = torch.LongTensor
            else:
                tensor_fn = torch.FloatTensor
            enc = encoder.transform([label])
            tensorized[label_name] = tensor_fn(enc)
        return tensorized


def get_sentences_labels(path, label_keys=None, N=-1):
    sentences = []
    labels = []
    with open(path, 'r') as inF:
        for (i, line) in enumerate(inF):
            if i == N:
                break
            data = json.loads(line)
            sentences.append(data["sentence"])
            if label_keys is None:
                label_keys = [key for key in data.keys()
                              if key != "sentence"]
            labs = {key: value for (key, value) in data.items()
                    if key in label_keys}
            labels.append(labs)
    return sentences, labels


def preprocess_sentences(sentences, SOS, EOS, lowercase=True):
    out_data = []
    for sent in sentences:
        sent = sent.strip()
        if lowercase is True:
            sent = sent.lower()
        sent = re.sub(r"([.!?])", r" \1", sent)
        sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
        sent = sent.split()
        sent = [SOS] + sent + [EOS]
        out_data.append(sent)
    return out_data


def preprocess_labels(labels, label_encoders={}):
    raw_labels_by_name = defaultdict(list)
    for label_dict in labels:
        for (label_name, lab) in label_dict.items():
            raw_labels_by_name[label_name].append(lab)

    label_encoders = dict()
    enc_labels_by_name = dict()
    for (label_name, labs) in raw_labels_by_name.items():
        if label_name in label_encoders.keys():
            # We're passing in an already fit encoder
            le = label_encoders[label_name]
        else:
            le = LabelEncoder()
        y = le.fit_transform(labs)
        label_encoders[label_name] = le
        enc_labels_by_name[label_name] = y

    return labels, label_encoders
