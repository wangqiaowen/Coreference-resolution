import re
import paramiko
import csv
from collections import defaultdict


def load_data(filename):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='130.60.75.213', port=22, username='baml_02', password='CHANGE_6b4ba0bc')
    sftp_client = ssh.open_sftp()
    stdin, stdout, stderr = ssh.exec_command('ls')
    result = stdout.read()
    remote_file = sftp_client.open(filename)

    return remote_file


def nested_dict():
    return defaultdict(nested_dict)


def chain_extraction(data):
    chains = []
    all_sentence_ids = []
    prelim_chains = dict()
    sentence_max = dict()
    for line in data:
        if line and len(line) > 1:
            if "#begin document" not in line:
                if "#end document" not in line:
                    line = line.strip('\n')
                    linecols = line.split('\t')
                    sentid = linecols[1]
                    all_sentence_ids.append(sentid)
                    tokenid = linecols[2]
                    seg = linecols[0]
                    ids = linecols[-1].split('|')
                    try:
                        if int(sentence_max[sentid]) < int(tokenid):
                            sentence_max[sentid] = tokenid
                    except KeyError:
                        sentence_max[sentid] = tokenid
                    for identry in ids:
                        _id = re.search(r'\d+', identry)
                        if _id:
                            _id = int(_id.group(0))
                            if identry.startswith('(') and identry.endswith(')'):
                                chains.append([_id, seg, sentid, tokenid, (sentid, tokenid)])
                            else:
                                if identry.startswith('('):
                                    prelim_chains[_id] = [seg, sentid, tokenid]
                                elif identry.endswith(')'):
                                    prelim_chains[_id].append((sentid, tokenid))
                                    chains.append([_id] + prelim_chains[_id])
    return chains, all_sentence_ids, sentence_max


def create_feature_dict(sync_data, sentence_id_list):
    feature_dict = nested_dict()
    i = 0
    for line in sync_data:
        if line and len(line) > 1:
            if "#begin document" not in line:
                if "#end document" not in line:
                    line = line.strip('\n')
                    linecols = line.split('\t')
                    doc_id = linecols[0]
                    token_id = linecols[1]
                    pos_tag = linecols[5]
                    grammarfunct = linecols[9]
                    head = linecols[8]
                    person_number_gender = linecols[7].split('|')
                    person = None
                    number = None
                    gender = None
                    for png in person_number_gender:
                        if png.startswith("Person"):
                            person = png.split('=')[1]
                        elif png.startswith("Number"):
                            number = png.split("=")[1]
                        elif png.startswith("Gender"):
                            gender = png.startswith("Gender")
                    sent_id = sentence_id_list[i]
                    feature_dict[doc_id][str(sent_id)][token_id] = [pos_tag, grammarfunct, head,
                                                                    (person, number, gender)]
                    i += 1

    return feature_dict


def preprocess_gold_standard(_extracted_chains):
    # structure goldchains: {doc ID: [coref_id, sent ID, Start Token, End Token]}
    goldchains = defaultdict(list)
    segments = set()
    for row in _extracted_chains:
        goldchains[row[1]].append([row[0]] + row[2:])
        segments.add(row[1])
    return goldchains, segments


def generate_features(chain, doc_feats, sentence_id, max_ids, neg_sample=False):
    # lookup-dict: {doc id: {sentence_id: {tokenid:[pos_tag, grammarfunct, head, (person, number, gender)]}}}
    # chains: (sent_id, start_pos, end_pos)
    sentence_id1 = str(sentence_id)
    sentence_id2 = str(chain[-1][0])
    token_id1 = str(chain[-2])
    token_id2 = str(chain[-1][1])
    sentence_dist = int(sentence_id2) - int(sentence_id1)
    if sentence_dist == 0:
        distance = int(token_id2) - int(token_id1)
    else:
        weight = 0
        for i in max_ids.keys():
            if sentence_id1 <= i > sentence_id2:
                weight += int(max_ids[i])
        distance = (int(token_id2) + weight) - int(token_id1)
    pos1 = doc_feats[sentence_id1][token_id1][0]
    grammar1 = doc_feats[sentence_id1][token_id1][1]
    head1 = doc_feats[sentence_id1][token_id1][2]
    pos2 = doc_feats[sentence_id2][token_id2][0]
    grammar2 = doc_feats[sentence_id2][token_id2][1]
    head2 = doc_feats[sentence_id2][token_id2][2]

    match = doc_feats[sentence_id1][token_id1][3] == doc_feats[sentence_id2][token_id2][3]
    same_grammar = grammar1 == grammar2
    same_pos = pos1 == pos2
    same_head = head1 == head2

    if neg_sample:
        label = 0
    else:
        label = 1

    with open('gold_standard2.csv', 'a') as f:
        filewriter = csv.writer(f, delimiter=',')
        filewriter.writerow([label, distance, pos1, grammar1, pos2, grammar2,
                             int(same_pos), int(same_grammar), int(same_head), int(match)])


def get_true_pos_positions(segchains, doc_id):
    # get the start and end positions for the true positives
    true_positives = []
    for chain in segchains:
        start_pos = int(chain[2])
        end_pos = int(chain[3][1])
        sent_id = int(chain[1])
        coref_id = chain[0]
        true_positives.append((doc_id, sent_id, start_pos, end_pos, coref_id))

    sorted_positives = sorted(true_positives, key=lambda element: (element[0], element[1], element[2], element[3]))
    return sorted_positives


def get_negative_samples(seg, segchains, neg_sample_max=5):
    neg_samples = defaultdict(list)
    true_positives = get_true_pos_positions(segchains, seg)
    start_idx = 0
    for doc_id, sentence_id, start_token, end_token, coref_id in true_positives:
        current_coref_id = coref_id
        end_idx = None
        i = start_idx
        num_negs = 0
        for d, s, st, end, coref in true_positives[start_idx + 1:]:
            if num_negs <= neg_sample_max:
                if d == doc_id:
                    i += 1
                    if coref == current_coref_id:
                        end_idx = i
                    else:
                        end_idx = None
        num_negs += 1
        start_idx += 1
        if end_idx:
            neg_sample_list = true_positives[start_idx:end_idx]
            for sample in neg_sample_list:
                neg_samples[doc_id].append([coref_id, str(sentence_id), end_token, (sample[1], sample[3])])

    return neg_samples


def generate_gold_standard(_extracted_chains, lookup_dict, max_ids):
    print('loading data')
    goldchains, segments = preprocess_gold_standard(_extracted_chains)
    with open('gold_standard2.csv', 'w') as f:
        filewriter = csv.writer(f, delimiter=',')
        filewriter.writerow(['Label', 'Distance', 'POS1', 'Grammar1', 'POS2', 'Grammar2',
                             'samePOS', 'sameGrammar', 'sameHead', 'Match'])
    total_positives = 0
    total_negatives = 0
    total_segments = 0
    for seg in segments:
        total_segments += 1
        print('seg', seg)
        segchains = goldchains[seg]
        doc_feats = lookup_dict[seg]
        print('create positive samples for document ', seg)
        for chain in segchains:
            # print('chain', chain)
            generate_features(chain, doc_feats, chain[1], max_ids)
        print('Document %s contains %s positive samples.' % (seg, len(segchains)))
        total_positives += len(segchains)
        neg_sample_chain = get_negative_samples(seg, segchains)
        print('create negative samples for document ', seg)
        neg_chains = neg_sample_chain[seg]
        for chain in neg_chains:
            # print(chain)
            generate_features(chain, doc_feats, chain[1], max_ids, neg_sample=True)
        print('Document %s contains %s negative samples.' % (seg, len(neg_chains)))
        print()
        total_negatives += len(neg_chains)
        print('document %s/%s done' % (total_segments, len(segments)))

    print("total positives %s" % total_positives)
    print("total negatives %s" % total_negatives)


if __name__ == '__main__':

    full_data = load_data('train.tueba.coref.txt')
    sync_data = load_data('train.tueba.sync.txt')
    extracted_chains, sentence_ids, max_ids = chain_extraction(full_data)
    lookup_dict = create_feature_dict(sync_data, sentence_ids)
    generate_gold_standard(extracted_chains, lookup_dict, max_ids)
