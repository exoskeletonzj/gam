from collections import defaultdict
import re
import torch

MAX_CONTEXT_LENGTH = 350  # measured in words
WORDS_PER_EVENT = 10
MAX_LENGTH = 512
MAX_TGT_LENGTH = 40


def get_template(ex, index, ontology_dict, tokenizer):
    event_type = ex['event_mentions'][index]['event_type']

    role2arg = defaultdict(list)
    for argument in ex['event_mentions'][index]['arguments']:
        role2arg[argument['role']].append(argument)
    role2arg = dict(role2arg)

    arg_idx2text = defaultdict(list)
    for role in role2arg.keys():
        if role not in ontology_dict[event_type]:
            continue
        for i, argument in enumerate(role2arg[role]):
            arg_text = argument['text']
            if i < len(ontology_dict[event_type][role]):
                arg_idx = ontology_dict[event_type][role][i]
            else:
                arg_idx = ontology_dict[event_type][role][-1]
            # 如arg1对应argument['text']
            arg_idx2text[arg_idx].append(arg_text)

    template = ontology_dict[event_type]['template']
    input_template = re.sub(r'<arg\d>', '<arg>', template)
    for arg_idx, text_list in arg_idx2text.items():
        text = ' and '.join(text_list)
        template = re.sub('<{}>'.format(arg_idx), text, template)
    output_template = re.sub(r'<arg\d>', '<arg>', template)

    space_tokenized_input_template = input_template.split()
    tokenized_input_template = []
    for w in space_tokenized_input_template:
        tokenized_input_template.extend(tokenizer.tokenize(w, add_prefix_space=True))

    space_tokenized_output_template = output_template.split()
    tokenized_output_template = []
    for w in space_tokenized_output_template:
        tokenized_output_template.extend(tokenizer.tokenize(w, add_prefix_space=True))
    return tokenized_input_template, tokenized_output_template

def get_context(ex, index, max_length):
    '''
    RETURN:
    context: part of the context with the center word and no more than max length.
    offset: the position of the first token of context in original document
    '''
    trigger = ex['event_mentions'][index]['trigger']
    offset = 0
    context = ex["tokens"]
    center_sent = trigger['sent_idx']
    if len(context) > max_length:
        cur_len = len(ex['sentences'][center_sent][0])
        context = [tup[0] for tup in ex['sentences'][center_sent][0]]
        if cur_len > max_length:
            trigger_start = trigger['start']
            start_idx = max(0, trigger_start - max_length // 2)
            end_idx = min(len(context), trigger_start + max_length // 2)
            context = context[start_idx: end_idx]
            offset = sum([len(ex['sentences'][idx][0]) for idx in range(center_sent)]) + start_idx
        else:
            left = center_sent - 1
            right = center_sent + 1

            total_sents = len(ex['sentences'])
            prev_len = 0
            while cur_len > prev_len:
                prev_len = cur_len
                # try expanding the sliding window
                if left >= 0:
                    left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                    if cur_len + len(left_sent_tokens) <= max_length:
                        context = left_sent_tokens + context
                        left -= 1
                        cur_len += len(left_sent_tokens)

                if right < total_sents:
                    right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                    if cur_len + len(right_sent_tokens) <= max_length:
                        context = context + right_sent_tokens
                        right += 1
                        cur_len += len(right_sent_tokens)
            offset = sum([len(ex['sentences'][idx][0]) for idx in range(left + 1)])

    assert len(context) <= max_length

    assert ex["tokens"][offset:offset + len(context)] == context

    return context, offset


def simple_type(type_name):
    if ':' in type_name:
        t1, t2 = type_name.split(':')
        return t2.lower()

    _, t1, t2 = type_name.split('.')
    if t2 == "Unspecified":
        return t1.lower()
    if len(t1) < len(t2):
        return t1.lower()
    return t2.lower()


def tokenize_with_labels(tokens, labels, tokenizer, type='bart'):
    '''
    tokens: a list of tokens
    labels: a list of labels, each of them matches with the token
    RETURN:
    tokenized_tokens: a list of tokenized tokens
    tokenized_labels
    '''
    assert len(tokens) == len(labels)

    if type == 'bart':
        tokenized_tokens = tokenizer.tokenize(' '.join(tokens), add_prefix_space=True)
        tokenized_labels = [0] * len(tokenized_tokens)
        ptr = 0
        for idx, token in enumerate(tokenized_tokens):
            tokenized_labels[idx] = labels[ptr]
            if idx + 1 < len(tokenized_tokens) and (
                    tokenized_tokens[idx + 1][0] == "Ġ" or tokenized_tokens[idx + 1] == ' <tgr>'):
                ptr += 1
    else:
        tokenized_tokens = tokenizer.tokenize(' '.join(tokens))
        tokenized_labels = [0] * len(tokenized_tokens)
        ptr = 0
        current_word = ''
        for idx, token in enumerate(tokenized_tokens):
            if token.startswith('##'):
                current_word += token[2:]
            else:
                current_word += token
            tokenized_labels[idx] = labels[ptr]
            if current_word == tokens[ptr]:
                ptr += 1
                current_word = ''
    assert len(tokenized_tokens) == len(tokenized_labels)

    return tokenized_tokens, tokenized_labels


def create_instance_normal(ex, ontology_dict, index=0, id2entity=None, tokenizer=None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH)

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset

    # get context with trigger being tagged and its argument mask list
    prefix = tokenizer.tokenize(' '.join(context[:trigger_start]), add_prefix_space=True)
    tgt = tokenizer.tokenize(' '.join(context[trigger_start: trigger_end]), add_prefix_space=True)
    suffix = tokenizer.tokenize(' '.join(context[trigger_end:]), add_prefix_space=True)
    context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
    return input_template, output_template, context_tag_trigger, offset


def create_instance_tag(ex, ontology_dict, index=0, close_events=None, id2entity=None, tokenizer=None):
    input_template, output_template = get_template(ex, index, ontology_dict, tokenizer)
    context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH - WORDS_PER_EVENT * len(close_events))
    # context, offset = get_context(ex, index, MAX_CONTEXT_LENGTH)

    trigger = ex['event_mentions'][index]['trigger']
    trigger_start = trigger['start'] - offset
    trigger_end = trigger['end'] - offset
    original_mask = [0] * len(context)
    prefix, prefix_labels = tokenize_with_labels(context[:trigger_start], original_mask[:trigger_start], tokenizer)
    tgt, tgt_labels = tokenize_with_labels(context[trigger_start: trigger_end],
                                           original_mask[trigger_start:trigger_end], tokenizer)
    suffix, suffix_labels = tokenize_with_labels(context[trigger_end:], original_mask[trigger_end:], tokenizer)

    context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
    context_tag_trigger_mask = prefix_labels + [0] + tgt_labels + [0] + suffix_labels

    return input_template, output_template, context_tag_trigger, context_tag_trigger_mask,offset


def get_devmetion_mean(ontology_dict, ontology_dict_roles,input_template, context_tag_trigger, ex,offset,coref_dict):

    startid = 0
    # print(re.findall(r'\d',''.join(re.findall(r'<arg\d>',ontology_dict_roles['template']))))
    roles=re.findall(r'\d',''.join(re.findall(r'<arg\d>',ontology_dict_roles['template'])))
    metion_id_dict={}
    for key in range(len(ex['entity_mentions'])):
        if ex['entity_mentions'][key]['start'] >= offset:
            startid = key
            break
    inputs = '[CLS]'.split() + input_template + '[SEP]'.split() + '[CLS]'.split() + context_tag_trigger + list(
        'P' * (MAX_LENGTH - 3 - len(context_tag_trigger) - len(input_template)))
    inputs = inputs[:MAX_LENGTH]
    len_metion = len('[CLS]'.split() + input_template + '[SEP]'.split() + '[CLS]'.split())
    sentence_dict={}
    input_metion=[]
    start_me=0
    append_id=1
    i=0
    while i<len_metion:
        if inputs[i]==' <arg>':
            input_metion.append(0)
            j = i+1
            len_role=len(ontology_dict_roles['roles'][int(roles[start_me])-1])
            len_tmp=-1
            while j<len_metion and len_tmp<len_role:
                len_tmp+=len(inputs[j])
                input_metion.append(append_id)
                j+=1
            sentence_dict[append_id] = [0,ontology_dict_roles['roles'][int(roles[start_me])-1],ontology_dict[int(roles[start_me])-1]]
            start_me += 1
            append_id += 1
            i=j-1
        else:
            input_metion.append(0)
        i+=1

    i = len('[CLS]'.split() + input_template + '[SEP]'.split() + '[CLS]'.split())

    l = len('[CLS]'.split() + input_template + '[SEP]'.split() + '[CLS]'.split() + context_tag_trigger)
    while i < len(inputs):
        if inputs[i] == 'ĠâĢ':
            input_metion.append(0)
            input_metion.append(0)
            # l += 3
            i += 2
            offset+=1
            continue

        if startid<len(ex['entity_mentions']) and ex['entity_mentions'][startid]['start']==offset and i<l:
            arg=ex['entity_mentions'][startid]['text'].replace(' ', '')
            len_txt = len(inputs[i])
            len_arg = len(arg)
            sentence_dict[append_id]=[ex['entity_mentions'][startid]['sent_idx']+1,arg,ex['entity_mentions'][startid]['entity_type'].split()]
            input_metion.append(append_id)
            tmp = len_txt - 1
            j = i + 1
            while tmp < len_arg and j < len(inputs):
                input_metion.append(append_id)
                tmp += len(inputs[j])
                if inputs[j][0] == 'Ġ':
                    tmp -= 1
                j += 1
            i = j - 1
            offset =ex['entity_mentions'][startid]['end']
            metion_id_dict[ex['entity_mentions'][startid]['id']]=append_id
            startid+=1
            append_id+=1
        elif offset<len(ex['tokens']):
            len_txt = len(inputs[i])-1
            len_arg = len(ex['tokens'][offset])
            input_metion.append(0)
            tmp = len_txt
            j = i + 1
            while tmp < len_arg and j < len(inputs):
                input_metion.append(0)
                tmp += len(inputs[j])
                if inputs[j][0] == 'Ġ':
                    tmp -= 1
                j += 1
            i = j - 1
            offset+=1
        else :
            input_metion.append(0)
        i += 1
    coref_dict_change=coref_dict
    coref_dict_change['dict']=metion_id_dict


    return input_metion,sentence_dict,coref_dict_change


def get_entity(input_metion,input_tokens):
    entity_dict = []
    for en in input_metion:
        if en != 0:
            entity_dict.append(en - 1)
    # new_dict = list(set(entity_dict))
    # new_dict.sort()
    input_entity = []
    # count=0
    i=0
    # output_entdict = {}
    entity_key=0
    while i<len(input_tokens):
        if input_metion[i]!=0:
            # if bool(output_entdict.get(input_metion[i])) :
            #     entity_key = output_entdict[input_metion[i]]
            # else:
            input_entity.append([])
            #     output_entdict[input_metion[i]]=count
            #     entity_key=count
            #     count+=1
            j=i
            while j<len(input_metion) and input_metion[i]==input_metion[j]:
                input_entity[entity_key].append(input_tokens[j])
                j+=1
            i=j-1
            entity_key+=1
        i+=1
    return input_entity
