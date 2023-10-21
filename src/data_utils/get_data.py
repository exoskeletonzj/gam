
import json
from genie.utils import load_ontology, check_pronoun, clean_mention
from .data_utils import create_instance_tag, create_instance_normal
from .data_utils import  get_devmetion_mean, get_entity
from tqdm import tqdm

MAX_CONTEXT_LENGTH = 350  # measured in words
WORDS_PER_EVENT = 10
MAX_LENGTH = 512
MAX_TGT_LENGTH = 70

def get_data_normal(source=None, target=None, tokenizer=None, dataset="wikievents",coref_target=None):
    ontology_dict = load_ontology(dataset)
    max_tokens = 0
    max_tgt = 0
    print(f"source:{source}")
    print(f"target:{target}")
    print(f"dataset:{dataset}")
    coref_dict = {}
    with open(coref_target, 'r') as reader:
        for line in reader:
            ex = json.loads(line)
            coref_dict[ex['doc_key']]={}
            coref_dict[ex['doc_key']]['clusters'] = ex['clusters']
    with open(source, 'r') as reader, open(target, 'w') as writer:
        total_cnt = 0

        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']

                assert evt_type in ontology_dict

                input_template, output_template, context_tag_trigger,offset = create_instance_normal(ex, ontology_dict,
                                                                                              index=i,
                                                                                              tokenizer=tokenizer,
                                                                                              id2entity=id2entity)

                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) + 1, max_tgt)
                assert max_tgt <= MAX_TGT_LENGTH

                input_tokens = tokenizer.encode_plus(input_template, context_tag_trigger,
                                                     add_special_tokens=True,
                                                     add_prefix_space=True,
                                                     max_length=MAX_LENGTH,
                                                     truncation='only_second',
                                                     padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template,
                                                   add_special_tokens=True,
                                                   add_prefix_space=True,
                                                   max_length=MAX_TGT_LENGTH,
                                                   truncation=True,
                                                   padding='max_length')

                input_metion, sentence_dict, coref_dict_change = get_devmetion_mean(
                    ontology_dict[evt_type]['role_types'], ontology_dict[evt_type], input_template, context_tag_trigger,
                    ex, offset, coref_dict[ex['doc_id']])
                input_entity = get_entity(input_metion, input_tokens['input_ids'])

                processed_ex = {
                    'event_idx': i,
                    'doc_key': ex['doc_id'],
                    'input_token_ids': input_tokens['input_ids'],
                    'input_attn_mask': input_tokens['attention_mask'],
                    'tgt_token_ids': tgt_tokens['input_ids'],
                    'tgt_attn_mask': tgt_tokens['attention_mask'],
                    'input_metion': input_metion,
                    'input_entity': input_entity,
                    # 'output_entdict': output_entdict,
                    'sentence_dict':sentence_dict,
                    'coref_dict': coref_dict_change,
                }
                # print(input_template)
                # print(context_tag_trigger)
                tokens = tokenizer.convert_ids_to_tokens(input_tokens['input_ids'])
                # print(tokens)
                # assert 1==0

                # tokens = tokenizer.decode(processed_ex["input_token_ids"], skip_special_tokens=True)

                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))



#use
def  get_data_tag(source=None, target=None, tokenizer=None, trigger_dis=40, dataset='wikievents',coref_target=None):
    # in this function, we try to convert the input original data file into:
    # For each event:
    #   'event_idx'
    #   'doc_key'
    #   'input_token_ids':template + context with trigger highlighted
    #   'input_attn_mask'
    #   'tgt_token_ids': gold template
    #   'tgt_attn_mask'

    #   'compare_token_ids': template + context with trigger and arguments of other events highlighted
    #   'compare_attn_mask'
    #
    #   'input_mask' : 1 at arguments for the current event else 0 (input_token_ids)
    #   'compare_mask': 1 at arguments for the current event else 0 (compare_token_ids)
    ontology_dict = load_ontology(dataset)
    max_tokens = 0
    max_tgt = 0
    print(f"source:{source}")
    print(f"target:{target}")
    print(f'trigger dis:{trigger_dis}')
    print(f'dataset:{dataset}')
    coref_dict={}
    with open(coref_target, 'r') as reader:
        for line in reader:
            ex=json.loads(line)
            coref_dict[ex['doc_key']] = {}
            coref_dict[ex['doc_key']]['clusters'] = ex['clusters']
    with open(source, 'r') as reader, open(target, 'w') as writer:
        total_cnt = 0
        cnt = 0
        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None

            event_range = {}
            for i in range(len(ex['event_mentions'])):
                # if len(ex['event_mentions'][i]['arguments']) > 0:
                start = ex['event_mentions'][i]["trigger"]['start']
                end = ex['event_mentions'][i]["trigger"]['end']
                event_range[i] = {'start': start, 'end': end}
            events = event_range.keys()

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']

                assert evt_type in ontology_dict

                close_events = list(
                    filter(lambda x: abs(event_range[x]['start'] - event_range[i]['start']) <= trigger_dis and x != i,
                           events))  # events whose triggers are close to the current trigger
                if len(close_events):
                    cnt += 1
                input_template, output_template, context_tag_trigger, context_tag_trigger_mask,offset = create_instance_tag(
                    ex, ontology_dict, index=i, close_events=close_events, tokenizer=tokenizer, id2entity=id2entity)

                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) + 1, max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH

                input_tokens = tokenizer.encode_plus(input_template, context_tag_trigger,
                                                     add_special_tokens=True,
                                                     add_prefix_space=True,
                                                     max_length=MAX_LENGTH,
                                                     truncation='only_second',
                                                     padding='max_length'
                                                     )

                tgt_tokens = tokenizer.encode_plus(output_template,
                                                   add_special_tokens=True,
                                                   add_prefix_space=True,
                                                   max_length=MAX_TGT_LENGTH,
                                                   truncation=True,
                                                   padding='max_length')

                input_mask = [0] * (1 + len(input_template) + 2) + context_tag_trigger_mask + [0] * (
                            MAX_LENGTH - 3 - len(context_tag_trigger_mask) - len(input_template))
                input_mask = input_mask[:MAX_LENGTH]

                input_metion,sentence_dict,coref_dict_change = get_devmetion_mean(
                    ontology_dict[evt_type]['role_types'],ontology_dict[evt_type], input_template, context_tag_trigger, ex,offset,coref_dict[ex['doc_id']])

                input_entity= get_entity(input_metion, input_tokens['input_ids'])


                processed_ex = {
                    'event_idx': i,
                    'doc_key': ex['doc_id'],
                    'input_token_ids': input_tokens['input_ids'],
                    'input_attn_mask': input_tokens['attention_mask'],
                    'tgt_token_ids': tgt_tokens['input_ids'],
                    'tgt_attn_mask': tgt_tokens['attention_mask'],
                    'input_mask': input_mask,
                    'input_metion': input_metion,
                    'input_entity': input_entity,
                    'sentence_dict':sentence_dict,
                    'coref_dict':coref_dict_change,
                }

                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'has close events: {cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))
