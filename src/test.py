import logging

import torch
from torch.utils.data import DataLoader

from data_utils.get_data import get_data_normal
from genie.scorer_class import scorer, scorer_rams
from model.gam import GAM

logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed

from transformers import BartTokenizer, BartConfig
from data_utils.data import IEDataset, my_collate

from tqdm import tqdm
import json


class score_args:
    def __init__(self, gen_file, test_file, coref_file, score_th=0, dataset="KAIROS") -> None:
        self.gen_file = gen_file
        self.test_file = test_file
        self.coref_file = coref_file
        self.score_th = score_th
        self.dataset = dataset
        self.coref = True
        self.head_only = True
        if dataset=='rams':
            self.coref = False
            self.head_only = False


def main():
    args = parse_arguments()
    print(args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Training/evaluation parameters %s", args)

    set_seed(args.seed)
    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    config = BartConfig.from_pretrained('./bart-large')
    tokenizer = BartTokenizer.from_pretrained('./bart-large')
    tokenizer.add_tokens([' <arg>', ' <tgr>', ' <tag>', ' </tag>'])
    model = GAM(config, tokenizer)
    model.resize_token_embeddings()
    device = f'cuda:{args.gpus}'
    model.to(device)

    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt, map_location=model.device)['state_dict'])

    file_name = 'test'
    if args.eval_on_dev:
        file_name = 'dev'
    if args.dataset == "wikievents":
        if args.use_info:
            source = f'./data/wikievents/{file_name}_info_no_ontology.jsonl'
        else:
            source = f'./data/wikievents/{file_name}_no_ontology.jsonl'
        coref_target = f'./data/wikievents/coref/test.jsonlines'
    else:
        source = f'./data/rams/test_ner.jsonlines'
        coref_target = f'./data/rams/coref/test.jsonlines'
    target = f'./{args.data_file}/{file_name}_data.jsonl'
    get_data_normal(source=source, target=target, tokenizer=tokenizer, dataset=args.dataset,coref_target=coref_target)
    eval_dataset = IEDataset(target, tokenizer=tokenizer)

    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=my_collate,
                                 batch_size=args.eval_batch_size,
                                 shuffle=False)

    pbar_et = tqdm(total=len(eval_dataloader))
    result_dir = (args.load_ckpt).replace(".ckpt", f"_{file_name}_predictions.jsonl")
    model.eval()
    with open(result_dir, 'w') as writer:
        for step, batch in enumerate(eval_dataloader):
            doc_key = batch['doc_key']  # list
            tgt_token_ids = batch['tgt_token_ids']

            input = {
                "input_ids":batch['input_token_ids'].to(device),
                'input_metion': batch['input_metion'].to(device),
                'input_entity': batch["input_entity"],
                'sentence_dict': batch["sentence_dict"],
                'coref_dict': batch["coref_dict"],
                     }
            sample_output, scores = model.generate(**input, do_sample=False, max_length=30, num_return_sequences=1,
                                                   num_beams=1, decoder_start_token_id=0)

            for idx in range(len(doc_key)):
                output_ids = sample_output[idx]
                tokens = tokenizer.convert_ids_to_tokens(output_ids)[1:-1]
                score = scores[idx][1:-1]
                output = tokenizer.decode(output_ids, skip_special_tokens=True)

                gold_output = tokenizer.decode(tgt_token_ids[idx], skip_special_tokens=True)

                pred = {
                    'doc_key': doc_key[idx],
                    'predicted': output,
                    'gold': gold_output,
                }
                writer.write(json.dumps(pred) + '\n')
            pbar_et.update(1)

    print("start scoring")
    if args.dataset == "wikievents":
        if args.use_info:
            test_file = f'data/wikievents/{file_name}_info_no_ontology.jsonl'
        else:
            test_file = f'data/wikievents/{file_name}_no_ontology.jsonl'
        coref_file = f'data/wikievents/coref/{file_name}.jsonlines'
        scorer(score_args(result_dir, test_file, coref_file, args.score_th, args.dataset))
    elif args.dataset=='rams':
        test_file = f'./data/rams/test_ner.jsonlines'
        coref_file = None
        scorer_rams(score_args(result_dir, test_file, coref_file, args.score_th, args.dataset))


if __name__ == "__main__":
    main()