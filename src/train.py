import logging

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from model.gam import GAM
from data_utils.get_data import get_data_tag, get_data_normal

logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BartTokenizer, BartConfig
from data_utils.data import IEDataset, my_collate_training, my_collate

from tqdm import tqdm


class score_args:
    def __init__(self, gen_file, test_file, coref_file) -> None:
        self.gen_file = gen_file
        self.test_file = test_file
        self.coref_file = coref_file
        self.dataset = "wikievents"
        self.coref = True
        self.head_only = True


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

    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    config = BartConfig.from_pretrained('./bart-large')
    tokenizer = BartTokenizer.from_pretrained('./bart-large')
    tokenizer.add_tokens([' <arg>', ' <tgr>', ' <tag>', ' </tag>'])
    model = GAM(config, tokenizer)
    model.resize_token_embeddings()
    device = f'cuda:{args.gpus}'
    model.to(device)

    if args.dataset == "wikievents":
        if args.use_info:
            source = './data/wikievents/train_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/train_no_ontology.jsonl'
        target = f'./{args.data_file}/train_data.jsonl'
        coref_target = f'./data/wikievents/coref/train.jsonlines'
    elif args.dataset == "rams":
        if args.use_info:
            source = './data/rams/train_ner.jsonlines'
        else:
            source = './data/rams/train_ner.jsonlines'
        target = f'./{args.data_file}/train_data.jsonl'
        coref_target = f'./data/rams/coref/train.jsonlines'
    get_data_tag(source=source, target=target, tokenizer=tokenizer, trigger_dis=args.trg_dis,dataset=args.dataset,coref_target=coref_target)#
    train_dataset = IEDataset(target)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=my_collate_training,#
                                  batch_size=args.train_batch_size,
                                  shuffle=True)

    if args.dataset == "wikievents":
        if args.use_info:
            source = './data/wikievents/dev_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/dev_no_ontology.jsonl'
        coref_target = f'./data/wikievents/coref/dev.jsonlines'
    elif args.dataset == "rams":
        if args.use_info:
            source = './data/rams/dev_ner.jsonlines'
        else:
            source = './data/rams/dev_ner.jsonlines'
        coref_target = f'./data/rams/coref/dev.jsonlines'
    target = f'./{args.data_file}/dev_data_normal.jsonl'
    get_data_normal(source=source, target=target, tokenizer=tokenizer, dataset=args.dataset,coref_target=coref_target)
    eval_dataset = IEDataset(target, tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, num_workers=2,
                                 collate_fn=my_collate,
                                 batch_size=args.eval_batch_size,
                                 shuffle=True)

    train_len = len(train_dataloader)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // train_len // args.accumulate_grad_batches + 1
    else:
        t_total = train_len // args.accumulate_grad_batches * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    print('--------------------------------------------------------------------')
    min_eval_loss = 1000
    for epoch in range(args.num_train_epochs):
        print("start training")
        pbar = tqdm(total=len(train_dataloader))
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step and step % args.accumulate_grad_batches == 0 or step == len(train_dataloader) - 1:
                clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            inputs = {
                "input_ids": batch["input_token_ids"].to(device),
                "attention_mask": batch["input_attn_mask"].to(device),
                "decoder_input_ids": batch['tgt_token_ids'].to(device),
                "decoder_attention_mask": batch["tgt_attn_mask"].to(device),
                'input_metion': batch["input_metion"].to(device),
                'input_entity': batch["input_entity"],
                'sentence_dict': batch["sentence_dict"],
                'coref_dict': batch["coref_dict"],
                "task": 0,
            }
            outputs, encoder_last_hidden_state = model(**inputs)
            loss = outputs[0]
            loss1 = torch.mean(loss)
            loss = (loss1) / args.accumulate_grad_batches
            loss.backward()

            pbar.update(1)

            pbar.set_postfix({'loss': float(loss)})

        print("start evaluating on evalset")
        model.eval()
        avg_loss = []
        pbar = tqdm(total=len(eval_dataloader))
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_dataloader)):
                inputs = {
                    "input_ids": batch["input_token_ids"].to(device),
                    "attention_mask": batch["input_attn_mask"].to(device),
                    "decoder_input_ids": batch['tgt_token_ids'].to(device),
                    "decoder_attention_mask": batch["tgt_attn_mask"].to(device),
                    'input_metion': batch["input_metion"].to(device),
                    'input_entity': batch["input_entity"],
                    'sentence_dict': batch["sentence_dict"],
                    'coref_dict': batch["coref_dict"],
                    "task": 0
                }

                outputs, _ = model(**inputs)
                loss = outputs[0]
                loss = torch.mean(loss)
                avg_loss.append(loss)
                pbar.update(1)
        avg_loss = sum(avg_loss) / len(avg_loss)
        print(f"avg_loss :{avg_loss}")
        if avg_loss < min_eval_loss:
            min_eval_loss = avg_loss
            print(f"new better ckpt {epoch}")
            save_dir = f'./checkpoints/{args.ckpt_name}/best_epoch.ckpt'

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir)


if __name__ == "__main__":
    main()
