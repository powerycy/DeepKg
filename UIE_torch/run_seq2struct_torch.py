# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chaoyi Yuan, chaoyiyuan3721@gmail.com
# Status: Active
from statistics import mode
from uie.seq2struct.utils_torch import get_train_dataloader,set_logger,get_writer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from Config import Config
config_path = 'config.ini'
config = Config(config_path)
import torch
from transformers import T5ForConditionalGeneration
from uie.seq2struct.t5tokenizer import T5BertTokenizer
tokenizer = T5BertTokenizer.from_pretrained(config.model_path)
model = T5ForConditionalGeneration.from_pretrained(config.model_path)
from transformers import AdamW,get_linear_schedule_with_warmup
import logging
import math
from uie.evaluation.sel2record import evaluate_extraction_results
logger = logging.getLogger(__name__)
to_add_special_token = list()
import uie.evaluation.constants as constants
for special_token in [constants.type_start, constants.type_end, constants.text_start, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
    if special_token not in tokenizer.get_vocab():
        to_add_special_token += [special_token]
tokenizer.add_special_tokens(
    {"additional_special_tokens": tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token}
)
out_put_dir = config.out_put_dir
logging_dir = config.log_path
set_logger(out_put_dir)
tr_loss = config.tr_loss
logging_loss = config.logging_loss
global_steps = config.global_steps
metric_for_best_model = config.metric_for_best_model
save_model_path = config.model_save_path
model.resize_token_embeddings(len(tokenizer))
device = config.device
from uie.seq2struct.utils_torch import load_eval_tasks
train_dataloader = get_train_dataloader(
    model=model,
    tokenizer=tokenizer,
)
max_target_length = config.max_target_length
epochs = config.epochs
logging_steps = config.logging_steps
learning_rate = config.learning_rate
writer_type = config.writer_type
num_update_steps_per_epoch = len(train_dataloader)
max_steps = epochs * num_update_steps_per_epoch
num_warmup_steps = config.warmup_ratio * max_steps
model = model.to(device)
from uie.seq2struct.utils_torch import better_print_multi
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
writer = get_writer(logging_dir,writer_type)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps,max_steps,last_epoch = -1)
def logging_lr_loss(tr_loss,logging_loss):
    cur_lr = scheduler.get_last_lr()[-1]
    cur_loss = (tr_loss - logging_loss) / logging_steps
    # writer.add_scalar("lr", cur_lr, global_steps)
    # writer.add_scalar("loss", cur_loss, global_steps)
    logger.info(f"global_steps {global_steps}/{max_steps}"
                    f" - lr: {cur_lr:.10f}  loss: {cur_loss:.10f}")
def test(model, tokenizer):
    eval_tasks = load_eval_tasks(model=model, tokenizer=tokenizer)

    eval_overall_results, _ = eval_all_tasks(
        eval_tasks=eval_tasks,
        model=model,
        tokenizer=tokenizer,
        generate_max_length = max_target_length,
    )
    for line in better_print_multi(eval_overall_results).split('\n'):
        logger.info(line)
def evaluate(model, tokenizer, data_loader, generate_max_length, eval_instances,
             sel2record, eval_match_mode):
    """ Evaluate single task """

    model.eval()
    
    to_remove_token_list = list()
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()

    # Generate SEL using Trained Model
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:

            outputs = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_length=generate_max_length,
                # use_faster=True,
                return_dict_in_generate = True,
                # output_scores = True,
                use_cache=True
            )

            # Convert Token id to Token String
            outputs = tokenizer.batch_decode(outputs['sequences'],
                                            clean_up_tokenization_spaces=False,
                                            skip_special_tokens=False)

            preds = [postprocess_text(output) for output in outputs]
            all_preds.extend(preds)

    assert len(all_preds) == len(eval_instances)

    # Parsing SEL to Record
    all_records = []
    for predicted_sel, instance in zip(all_preds, eval_instances):
        record = sel2record.sel2record(pred=predicted_sel,
                                       text=instance['text'],
                                       tokens=instance['tokens'])
        all_records += [record]

    task_metrics = evaluate_extraction_results(eval_instances,
                                               all_records,
                                               eval_match_mode=eval_match_mode)

    prediction = {
        'record': all_records,
        'sel': all_preds,
        'metric': task_metrics
    }

    return task_metrics, prediction
def eval_all_tasks(eval_tasks, model, tokenizer, generate_max_length):
    """ Evaluate all tasks """
    eval_overall_results = dict()
    eval_overall_predictions = dict()
    for task_name, eval_task in eval_tasks.items():
        # Evaulate single task
        logger.info(f"Evaluate {task_name} ...")
        eval_results, eval_prediction = evaluate(
            model=model,
            tokenizer=tokenizer,
            data_loader=eval_task.dataloader,
            generate_max_length=generate_max_length,
            eval_instances=eval_task.val_instances,
            sel2record=eval_task.sel2record,
            eval_match_mode=eval_task.config.eval_match_mode,
        )

        for metric_name in eval_task.metrics:
            metric_key = f"{task_name}:{metric_name}"
            eval_overall_results[metric_key] = eval_results[metric_name]

        eval_overall_predictions[task_name] = eval_prediction

    sum_metric = sum(eval_overall_results.values())
    number_metric = len(eval_overall_results.values())
    eval_overall_results['all-task-ave'] = sum_metric / float(number_metric)

    return eval_overall_results, eval_overall_predictions
def math_ceil(x, y):
    return math.ceil(x / float(y))

    
def train(tr_loss,global_steps,logging_loss):
    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Total optimization steps = {max_steps}")
    best_score = 0.0
    for epoch in range(epochs):
        eval_tasks = load_eval_tasks(model=model, tokenizer=tokenizer)
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['decoder_attention_mask'] = batch['decoder_attention_mask'].to(device)
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            global_steps += 1
            if global_steps % logging_steps == 0:
                logging_lr_loss(tr_loss,logging_loss)
                logging_loss = tr_loss
        logger.info(f"********** Running evaluating **********")
        logger.info(f"************* Epoch {epoch} ***********")
        eval_overall_results, _ = eval_all_tasks(
                eval_tasks=eval_tasks,
                model=model,
                tokenizer=tokenizer,
                generate_max_length=max_target_length,
            )
        for line in better_print_multi(eval_overall_results).split('\n'):
                logger.info(line)
        current_score = eval_overall_results[metric_for_best_model]
        if current_score > best_score:
                logger.info("********** Saving Model **********")
                best_score = current_score
                torch.save(model.state_dict(),save_model_path)
def main():
    logger.info("**********************************************")
    train(tr_loss,global_steps,logging_loss)

if __name__ == "__main__":
    main()