from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets 
import argparse
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset
parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth", action="store_true")
parser.add_argument("--per_device_train_batch_size", type=int, default=32)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
args = parser.parse_args()
def transform_labels(labels, wl_mapping):
    transformed_label = []
    for i in range(len(wl_mapping)):
        for j in wl_mapping[i]:
            if j in labels:
                transformed_label += [i]
    return transformed_label
class CustomDataset(Dataset):
    def __init__(self, file, tc_mapping):
        self.file = file
        self.tc_mapping = tc_mapping

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        text = self.file['text'][idx]
        label = self.file['label'][idx]
        label = transform_labels(label, self.tc_mapping)
        breakpoint()
        return text, label
if __name__ == "__main__":
    model_id = "cardiffnlp/tweet-topic-21-multi"
    split = "train_2021"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    ds = datasets.load_dataset("cardiffnlp/tweet_topic_multi")
    model.train()
    wl_mapping = {0: [7, 16]}
    dataset = CustomDataset(ds[split], wl_mapping )
    train_dataloader = DataLoader(
        dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            optimizer.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        torch.save(model.state_dict(), f"{args.output_dir}/model_{epoch}.pt")