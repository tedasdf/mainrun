from model.gpt import GPT, GPTConfig
from model.attention.attention import (
    AttnConfig,
    SparseAttnConfig
)
import os
from dotenv import load_dotenv
import wandb
from omegaconf import OmegaConf
from ptflops import get_model_complexity_info

import argparse
from model.tokenizer.BPETokenizer import BPETokenizer
from model.unet import GPUnetT, UnetGPTConfig
import copy
import  random, time
import json
from pathlib import Path
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog
from hyperparam_class import Hyperparameters 


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer




def main(cfg, test=True):
    # Convert the OmegaConf section into a normal dict
    hparams = OmegaConf.to_container(cfg.hyperparams, resolve=True)
    modelparams = OmegaConf.to_container(cfg.model_configs[hparams['model_architecture']], resolve=True)
    attnparams = OmegaConf.to_container(cfg.attn_configs[modelparams['attention_layer']], resolve=True)
    
    if not test:
        wandb.init(
            project="gpt-from-scratch", 
            entity="arc_agi", 
            config=hparams   # <--- pass hyperparams to W&B
        )

    # Map into dataclass for your code
    args = Hyperparameters(**hparams)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    
    logger.log("hyperparameters_configured", **hyperparams_dict)



    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.context_length * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)
    
    ### Attention setup
    if modelparams['attention_layer'] == 'causal':
        attn = AttnConfig(
            d_model=modelparams['d_model'],
            block_size=args.context_length,
            dropout=modelparams['dropout'],
            **attnparams
        )

    elif modelparams['attention_layer'] == 'sparse':
        attn = SparseAttnConfig(
            d_model=modelparams['d_model'],
            block_size=args.context_length,
            dropout=modelparams['dropout'],
            **attnparams
        )
        


    #### Model setup
    if args.model_architecture == "gpt":
        cfg = GPTConfig(
                vocab_size=args.vocab_size,
                block_size=args.context_length,
                attn_config = attn,
                activation_function = 'gelu',
                **modelparams
            )
        model = GPT(cfg).to(device)
    elif args.model_architecture == "unet_gpt":
        cfg = UnetGPTConfig(
            vocab_size=args.vocab_size,
            block_size=args.context_length,
            attn_config = attn,
            activation_function='gelu',
            **modelparams
        )
        model = GPUnetT(cfg).to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {args.model_arhitecture}")
    

    ###############
    # MODEL FLOPS

    
    flops, params = get_model_complexity_info(model, (args.context_length,), as_strings=True,
                                        print_per_layer_stat=True)
    logger.log(f"model_info: \n FLOPs: {flops}, \nParameters: {params}")

    model_dict = vars(cfg).copy()
    model_dict['attn_config'] = vars(cfg.attn_config)
    logger.log("model_configured", **model_dict)

    logger.log(f"estimation of memory {model.memory_before_inference()} MB")

    ###############
    # MODEL PARAMS
    
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)



    ### Optimizer and Scheduler
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr , weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.context_length, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()

    scaler = GradScaler()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.context_length, args.batch_size, device)
            

            if args.amp_bool:
                with autocast():  # enables float16 for eligible ops
                    _, loss = model(xb, yb)

                # Backward with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (scale before unscale!)
                scaler.unscale_(opt)  # important for clip_grad_norm_
        
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                scaler.step(opt)
                scaler.update()

                # Scheduler step (unchanged)
                scheduler.step()
            else:
                _, loss = model(xb, yb)
            
                # l1_norm = sum(p.abs().sum() for p in model.parameters())
                # l2_norm = sum(p.pow(2).sum() for p in model.parameters())

                # loss = loss + l1_norm * L1 + l2_norm * L2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)
            if not test:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": step,
                    "train/elapsed_time": elapsed
                })
            
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)
                
                if not test:
                    wandb.log({
                        "val/step" : step,
                        "val/loss": val_loss,
                        "val/step": step,
                        "val/elapsed_time": elapsed
                    })
    if not test:
        artifact = wandb.Artifact("logs" , type="log")

        artifact.add_file(args.log_file)
        wandb.log_artifact(artifact)
        wandb.finish()

def merge_dotted_keys(base_dict, update_dict, target_path=None):
    """
    Merge keys with dots into nested dicts.
    If target_path is given, merge inside that nested dict.
    """
    import copy
    merged = copy.deepcopy(base_dict)
    
    # if target_path is provided, get the nested dict
    if target_path:
        d = merged
        for k in target_path:
            d = d.setdefault(k, {})
    else:
        d = merged
    
    for key, value in update_dict.items():
        parts = key.split(".")
        curr = d
        for p in parts[:-1]:
            curr = curr.setdefault(p, {})
        curr[parts[-1]] = value
    return merged


if __name__ == "__main__":

    import torch
    torch.cuda.empty_cache()
    load_dotenv(dotenv_path=".env")
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--sweep_config", type=str, help="Path to sweep YAML config")
    parser.add_argument("--orig_yaml", type=str, default="config/hyperparams.yaml")
    args = parser.parse_args()

 
    def sweep_train():
        orig_cfg = OmegaConf.load(args.orig_yaml) # defaults
        cfg = copy.deepcopy(orig_cfg)  # create a separate copy to modify

        with wandb.init() as run:
            print("RUNCONFIG")
            print(dict(run.config))
            
            print("Before")
            print(cfg)
            
            for key, val in dict(run.config).items():
                parts = key.split(".")  # split by all dots
                d = cfg
                # traverse down to the last dictionary
                for p in parts[:-1]:
                    d = d[p]
                d[parts[-1]] = val  # set the value

            print("After applying sweep")
            print(cfg)
            main(cfg , False)


    if not args.test:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        import utils
    
    if args.sweep:

        cfg = OmegaConf.load(args.sweep_config)
        # Convert to a plain dictionary
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        sweep_id = wandb.sweep(cfg_dict, project="gpt-from-scratch", entity="arc_agi")
        wandb.agent(sweep_id, function=sweep_train , count=50)
    else:
        cfg = OmegaConf.load(args.orig_yaml )
        main(cfg, args.test)
