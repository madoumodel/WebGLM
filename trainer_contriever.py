from transformers import RobertaTokenizer, RobertaModel, AutoModelWithLMHead, AutoTokenizer, Trainer, AutoModel, BertLMHeadModel
from datasets.load import load_dataset, load_from_disk
import torch, os, sys, time, random, json
from rouge_score.rouge_scorer import RougeScorer
import logging
import datetime

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from contriever.src.contriever import Contriever

'''
arguments
'''
logging_name = "contriever" + "2-23-1"
dataset_path = "data"
pretrain_tokenizer = "facebook/mcontriever-msmarco"
model_path = "facebook/mcontriever-msmarco"
load_checkpoint = "/mnt/vepfs/workspace/xuyifan/WebGLM/ckpt/ckpt_contriever/20230218-174221/step-869-epoch-0.ckpt" #optional
save_checkpoint = "ckpt/ckpt_contriever/%s"%(time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
max_epoch = 5



def get_logging(logging_name):
    """

    :param logging_name:
    :return: logger

    记录日志


    """
    #查找或创建指定名称的logger，若没有则会创建
    logger = logging.getLogger("train-contriever")
    #日志输出到的文件名和文件模式
    logging.basicConfig(filename='logs/train_contriever/'+logging_name+'.log',filemode='a')
    #日志等级info
    logger.setLevel(logging.INFO)
    
    rf_handler = logging.StreamHandler(sys.stderr)#默认是sys.stderr
    rf_handler.setLevel(logging.DEBUG) 
    #rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
 
    f_handler = logging.FileHandler('error.log')
    f_handler.setLevel(logging.ERROR)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
 
    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    return logger


#logger = get_logging("contriever" + str(time.time()))
logger = get_logging(logging_name)

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(x: torch.tensor):
    if not dist.is_initialized():
        return x
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather

class QuestionReferenceDensity_fromContriever(torch.nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.question_encoder = Contriever.from_pretrained(model_path)
        self.reference_encoder = Contriever.from_pretrained(model_path)

        total = sum([param.nelement() for param in self.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
        
    def forward(self, question, pos, neg):
        global temp, device
        
        cls_q = self.question_encoder(**question)
        cls_r_pos = self.reference_encoder(**pos)
        cls_r_neg = self.reference_encoder(**neg)
        cls_q /= temp
        
        bsz = cls_q.shape[0]
        
        kemb = torch.cat([cls_r_pos, cls_r_neg]) # [bs * 2, emb_dim]
        gather_kemb = gather(kemb) # [bs * 2 * word_size, emb_dim]
        
        scores = torch.matmul(cls_q, torch.transpose(gather_kemb, 0, 1)) # [bs, bs * 2 * word_size]
        
        labels = torch.arange(0, bsz, dtype=torch.long, device=device)
        labels = labels + dist.get_rank() * len(kemb)

        
        
        loss = torch.nn.functional.cross_entropy(scores, labels)

        return loss
        


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )

train_set = load_from_disk(dataset_path + "/train").shuffle(seed=42)
test_set = load_from_disk(dataset_path + "/test").shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer)

def collate(data):
    question = tokenizer([item["question"] for item in data], return_tensors="pt", padding=True, truncation=True)
    positive_reference = tokenizer([item["positive"] for item in data], return_tensors="pt", padding=True, truncation=True)
    negative_reference = tokenizer([item["negative"] for item in data], return_tensors="pt", padding=True, truncation=True)

    for key in question: question[key] = question[key].to(device)
    for key in positive_reference: positive_reference[key] = positive_reference[key].to(device)
    for key in negative_reference: negative_reference[key] = negative_reference[key].to(device)

    return question, positive_reference, negative_reference

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, batch_size=32, collate_fn=collate, sampler=sampler)
test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate, sampler=sampler)
total_step = len(train_loader) * max_epoch

model = QuestionReferenceDensity_fromContriever(model_path)

model = model.to(device)
if load_checkpoint:
    ckpt = torch.load(load_checkpoint)
    model.load_state_dict({key.replace('module.', ''): ckpt[key] for key in ckpt.keys()})

model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
opt = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
scheduler_args = {
    "warmup": int(total_step / 1000 * 25), #2.5% warm-up
    "total": total_step,
    "ratio": 0.0,
}
scheduler = WarmupLinearScheduler(opt, **scheduler_args)
temp = 0.05

def eval(eval_steps = -1):
    print("EVAL ...")
    model.eval()
    with torch.no_grad():
        total_acc = 0
        if eval_steps != -1:
            test_loader = random.choice(test_loader, k = eval_steps)
        total_step = len(test_loader)
        for q, pos, neg in test_loader:
            tot_cr = model.num_correct(*model(q, pos, neg))
            total_acc += tot_cr

        logger.info("EVALUATION, Acc: %10.6f , Step: %4d"%(total_acc / len(test_set),total_step))
    
def save(name):
    os.makedirs(save_checkpoint, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_checkpoint, "%s.ckpt"%(name)))
    logger.info("Sussessfully saved " + "%s.ckpt"%(name))

def train(max_epoch = 5, eval_step = 20, save_step = 500, print_step = 10):
    step = 0
    total_step = len(train_loader) * max_epoch
    for epoch in range(0, max_epoch):
        logger.info("EPOCH %d" % epoch)
        for q, pos, neg in train_loader:
            model.train()
            step += 1

            loss = model(q, pos, neg)
            
            if step % print_step == 0 and dist.get_rank() == 0:
                logger.info("Step %4d/%4d, Loss: %10.6f"%(step, total_step, loss))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            
            model.zero_grad()
            #if step % eval_step == 0:
                #eval(eval_steps = 5)

            if step % save_step == 0:
                if dist.get_rank() == 0:
                    save("step-%d"%(step))
            
        if dist.get_rank() == 0:
            save("step-%d-epoch-%d"%(step, epoch))
        #eval()
    
if __name__ == "__main__":
    train(max_epoch = max_epoch)

