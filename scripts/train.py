import logging
import os
import random
import sys
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import timm
import timm.optim

from CBRTiny import CBRTiny

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report)

                            
from key2med.data.loader import CheXpertDataLoader, ColorCheXpertDataLoader, StudiesDataLoader
 
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments. Type python train_bert.py --help to display help.
    """
    parser = argparse.ArgumentParser()
    # model args
    # timm.list_models(pretrained=True)
    parser.add_argument('--model', type=str, required=False, default='efficientnet_b0',
                        help='Pretrained model to initialize the training with.\n'
                             'Default german model: "bert-base-german-cased"\n'
                             'German Med-BERT: "smanjil/German-MedBERT"')
    parser.add_argument('--channel-in', type=int, default=3,
                        help='Number of channels to use as input (3 means copy and imagenet). (default 3)')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze model (not classificaiton).')
    parser.add_argument('--class-positive', type=str, required=False, default='Edema',
                        help='Class to one-vs-all classification (Edema, Atelectasis, Cardiomegaly, Consolidation, Pleural Effusion). Default Edema')
    
    parser.add_argument('--model-to-load-dir', type=str, required=False, default=None,
                        help='Directory to load already trained state.')
    parser.add_argument('--optim', type=str, required=False,default='AdamP',
                        help='Optimizer Default AdamP (AdaBelief,Adafactor,Adahessian,AdamP,AdamW,Lamb,Lars,MADGRAD,Nadam,NvNovoGrad,RAdam,RMSpropTF,SGDP')

    parser.add_argument('--data-dir', type=str, required=False, default='/home/admin_ml/NAMT/data/CheXpert-v1.0-small',
                        help='Path to dataset.')
    parser.add_argument('--output-dir', type=str, required=False, default=None,
                        help='Output directory for trained model, model checkpoints, prediction results and logging. Default: None (basePath/training_output/curTraining/)')
    parser.add_argument('--do-train', action='store_true',
                        help='Train on train split of tokenized data.')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs. Default 10.')
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='Set to low number for debugging. default =-1 (no limit)')
    parser.add_argument('--max-dataloader-size', type=int, default=None,
                        help='Set to low number for debugging. default = none (no limit)')
    parser.add_argument('--view', type=str, default='Frontal', #'Lateral
                        help='Which view to train on (Frontal, Lateral). default = Frontal')
    parser.add_argument('--batch-size', type=int, default=24,
                        help='Training batch size. Default: 24.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Num workers for dataloader. Default: 4.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. Default: 42.')
    
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        help='Learning rate Scheduler (linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup). Default: linear.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Training learning rate. Default: 1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='Training weight decay (AdamW). Default: 1e-6.')
    
    parser.add_argument('--do-eval', action='store_true',
                        help='Validate on validation split of tokenized data.')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Number of batches/steps to eval. Default 500.')
   
    # parser.add_argument('--fp16', action='store_true',
    #                     help='Train with fp16 precision.')
    parser.add_argument('--do-weight-loss-even', action='store_true',
                        help='If the loss shall be weighted according to the class imbalance of all classes.')
    parser.add_argument('--do-upsample', action='store_true',
                        help='If the positive class shall be upsampled by the CheXpertDataset.')    
    
    parser.add_argument('--no-cuda', action='store_true',
                        help='Train on CPU (for debug)')
    parser.add_argument('--do-early-stopping', action='store_true',
                        help='Apply early stopping.')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='How many eval steps to wait for improvement.')

    args = parser.parse_args()
    args.basePath = os.path.dirname(os.path.realpath(__file__))+os.sep
    
    if args.output_dir is None:
         args.output_dir = f'{args.basePath}training_output{os.sep}curTraining{os.sep}'
    else:
        if args.output_dir[-1] != os.sep: args.output_dir += os.sep
         
    if args.model_to_load_dir is not None:
        if args.model_to_load_dir[-1] != os.sep: args.model_to_load_dir += os.sep

    args.device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')   

    ###debug
    # print('!!!!!!! THERE ARE HARDCODED DEBUG SETTING CHECK SCRIPT IF UNWANTED !!!!!!!')
    # args.do_train = True
    # args.freeze = True
    # args.do_eval = True
    # args.max_steps=100
    # args.eval_steps=1
    
    # args.model = 'CBRTiny'
    return args

def save_obj_pkl(path, obj):
    with open( path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_pkl(path ):
    with open( path, 'rb') as f:
        return pickle.load(f)
    
def get_metric_dict(preds, y_true, y_pred):
 
    metric_dict = {'Num_samples':len(y_true)}
    metric_dict['Num'] = int(y_true.sum())
    metric_dict['Acc'] = accuracy_score(y_true, y_pred)
    metric_dict['bAcc'] = balanced_accuracy_score(y_true, y_pred)
    metric_dict['Precision'] =  precision_score(y_true=y_true, y_pred=y_pred, zero_division = 0)
    metric_dict['Recall'] =  recall_score(y_true=y_true, y_pred=y_pred, zero_division = 0)
    metric_dict['F1'] =  f1_score(y_true=y_true, y_pred=y_pred, zero_division = 0)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    #Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
    metric_dict['Sensitivity'] =  recall_score(y_true=y_true, y_pred=y_pred, zero_division = 0)
    metric_dict['Specificity'] =  recall_score(y_true=~(y_true>0), y_pred=~(y_pred>0), zero_division = 0)
    fpr, tpr, _ = roc_curve(y_true, preds[:,1])
    metric_dict['AUC'] = auc(fpr, tpr)

    return metric_dict
 
def eval_model(args, model, dataloader, dataset='valid'):
    
    model.eval()
    
    preds = []
    y_preds = []
    y_true = []
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(args.device)
        targets = targets.squeeze(dim=1).detach().cpu().numpy()
        y_true += list(targets)
        cur_preds = torch.nn.functional.softmax(model(inputs), dim=-1).detach().cpu().numpy()
        preds += list(cur_preds)
        y_preds += list( (cur_preds[:,1] > 0.5).astype(int))
        
    preds, y_preds, y_true =  np.asarray(preds), np.asarray(y_preds), np.asarray(y_true)
    metric_dict = get_metric_dict(preds, y_true, y_preds)
    with open(args.output_dir+f'results_{dataset}_{args.class_positive}.json', 'w', encoding='utf-8') as file:
        json.dump(metric_dict, file, indent=2)
    with open(args.output_dir+f'results_{dataset}_{args.class_positive}.pkl', 'wb') as file:
        pickle.dump([metric_dict, y_true, y_preds, preds], file)

         
def main():
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        filename=args.output_dir+'train.log',
        filemode='w',
    )
    
    g = None
    worker_init_fn = None
    if args.seed is not None:
        logger.info(f'Applying seed {args.seed}')
        #https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
   
    # dataloader = StudiesDataLoader(
    #     data_path="/home/admin_ml/NAMT/data/CheXpert-v1.0-small",
    #     batch_size=4,
    #     img_resize=224,
    #     splits="train_valid_test",
    #     channels=3,
    #     do_random_transform=True,
    #     use_cache=True,
    #     in_memory=False,
    #     max_size=10_000,
    #     plot_stats=False,
    #     n_workers=0,
    #     frontal_lateral_values = ['Frontal'], #trotzdem manhcml lateral ?? falsche metadaten?
    #     label_filter = [5],
    # )
    
    dataloader = ColorCheXpertDataLoader( #CheXpertDataLoader #for 1 ch
        data_path=args.data_dir,
        batch_size=args.batch_size,
        img_resize=224,
        splits="train_valid_test",
        channels=args.channel_in,
        do_random_transform=True,
        use_cache=False,
        in_memory=True,
        max_size= args.max_dataloader_size,
        use_upsampling = args.do_upsample,
        upsample_labels = [args.class_positive],
        plot_stats=False,
        n_workers=args.num_workers,
        frontal_lateral_values = [args.view], #trotzdem manhcml lateral ?? falsche metadaten?
        label_filter = [args.class_positive],
        uncertain_to_one = [args.class_positive],
        uncertain_to_zero = [],
    )

    
    if args.model == 'CBRTiny':
        model = CBRTiny(num_classes=2, channel_in=args.channel_in).to(args.device)
    else:
        #timm.list_models('eff*', pretrained=True)
        model = timm.create_model(args.model, num_classes=2, in_chans=args.channel_in, pretrained=True).to(args.device)
    
    if args.model_to_load_dir is not None:
        checkpoint = torch.load(osp.join(args.model_to_load_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.freeze:
        logger.info(f'Freezing {args.model}')
        for name, param in model.named_parameters():
            if 'classifier' not in name:    param.requires_grad = False
    

    optimizer = timm.optim.create_optimizer_v2(model,
                                            optimizer_name=args.optim,
                                            learning_rate=args.lr,
                                            weight_decay=args.wd)
     
    num_steps = args.num_epochs*len(dataloader.train) if args.max_steps<0 else args.max_steps #for Debug !
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.lr,
                                                    total_steps=num_steps)   

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
    
    writer = SummaryWriter(args.output_dir+os.sep+'runs')
    writer_dict = {
                    'epochs': [],#np.zeros(howOftenValid*howOftenRepeat,dtype=int),
                    'lr': [], #np.zeros(howOftenValid*howOftenRepeat),
                    'loss_train': [],#np.zeros(howOftenValid*howOftenRepeat),
                    'loss_valid': [],#np.zeros(howOftenValid*howOftenRepeat),
                    'walltime': [],#np.zeros(howOftenValid*howOftenRepeat)
                    }
    
    args.loss_weights = [1, 1]
    if args.do_weight_loss_even:
         #https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514/2
        num_total = len(dataloader.train.dataset.all_labels())
        num_pos = sum(dataloader.train.dataset.all_labels())
        num_neg = num_total-num_pos
        args.loss_weights = [1 - (x / num_total) for x in [ num_neg, num_pos ] ]
    
    args.loss_weights = torch.tensor(args.loss_weights).float().cuda()
    logger.info(f'LOSS WEIGHTS: {args.loss_weights}')
    loss_function = torch.nn.CrossEntropyLoss(weight=args.loss_weights)
    
    best_loss = np.inf
    eval_steps_since_last_better_model = 0
    if args.do_train:
        model.train()
        steps=0
        steps_since_last_eval=0
        logger.info('Start Training')
        for epoch in tqdm(range(args.num_epochs)):
            
            writer_dict['epochs'].append(epoch)
            writer.add_scalar('utils/epochs', epoch, steps)
    
            for batch in tqdm(dataloader.train, leave=False):
                steps += 1
                steps_since_last_eval +=1
                
                if steps > num_steps: break
 
                inputs, targets = batch
                inputs = inputs.to(args.device)
                targets = targets.squeeze(dim=1).long().to(args.device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                writer_dict['walltime'].append( time.time() )
                lr = optimizer.param_groups[0]['lr']
                writer_dict['lr'].append(lr)
                writer.add_scalar('utils/lr', lr, steps)
                loss = loss.detach().cpu().numpy()
                writer_dict['loss_train'].append(loss)
                writer.add_scalar('loss/train', loss, steps)

                if steps_since_last_eval >= args.eval_steps:
                    steps_since_last_eval = 0
                    if dataloader.validate is not None:
                        model.eval()
                        mean_loss = 0
                        for batch in dataloader.validate:
                            inputs, targets = batch
                            inputs = inputs.to(args.device)
                            targets = targets.squeeze(dim=1).long().to(args.device)
                            outputs = model(inputs)
                            #error when hits batch of size 1 becasue batch dim missing        
                            try:
                                mean_loss += loss_function(outputs, targets).detach().cpu().numpy()
                            except:
                                print()
                        mean_loss /= len(dataloader.validate)
                        writer_dict['loss_valid'].append(mean_loss)
                        writer.add_scalar('loss/valid', mean_loss, steps)
                        model.train()
                        if mean_loss < best_loss:
                            
                            best_loss = mean_loss
                            eval_steps_since_last_better_model = 0
                            
                            with open( args.output_dir+'best_loss.txt', 'w') as file:
                                print( f'loss {best_loss}\n step {steps}', file=file )

                            torch.save({
                                        'step': steps,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()
                                       }, args.output_dir+'best_model.pth')
                        else:
                            eval_steps_since_last_better_model += 1
                            
                    if args.do_early_stopping:
                        if eval_steps_since_last_better_model >= args.early_stopping_patience: break

                if steps >= num_steps: break
            
        with open( args.output_dir+'last_loss.txt', 'w') as file:
            print( f'loss {mean_loss}\n step {steps}', file=file )

        torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   }, args.output_dir+'last_model.pth')
            
        save_obj_pkl( args.output_dir+'tensorboard_writer.pkl', writer_dict )       
        writer.close()
        logger.info(f'End of training ')
        
    if args.do_eval:
        if dataloader.validate is not None:
            logger.info('Start evaluation valid')
            eval_model(args, model, dataloader.validate, dataset='valid')
            
        if dataloader.test is not None:
            logger.info('Start evaluation test')
            eval_model(args, model, dataloader.test, dataset='test')
        

if __name__ == '__main__':
    main()
