import six
import logging
import torch
import torch.autograd
import os

import torch.nn.functional as F
from torch.autograd import Variable

import contextlib

from baseline.utils import verbose_output
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.pytorch.optz import OptimizerManager
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
logger = logging.getLogger('baseline')



def _add_to_cm(cm, y, pred):
    _, best = pred.max(1)
    yt = y.cpu().int()
    yp = best.cpu().int()
    cm.add_batch(yt.data.numpy(), yp.data.numpy())


@register_trainer(task='classify', name='default')
class ClassifyTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerPyTorch, self).__init__()
        self.clip = float(kwargs.get('clip', 5))
        self.labels = model.labels
        self.gpus = int(kwargs.get('gpus', -1))
        if self.gpus == -1:
            self.gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        self.optimizer = OptimizerManager(model, **kwargs)
        self.model = model
        if self.gpus > 0:
            self.crit = model.create_loss().cuda()
            if self.gpus > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model.cuda()
        else:
            logger.warning("Requested training on CPU.  This will be slow.")
            self.crit = model.create_loss()
            self.model = model
        
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        self.convat = kwargs.get('convat', False)
        


    def _make_input(self, batch_dict):
        if self.gpus > 1:
            return self.model.module.make_input(batch_dict)
        return self.model.make_input(batch_dict)

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])
    @staticmethod
    def _vat_loss(model, x,eps=1.5, ip=1,xi=1e-6):

 

        def _l2_normalize(d):
            d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
            d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
            return d


        with torch.no_grad():
            pred = F.softmax(model.context_forward(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5)
        d = _l2_normalize(d)

        for _ in range(ip):
            d = Variable(d, requires_grad=True)
            
            pred_hat = model.context_forward(x.detach() + xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred.detach(), reduction='batchmean')
            adv_distance.backward()
            d = d.grad.data.clone()
            model.zero_grad()

        # calc LDS
        r_adv = d * eps
        pred_hat = model.context_forward(x + r_adv.detach())
        logp_hat = F.log_softmax(pred_hat, dim=1)
        lds = F.kl_div(logp_hat, pred.detach(), reduction='batchmean')

        return lds

    @staticmethod
    def _convat(model, epsilon=1.5, num_iters=1,xi=1e-6):
        """
        logit --> output of the model 
        """
    
        def kl_div_with_logit(p_logit, q_logit):

            p = F.softmax(p_logit, dim=1)
            logp = F.log_softmax(p_logit, dim=1)
            logq = F.log_softmax(q_logit, dim=1)

            plogp = ( p *logp).sum(dim=1).mean(dim=0)
            plogq = ( p *logq).sum(dim=1).mean(dim=0)

            return plogp - plogq

        def _l2_normalize(d,norm_length):
            # shape(x) = (batch, num_timesteps, d)
            # Divide x by max(abs(x)) for a numerically stable L2 norm.
            # 2norm(x) = a * 2norm(x/a)
            # Scale over the full sequence, dims (1, 2)  
            alpha = torch.max(torch.abs(d),-1,keepdim=True).values + 1e-24
            #  l2_norm = alpha * tf.sqrt(
            #  tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
            l2_norm = alpha * torch.sqrt(
                torch.sum((d/alpha)**2, (1),keepdim=True)  + 1e-16
            )
            d_unit = d/ l2_norm
            return norm_length * d_unit
        #feature is context vector (batch, kernel_num*kernel_size)
        # find r 
    
        context_vec = model.context_vec
        d = torch.Tensor(context_vec.size()).normal_()
        for i in range(num_iters):
            #d = 1e-3 *_l2_normalize(mask_by_length(d,input_length))
            d = _l2_normalize(
                d , xi)
            d = Variable(d, requires_grad=True)
            y_hat = model.context_forward(context_vec.detach() + d)
            #print([x.grad for x in model.output.parameters()])
            delta_kl = kl_div_with_logit(model.logit.detach(), y_hat)
            delta_kl.backward()
            

            d = d.grad.data.clone()
            model.zero_grad()
        
        #d = _l2_normalize(d)
        d=_l2_normalize(d,epsilon)
        d = Variable(d)
        #r_adv = eps *d
        # compute lds
        y_hat = model.context_forward(context_vec + d.detach())
        #print([x.grad for x in model.output.parameters()])
        #y_hat = model(feature + r_adv.detach())
        delta_kl = kl_div_with_logit(model.logit.detach(), y_hat)
        return delta_kl  




    def _train(self, loader, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        epoch_loss = 0
        epoch_div = 0
        for batch_dict in pg(loader):
            self.optimizer.zero_grad()
            example = self._make_input(batch_dict)
            y = example.pop('y')
            
            if self.convat:

                pred = self.model(example)
                #vat_loss = self._vat_loss(self.model, self.model.context_vec)
                vat_loss = self._convat(self.model)
                
                loss = self.crit(pred, y) + vat_loss

                
            else:
                pred = self.model(example)
                loss = self.crit(pred, y)


            batchsz = self._get_batchsz(batch_dict)
            report_loss = loss.item() * batchsz
            epoch_loss += report_loss
            epoch_div += batchsz
            self.nstep_agg += report_loss
            self.nstep_div += batchsz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            _add_to_cm(cm, y, pred)
            self.optimizer.step()

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = epoch_loss / float(epoch_div)
        return metrics


    def _test(self, loader, **kwargs):
        self.model.eval()
        total_loss = 0
        total_norm = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        verbose = kwargs.get("verbose", None)
        output = kwargs.get('output')
        txts = kwargs.get('txts')
        handle = None
        line_number = 0
        if output is not None and txts is not None:
            handle = open(output, "w")
        
        for batch_dict in pg(loader):
            example = self._make_input(batch_dict)
            ys = example.pop('y')
            pred = self.model(example)
            loss = self.crit(pred, ys)
            if handle is not None:
                for p, y in zip(pred, ys):
                    handle.write('{}\t{}\t{}\n'.format(" ".join(txts[line_number]), self.model.labels[p], self.model.labels[y]))
                    line_number += 1
            batchsz = self._get_batchsz(batch_dict)
            total_loss += loss.item() * batchsz
            total_norm += batchsz
            _add_to_cm(cm, ys, pred)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)
        if handle is not None:
            handle.close()

        return metrics



@register_training_func('classify')
def fit(model, ts, vs, es, **kwargs):
    """
    Train a classifier using PyTorch
    :param model: The model to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs: See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) -- Stop after eval data is not improving. Default to True
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *optim* --
           Optimizer to use, defaults to `sgd`
        * *eta, lr* (``float``) --
           Learning rate, defaults to 0.01
        * *mom* (``float``) --
           Momentum (SGD only), defaults to 0.9 if optim is `sgd`
    :return:
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'pytorch', kwargs.get('basedir'))
    output = kwargs.get('output')
    txts = kwargs.get('txts')
    
    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('eatly_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)


    trainer = create_trainer(model, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns)

        if do_early_stopping is False:
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        test_metrics = trainer.test(es, reporting_fns, phase='Test', verbose=verbose, output=output, txts=txts)
    return test_metrics
