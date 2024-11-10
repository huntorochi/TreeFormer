import os
from torch.nn.functional import interpolate
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, \
    MeanDice
from monai.transforms import (
    Compose,
    AsDiscreted,
)
import torch
from torch.nn.utils import clip_grad_norm
from inference import relation_infer
import gc
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from monai.inferers import Inferer, SimpleInferer
from ignite.metrics import Metric
from monai.transforms import Transform
from ignite.engine import Engine, EventEnum
from monai.engines.utils import (
    GanKeys,
    IterationEvents,
    default_make_latent,
    default_metric_cmp_fn,
    default_prepare_batch,
)

from utils import get_total_grad_norm


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):
    def __init__(
            self,
            device: torch.device,
            max_epochs: int,
            train_data_loader: Union[Iterable, DataLoader],
            network: torch.nn.Module,
            optimizer: Optimizer,
            loss_function: Callable,
            epoch_length: Optional[int] = None,
            non_blocking: bool = False,
            prepare_batch: Callable = default_prepare_batch,
            iteration_update: Optional[Callable] = None,
            inferer: Optional[Inferer] = None,
            postprocessing: Optional[Transform] = None,
            key_train_metric: Optional[Dict[str, Metric]] = None,
            additional_metrics: Optional[Dict[str, Metric]] = None,
            metric_cmp_fn: Callable = default_metric_cmp_fn,
            train_handlers: Optional[Sequence] = None,
            amp: bool = False,
            event_names: Optional[List[Union[str, EventEnum]]] = None,
            event_to_attr: Optional[dict] = None,
            decollate: bool = True,
            optim_set_to_none: bool = False,
            last_epoch: Optional[int] = 1,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            train_handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            optim_set_to_none=optim_set_to_none,
        )
        self.last_epoch = last_epoch

    def _iteration(self, engine, batchdata):
        # print(batchdata[0][0])
        # print(len(batchdata))
        images, nodes, edges = batchdata[0][0], batchdata[0][1], batchdata[0][2]
        # ids = batchdata[3]

        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device, non_blocking=False)
        nodes = [node.to(engine.state.device, non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device, non_blocking=False) for edge in edges]
        target = {'nodes': nodes, 'edges': edges}

        self.network.train()
        self.optimizer.zero_grad()

        h, out = self.network(images)
        # 2 21 256   {pred_logits 2 21 2  pred_nodes 2 21 4}

        valid_token = torch.argmax(out['pred_logits'], -1)
        # print(out['pred_logits'].shape)
        # torch.Size([2, 20, 2])
        # print(out['pred_logits'])
        # tensor([[[ 0.2715,  0.1507],
        #          [ 0.1086, -0.5380],
        #          [ 0.7130, -0.3465],
        #          [ 0.6767, -0.0269],
        #          [ 0.4739,  0.0942],
        #          [ 0.3048, -0.4730],
        #          [ 0.6870,  0.7868],
        #          [-0.5074, -0.2536],
        #          [-0.4091,  0.3145],
        #          [-0.3005, -0.1388],
        #          [-0.0159, -0.4985],
        #          [ 0.1063, -0.0241],
        #          [ 0.2464,  0.1858],
        #          [ 0.0377,  0.7685],
        #          [-0.0859, -0.5718],
        #          [ 0.7740, -0.1000],
        #          [ 0.4407, -0.4973],
        #          [ 0.7442, -0.5126],
        #          [ 0.5167,  0.6227],
        #          [-0.3667,  0.4453]],
        #         [[-0.0240, -0.2004],
        #          [ 0.0391, -0.2693],
        #          [ 0.7106, -0.7233],
        #          [ 0.4748, -0.2278],
        #          [ 1.1952,  0.1263],
        #          [ 0.7627, -0.6092],
        #          [ 0.6753,  0.6909],
        #          [ 0.0830,  0.0448],
        #          [ 0.0669, -0.6275],
        #          [-0.8358, -0.5124],
        #          [-0.0477, -0.4176],
        #          [ 0.3626, -0.5934],
        #          [-0.1255,  0.1827],
        #          [-0.3929,  0.2866],
        #          [-0.0838, -0.0377],
        #          [ 0.6959, -0.2151],
        #          [ 0.6179, -0.6202],
        #          [ 0.9192, -0.4004],
        #          [ 0.5195,  0.8280],
        #          [ 0.1451,  0.3656]]], device='cuda:0', grad_fn=<AddBackward0>)
        # print(valid_token.shape)
        # torch.Size([2, 20])
        # tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]],
        #        device='cuda:0')

        # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5
        # 这个输出的就是batch里面每一个的预测值

        # print('valid_token number', valid_token.sum(1))

        # pred_nodes, pred_edges = relation_infer(h, out, self.network.relation_embed)

        losses = self.loss_function(h, out, target, engine.state.epoch, engine.state.max_epochs, self.last_epoch)
        # {'class': tensor(0.5013, device='cuda:0', grad_fn=<NllLoss2DBackward>),
        # 'nodes': tensor(0.5748, device='cuda:0', grad_fn=<DivBackward0>),
        # 'boxes': tensor(1.2905, device='cuda:0', grad_fn=<DivBackward0>),
        # 'edges': tensor(0.7045, device='cuda:0', grad_fn=<NllLossBackward>),
        # 'cards': tensor(0.0500, device='cuda:0'),
        # 'total': tensor(7.2122, device='cuda:0', grad_fn=<AddBackward0>)}

        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()

        if 0.1 > 0:
            _ = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        else:
            _ = get_total_grad_norm(self.networm.parameters(), 0.1)

        self.optimizer.step()

        # gc.collect()
        # torch.cuda.empty_cache()

        return {"images": images, "points": nodes, "edges": edges, "loss": losses}


def build_trainer(train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, last_epoch, fp16=False):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x["loss"]["total"]
        ),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_interval=1,
            n_saved=1),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch
        )
    ]
    # train_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        # post_transform=train_post_transform,
        # key_train_metric={
        #     "train_mean_dice": MeanDice(
        #         include_background=False,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        last_epoch=last_epoch,
        # amp=fp16,
    )

    return trainer
