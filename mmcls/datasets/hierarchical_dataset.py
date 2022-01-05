import warnings
from collections import OrderedDict

import numpy as np

from mmcls.core import average_performance, mAP, tpr_at_fprs
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from .base_dataset import BaseDataset
from .builder import DATASETS

v1_1_attrs = [
    'Male', 'Female', 'SexUnsure', 'UpperBodyLongSleeve', 'UpperBodyShortSleeve',
    'UpperBodyUnsure', 'LowerBodyTrousers', 'LowerBodyShorts', 'LowerBodyLongSkirt',
    'LowerBodyShortSkirt', 'LowerBodyUnsure', 'NoHats', 'Hats', 'Helmet', 'MotorHelmet',
    'HatUnsure', 'NoMask', 'Mask', 'MaskUnwear', 'MaskUnsure', 'Muffler', 'NoMuffler',
    'MufflerUnsure', 'NoGloves', 'Gloves', 'GlovesUnsure', 'BareFeet', 'Boots',
    'OtherShoes', 'ShoesUnsure', 'UpRight', 'Squat', 'Lie', 'PoseUnsure', 'Front',
    'Back', 'LeftSide', 'RightSide', 'OrientationUnsure', 'UpperTrunc', 'NoUpperTrunc',
    'LowerTrunc', 'NoLowerTrunc', 'NoOcclusion', 'SlightOcclusion', 'HeavyOcclusion',
    'NoSmoke', 'Smoke', 'SmokeOther', 'SmokeUnsure', 'NoPhone', 'Phone', 'PlayPhone',
    'PhoneUnsure', 'HandHoldSomething', 'HandHoldNothing', 'HandHoldUnsure',
]

GeneralAttribute_v1_1 = {v: idx for idx, v in enumerate(v1_1_attrs)}

version_map = {
    'v1.1': GeneralAttribute_v1_1
}


@DATASETS.register_module()
class HierarchicalDataset(BaseDataset):
    Male = 1

    def __init__(self,
                 file_type=[
                     dict(type='ce', max_len=3),
                     dict(type='bce', max_len=3, class_name=None, class_wise=True),
                 ],
                 eval_by_class=False,
                 fpr=(0.0001, 0.001, 0.01),
                 tpr_at_fpr=True,
                 version=None,
                 **kwargs,
                 ):
        assert file_type is not None
        assert isinstance(file_type, (list, tuple))
        assert isinstance(kwargs.get('ann_file', None), (list, tuple))
        assert len(file_type) == len(kwargs.get('ann_file', []))
        self.version_map = None
        if version is not None:
            assert version in version_map, f'Support version are {list(version_map.keys())}'
            self.version_map = version_map[version]
        self.eval_by_class = eval_by_class
        self.fpr = fpr
        self.tpr_at_fpr = tpr_at_fpr
        self.file_type = file_type
        super(HierarchicalDataset, self).__init__(**kwargs)

    def _combine(self, name_labels):
        base = name_labels[0]
        for idx in range(1, len(self.file_type)):
            assert len(base) == len(name_labels[idx])
        name_label = dict()
        for key, value in base.items():
            for idx in range(1, len(self.file_type)):
                cvalue = name_labels[idx][key]
                if not isinstance(value, list):
                    value = [value]
                if isinstance(cvalue, list):
                    value.extend(cvalue)
                else:
                    value.append(cvalue)
            name_label[key] = value
        return name_label

    def _collect_name_with_label(self):
        name_labels = dict()
        for idx in range(len(self.file_type)):
            name_labels[idx] = dict()
            file_type = self.file_type[idx]
            with open(self.ann_file[idx]) as f:
                for line in f.readlines():
                    line = line.strip().split(' ')
                    filename = line[0]
                    if file_type['type'] == 'ce':
                        label = line[-1]
                    elif file_type['type'] == 'bce':
                        max_len = file_type['max_len']
                        label = np.zeros(max_len)
                        if len(line) != 1:
                            if self.version_map is not None:
                                new_line = []
                                for idx2 in range(len(line[1:])):
                                    new_line.append(self.version_map[line[idx2 + 1]])
                                line = line[:1] + new_line
                            pos_inds = list(map(int, line[1:]))
                            pos_inds = [pind for pind in pos_inds if pind < max_len]
                            if pos_inds:
                                label[pos_inds] = 1
                        label = label.tolist()
                    else:
                        raise ValueError(f"Only support 'ce' and 'bce' file_type")
                    assert filename not in name_labels[idx], f'{filename}, {self.ann_file[idx]}'
                    name_labels[idx][filename] = label
        name_label = self._combine(name_labels)
        return name_label

    def load_annotations(self):
        name_label = self._collect_name_with_label()
        data_infos = []
        for filename, gt_label in name_label.items():
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate_ce(self,
                    results,
                    gt_labels,
                    metric='accuracy',
                    metric_options=None,
                    logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        # results = np.vstack(results)
        # gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results

    def evaluate_bce(self,
                     results,
                     gt_labels,
                     file_type,
                     metric='mAP',
                     metric_options=None,
                     logger=None,
                     ce_result=None,
                     **deprecated_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.

        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'thr': 0.5}

        if deprecated_kwargs != {}:
            warnings.warn('Option arguments for metrics has been changed to '
                          '`metric_options`.')
            metric_options = {**deprecated_kwargs}

        # if isinstance(metric, str):
        #     metrics = [metric]
        # else:
        #     metrics = metric
        if file_type['class_wise']:
            allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'C_wise_F1', 'C_wise_acc', 'OP', 'OR', 'OF1']
        else:
            allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        metrics = allowed_metrics
        eval_results = {}
        # results = np.vstack(results)
        # gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            if file_type['class_wise']:
                performance_keys = ['CP', 'CR', 'CF1', 'C_wise_F1', 'C_wise_acc', 'OP', 'OR', 'OF1']
            else:
                performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            metric_options['class_wise'] = file_type['class_wise']
            if self.tpr_at_fpr:
                info = tpr_at_fprs(results, gt_labels, fpr_value=self.fpr, class_names=file_type['class_name'],
                                   ce_res=ce_result)
                import logging
                logger = logging.getLogger('mmcls')
                logger.info(info)
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        res_end_idx = 0
        label_end_idx = 0
        gt_labels = self.get_gt_labels()
        results = np.vstack(results)
        final_eval_results = OrderedDict()
        for idx, file_type in enumerate(self.file_type):
            res_start_idx = 0 if idx == 0 else res_end_idx
            res_end_idx = res_end_idx + file_type['max_len']
            label_start_idx = 0 if idx == 0 else label_end_idx
            label_end_idx = label_end_idx + 1 if file_type['type'] == 'ce' \
                else label_end_idx + file_type['max_len']
            if file_type['type'] == 'ce':
                eval_results = self.evaluate_ce(
                    results[..., res_start_idx: res_end_idx],
                    gt_labels[..., label_start_idx: label_end_idx],
                    metric='accuracy',
                    metric_options=None,
                    logger=logger
                )
                ce_results = gt_labels[..., label_start_idx: label_end_idx]  # 0: face, 1: hand, N*2
            elif file_type['type'] == 'bce':
                bce_pd = results[..., res_start_idx: res_end_idx]
                if not self.eval_by_class:
                    ce_results = None
                eval_results = self.evaluate_bce(
                    bce_pd,
                    gt_labels[..., label_start_idx: label_end_idx],
                    file_type,
                    metric=['mAP', 'CP', 'CR', 'CF1', 'C_wise_F1', 'C_wise_acc', 'OP', 'OR', 'OF1'],
                    metric_options=None,
                    ce_result=ce_results,
                    logger=logger
                )
            else:
                raise TypeError(f"File type only support 'ce' and 'bce' but got {file_type['type']}")
            for key, value in eval_results.items():
                new_key = f'{idx}_{key}'
                final_eval_results[new_key] = value

        return final_eval_results
